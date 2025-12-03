import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import Array, Float, Complex, Int
from typing import Optional

# Enable 64-bit precision (crucial for accurate physics/complex numbers)
jax.config.update("jax_enable_x64", True)

# ==============================================================================
# 1. Physics Utilities & Feature Map
# ==============================================================================

def fibonacci_sphere(n: int, dtype=jnp.float64) -> Float[Array, "n 3"]:
    k = jnp.arange(n, dtype=dtype) + 0.5
    phi = jnp.pi * (1.0 + jnp.sqrt(5.0))
    z = 1.0 - 2.0 * k / n
    r = jnp.sqrt(jnp.clip(1.0 - z*z, a_min=0.0))
    theta = phi * k
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    dirs = jnp.stack([x, y, z], axis=-1)
    return dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)

def normalize(v: Array, axis: int = 1, eps: float = 1e-12) -> Array:
    return v / (jnp.linalg.norm(v, axis=axis, keepdims=True) + eps)

class PolarLightConeFeatureMap(eqx.Module):
    """Generates Maxwell-compliant basis functions."""
    base_dirs_raw: Float[Array, "n_spectral 3"]
    omega: Float[Array, ""]
    n_spectral: int = eqx.field(static=True)

    def __init__(self, n_spectral: int, omega: float, key=None, init_jitter: float=0.0):
        self.n_spectral = int(n_spectral)
        self.omega = jnp.array(omega, dtype=jnp.float64)
        
        base = fibonacci_sphere(self.n_spectral)
        if init_jitter > 0.0 and key is not None:
            base = base + init_jitter * jax.random.normal(key, base.shape, dtype=jnp.float64)
            base = normalize(base)
        self.base_dirs_raw = base

    @property
    def kdirs_unit(self):
        return normalize(self.base_dirs_raw)

    def __call__(self, X: Float[Array, "N 3"]) -> Complex[Array, "F 6N"]:
        N = X.shape[0]
        kdirs = self.kdirs_unit 
        w = self.omega

        # Polarization basis
        I = jnp.eye(3, dtype=jnp.float64)
        P = I - kdirs[:, :, None] * kdirs[:, None, :] 
        S = jnp.array([[1.,0.],[0.,1.],[0.,0.]], dtype=jnp.float64)
        V = jnp.einsum('rij,jk->rik', P, S)
        
        e1 = normalize(V[:,:,0])
        v2 = V[:,:,1] - e1 * jnp.sum(V[:,:,1]*e1, axis=1, keepdims=True)
        e2 = normalize(v2)
        pols = jnp.stack([e1, e2], axis=1) # (r, 2, 3)

        # Coefficients
        k_vec = kdirs * w
        k_exp = k_vec[:, None, :] 
        cross_k_pi = jnp.cross(k_exp, pols, axis=-1)
        E = w * cross_k_pi
        B = jnp.cross(k_exp, cross_k_pi, axis=-1)
        coeff6 = jnp.concatenate([E, B], axis=-1).astype(jnp.complex128) # (r, 2, 6)

        # Phases
        phase = jnp.exp(1j * (X @ k_vec.T)) # (N, r)

        # Combine
        feat = coeff6[:, :, None, :] * phase.T[:, None, :, None] # (r, 2, N, 6)
        
        # Flatten: (F, 6N)
        return feat.reshape(self.n_spectral * 2, -1)

# ==============================================================================
# 2. Kernel & GP (Strict Template)
# ==============================================================================

class MaxwellKernel(eqx.Module):
    feature_map: PolarLightConeFeatureMap
    log_w: Float[Array, "F"]

    def __init__(self, n_spectral: int, omega: float, key: Optional[jax.Array] = None):
        self.feature_map = PolarLightConeFeatureMap(n_spectral, omega, key, init_jitter=0.01)
        # Initialize log variance to 0 (variance = 1)
        self.log_w = jnp.zeros(n_spectral * 2, dtype=jnp.float64)

    def __call__(self, X1: Array, X2: Array | None = None) -> Array:
        # 1. Get features
        Phi_1 = self.feature_map(X1) # (F, 6N)
        Phi_2 = Phi_1 if X2 is None else self.feature_map(X2)

        # 2. Weighted Inner Product
        w = jnp.exp(self.log_w).astype(jnp.complex128)
        
        # K = Phi(X1)^H @ W @ Phi(X2)
        Phi_1_w = Phi_1.conj().T * w[None, :] 
        return jnp.dot(Phi_1_w, Phi_2) # (6N, 6M)

class GaussianProcess(eqx.Module):
    num_data: Int = eqx.field(static=True)
    kernel: MaxwellKernel
    X: Array

    def __init__(self, kernel, X: Array):
        self.kernel = kernel
        self.X = X
        self.num_data = X.shape[0] * 6

    def posterior_mean(self, X_star: Array, y_star: Array, jitter: Float = 1e-6) -> Array:
        y_star = y_star.astype(jnp.complex128)
        K = self.kernel(self.X, self.X) + jitter * jnp.eye(self.num_data)
        K_s = self.kernel(X_star, self.X)
        
        L = jax.scipy.linalg.cholesky(K, lower=True)
        z = jax.scipy.linalg.solve_triangular(L, y_star, lower=True)
        alpha = jax.scipy.linalg.solve_triangular(L.conj().T, z, lower=False)
        
        return jnp.dot(K_s, alpha)

    def nlml(self, y: Array, noise: Float = 1e-6) -> Array:
        y = y.astype(jnp.complex128)
        K = self.kernel(self.X, self.X) + noise * jnp.eye(self.num_data)
        L = jax.scipy.linalg.cholesky(K, lower=True)
        
        z = jax.scipy.linalg.solve_triangular(L, y, lower=True)
        alpha = jax.scipy.linalg.solve_triangular(L.conj().T, z, lower=False)
        
        data_fit = jnp.real(jnp.dot(y.conj().T, alpha))[0,0]
        log_det = 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diag(L))))
        
        return 0.5 * data_fit + 0.5 * log_det + (self.num_data / 2.0) * jnp.log(2 * jnp.pi)

# ==============================================================================
# 3. Ground Truth Data Generation (Replicating PyTorch inputs)
# ==============================================================================


def compute_ground_truth(X: Array, EE0s: Array, k0_dirs: Array, omega: float):
    """
    Computes superposition of plane waves.
    X: (N, 3)
    EE0s: (Sources, 3) - Electric field amplitudes
    k0_dirs: (Sources, 3) - Propagation directions
    """
    # Normalize k directions
    k_dirs_norm = normalize(k0_dirs)
    
    # Precompute k vectors
    k_vecs = k_dirs_norm * omega # (Sources, 3)
    
    # Calculate Phase: exp(i * k . x)
    # X: (N, 3), k_vecs: (S, 3) -> (N, S)
    phases = jnp.exp(1j * jnp.dot(X, k_vecs.T))
    
    # Calculate B fields: B = (1/w) * (k x E) ?
    # Standard plane wave: B = khat x E (assuming c=1)
    # Or specifically strictly matching the Feature Map logic: B = khat x E
    BB0s = jnp.cross(k_dirs_norm, EE0s)
    
    # Sum over sources
    # E_total = sum(E0_s * phase_s)
    E_total = jnp.dot(phases, EE0s.astype(jnp.complex128)) # (N, 3)
    B_total = jnp.dot(phases, BB0s.astype(jnp.complex128)) # (N, 3)
    
    # Concatenate to (N, 6)
    return jnp.concatenate([E_total, B_total], axis=-1)

# ==============================================================================
# 4. Main Execution
# ==============================================================================

def main():
    key = jax.random.PRNGKey(42)

    # --- A. Data Setup ---
    omega_val = 2.0 * jnp.pi
    
    # Exact parameters from your PyTorch snippet
    EE0s    = jnp.array([[-2, 0, 1], [1, 1, 0], [1, -1, -1], [3, 2, 1], [-7, 2, 3]], dtype=jnp.float64)
    k0_dirs = jnp.array([[1, 0, 2], [0, 0, 1], [0, -1, 1], [-1, 1, 1], [0, 3, -2]], dtype=jnp.float64)
    
    # Create Spatial Grid (N=1000 pts approx)
    axis = jnp.linspace(-1, 1, 10, dtype=jnp.float64)
    X1, X2, X3 = jnp.meshgrid(axis, axis, axis, indexing='ij')
    X_total = jnp.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=-1) # (1000, 3)
    
    # Generate Ground Truth
    EB_truth_matrix = compute_ground_truth(X_total, EE0s, k0_dirs, omega_val) # (1000, 6)
    
    # Create Train/Test Split
    n_train = 100
    indices = jax.random.permutation(key, X_total.shape[0])[:n_train]
    
    X_train = X_total[indices]
    y_train_matrix = EB_truth_matrix[indices]
    
    # FLATTEN y for the GP: (N, 6) -> (6N, 1)
    y_train_flat = y_train_matrix.reshape(-1, 1)
    
    print(f"Data Generated. X_total: {X_total.shape}, y_train_flat: {y_train_flat.shape}")

    # --- B. Model Initialization ---
    # N_spectral = 12 (Fibonacci sphere)
    kernel = MaxwellKernel(n_spectral=12, omega=omega_val, key=key)
    gp = GaussianProcess(kernel, X_train)
    
    # Log noise parameter (learnable)
    log_noise = jnp.array([-10.0], dtype=jnp.float64)

    # --- C. Training Loop ---
    optimizer = optax.adam(1e-3)
    params = (gp, log_noise)
    opt_state = optimizer.init(eqx.filter(params, eqx.is_inexact_array))

    @eqx.filter_value_and_grad
    def loss_fn(p):
        model, l_noise = p
        return model.nlml(y_train_flat, noise=jnp.exp(l_noise)[0])

    @eqx.filter_jit
    def train_step(p, state):
        loss, grads = loss_fn(p)
        updates, state = optimizer.update(grads, state, p)
        p = eqx.apply_updates(p, updates)
        return loss, p, state

    print("\nStarting Training...")
    for i in range(10000):
        loss_val, params, opt_state = train_step(params, opt_state)
        if i % 100 == 0:
            noise_val = jnp.exp(params[1])[0]
            print(f"Iter {i:03d} | NLML: {loss_val.item():.4e} | Noise: {noise_val:.2e}")
            rmse = jnp.sqrt(loss_val.item() / (n_train * 6))
            print(f"          Approx RMSE: {rmse:.4e}")
    print("Training Complete.")

    # --- D. Evaluation & RMSE ---
    gp_trained, _ = params
    
    # Predict on X_total
    print("\nPredicting on full grid...")
    # This returns (6*N_total, 1)
    mu_flat = gp_trained.posterior_mean(X_total, y_train_flat)
    
    # Reshape back to (N, 6) for comparison
    mu_matrix = mu_flat.reshape(X_total.shape[0], 6)
    
    # Calculate RMSE
    diff = mu_matrix - EB_truth_matrix
    
    # Complex RMSE
    mse_complex = jnp.mean(diff.conj() * diff).real
    rmse_complex = jnp.sqrt(mse_complex)
    
    # Real RMSE
    diff_real = mu_matrix.real - EB_truth_matrix.real
    rmse_real = jnp.sqrt(jnp.mean(diff_real**2))
    
    # Imaginary RMSE
    diff_imag = mu_matrix.imag - EB_truth_matrix.imag
    rmse_imag = jnp.sqrt(jnp.mean(diff_imag**2))

    print("-" * 30)
    print(f"Total RMSE (Complex) : {rmse_complex.item():.4e}")
    print(f"Real RMSE            : {rmse_real.item():.4e}")
    print(f"Imaginary RMSE       : {rmse_imag.item():.4e}")
    print("-" * 30)

if __name__ == "__main__":
    main()