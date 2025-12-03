import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import Array, Float, Complex, Int, Key
from typing import Optional

# Enable x64 (Crucial for GPs)
jax.config.update("jax_enable_x64", True)

# --------------------------------------------------------------------------
# 1. Utilities
# --------------------------------------------------------------------------
def fibonacci_sphere(n: int, dtype=jnp.float64) -> Float[Array, "n 3"]:
    k = jnp.arange(n, dtype=dtype) + 0.5
    phi = jnp.pi * (1.0 + jnp.sqrt(5.0))
    z = 1.0 - 2.0 * k / n
    r = jnp.sqrt(jnp.clip(1.0 - z*z, a_min=0.0))
    theta = phi * k
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    dirs = jnp.stack([x, y, z], axis=-1)
    norm = jnp.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / norm

def normalize(v: Array, axis: int = 1, eps: float = 1e-12) -> Array:
    return v / (jnp.linalg.norm(v, axis=axis, keepdims=True) + eps)

# --------------------------------------------------------------------------
# 2. Feature Map
# --------------------------------------------------------------------------
class PolarLightConeFeatureMap(eqx.Module):
    base_dirs_raw: Float[Array, "n_spectral 3"]
    # Static fields are ignored by JAX grads/transformations
    omega: float = eqx.field(static=True) 
    n_spectral: int = eqx.field(static=True)
    n_pol: int = eqx.field(static=True)

    def __init__(self, n_spectral: int, omega: float, key=None, init_jitter: float=0.0):
        self.n_spectral = int(n_spectral)
        self.n_pol = 2
        self.omega = float(omega)
        
        base = fibonacci_sphere(self.n_spectral)
        if init_jitter > 0.0 and key is not None:
            noise = init_jitter * jax.random.normal(key, base.shape, dtype=jnp.float64)
            base = base + noise
            base = normalize(base)
        self.base_dirs_raw = base

    @property
    def kdirs_unit(self):
        return normalize(self.base_dirs_raw)
    
    @property
    def n_features(self):
        return self.n_spectral * self.n_pol

    def __call__(self, X: Float[Array, "N 3"]) -> Complex[Array, "F 6N"]:
        N = X.shape[0]
        kdirs = self.kdirs_unit 
        w = jnp.array(self.omega, dtype=jnp.float64)

        # 1. Polarizations
        I = jnp.eye(3, dtype=jnp.float64)
        P = I - kdirs[:, :, None] * kdirs[:, None, :] 
        S = jnp.array([[1.,0.],[0.,1.],[0.,0.]], dtype=jnp.float64)
        V = jnp.einsum('rij,jk->rik', P, S)
        
        e1 = normalize(V[:,:,0])
        v2 = V[:,:,1] - e1 * jnp.sum(V[:,:,1]*e1, axis=1, keepdims=True)
        e2 = normalize(v2)
        pols = jnp.stack([e1, e2], axis=1)

        # 2. Coefficients
        k_vec = kdirs * w
        k_exp = k_vec[:, None, :] 
        cross_k_pi = jnp.cross(k_exp, pols, axis=-1)
        E = -w * cross_k_pi
        B = jnp.cross(k_exp, cross_k_pi, axis=-1)
        coeff6 = jnp.concatenate([E, B], axis=-1).astype(jnp.complex128)

        # 3. Phases
        phase = jnp.exp(1j * (X @ k_vec.T))

        # 4. Broadcast and Flatten
        phase_rn = phase.T[:, None, :, None]
        feat = coeff6[:, :, None, :] * phase_rn
        return feat.reshape(self.n_features, -1)

# --------------------------------------------------------------------------
# 3. Kernel & GP
# --------------------------------------------------------------------------
class MaxwellKernel(eqx.Module):
    feature_map: PolarLightConeFeatureMap
    log_w: Float[Array, "F"]

    def __init__(self, n_spectral: int, omega: float, key=None):
        self.feature_map = PolarLightConeFeatureMap(n_spectral, omega, key, init_jitter=0.0)
        self.log_w = jnp.zeros(n_spectral * 2, dtype=jnp.float64)

    def assemble_A(self, Phi: Array, jitter: float = 1e-6) -> Array:
        F = Phi.shape[0]
        W_diag = jnp.exp(self.log_w).astype(jnp.complex128)
        W = jnp.diag(W_diag)
        Phi_outer = Phi @ Phi.conj().T
        A = W + Phi_outer
        if jitter > 0:
            A = A + jitter * jnp.eye(F, dtype=jnp.complex128)
        return A

class GaussianProcess(eqx.Module):
    num_data: Int = eqx.field(static=True)
    kernel: MaxwellKernel
    X: Array

    def __init__(self, kernel: MaxwellKernel, X: Array):
        self.kernel = kernel
        self.X = X
        self.num_data = X.shape[0] * 6

    def nlml(self, y: Array, noise: Float) -> Array:
        y = y.astype(jnp.complex128)
        Phi = self.kernel.feature_map(self.X)
        A = self.kernel.assemble_A(Phi, jitter=1e-6)
        
        L = jax.scipy.linalg.cholesky(A, lower=True)
        
        b = Phi @ y
        z = jax.scipy.linalg.solve_triangular(L, b, lower=True)
        alpha = jax.scipy.linalg.solve_triangular(L.conj().T, z, lower=False)

        y_norm2 = jnp.sum((y.conj() * y).real)
        Fy = Phi.conj().T @ alpha
        quad = jnp.sum((Fy.conj() * Fy).real)
        
        term1 = (0.5 / noise) * (y_norm2 - quad)
        term2 = jnp.sum(jnp.log(jnp.diagonal(L).real))
        term3 = 0.5 * jnp.sum(jnp.exp(self.kernel.log_w))
        
        return term1 + term2 + term3

    def posterior_mean(self, X_query: Array, y_train: Array) -> Array:
        y = y_train.astype(jnp.complex128)
        Phi_x = self.kernel.feature_map(self.X)
        Phi_q = self.kernel.feature_map(X_query)
        
        A = self.kernel.assemble_A(Phi_x, jitter=1e-6)
        L = jax.scipy.linalg.cholesky(A, lower=True)
        
        b = Phi_x @ y
        z = jax.scipy.linalg.solve_triangular(L, b, lower=True)
        alpha = jax.scipy.linalg.solve_triangular(L.conj().T, z, lower=False)
        
        return Phi_q.conj().T @ alpha

# --------------------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------------------
def compute_ground_truth(X, EE0s, k0_dirs, omega):
    k_dirs_norm = normalize(k0_dirs)
    k_vecs = k_dirs_norm * omega 
    phases = jnp.exp(1j * jnp.dot(X, k_vecs.T))
    BB0s = jnp.cross(k_dirs_norm, EE0s)
    E_total = jnp.dot(phases, EE0s.astype(jnp.complex128))
    B_total = jnp.dot(phases, BB0s.astype(jnp.complex128))
    return jnp.concatenate([E_total, B_total], axis=-1)

def main():
    key = jax.random.PRNGKey(42)

    # Setup Data
    omega_val = 2.0 * jnp.pi
    EE0s    = jnp.array([[-2, 0, 1], [1, 1, 0], [1, -1, -1], [3, 2, 1], [-7, 2, 3]], dtype=jnp.float64)
    k0_dirs = jnp.array([[1, 0, 2], [0, 0, 1], [0, -1, 1], [-1, 1, 1], [0, 3, -2]], dtype=jnp.float64)
    
    axis = jnp.linspace(-1, 1, 100, dtype=jnp.float64)
    X1, X2, X3 = jnp.meshgrid(axis, axis, axis, indexing='ij')
    X_total = jnp.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=-1)
    
    EB_truth_matrix = compute_ground_truth(X_total, EE0s, k0_dirs, omega_val)
    
    n_train = 100
    indices = jax.random.permutation(key, X_total.shape[0])[:n_train]
    X_train = X_total[indices]
    y_train_flat = EB_truth_matrix[indices].reshape(-1, 1)

    # Init Model
    kernel = MaxwellKernel(n_spectral=12, omega=float(omega_val), key=key)
    gp = GaussianProcess(kernel, X_train)
    log_eps = jnp.array([-10.0], dtype=jnp.float64) # Matches PyTorch -10.0

    # Optimizer
    optimizer = optax.adam(1e-3)
    params = (gp, log_eps)
    opt_state = optimizer.init(eqx.filter(params, eqx.is_inexact_array))

    @eqx.filter_value_and_grad
    def loss_fn(p):
        model, l_eps = p
        return model.nlml(y_train_flat, noise=jnp.exp(l_eps)[0])

    @eqx.filter_jit
    def step(p, state):
        loss, grads = loss_fn(p)
        updates, state = optimizer.update(grads, state, p)
        p = eqx.apply_updates(p, updates)
        
        # --- CRITICAL FIX: CLAMP log_w to [-20, 10] ---
        # This matches `self.log_w.data.clamp_(-20, 10)` in PyTorch
        new_gp = eqx.tree_at(
            lambda g: g.kernel.log_w, 
            p[0], 
            jnp.clip(p[0].kernel.log_w, -20.0, 10.0)
        )
        p = (new_gp, p[1])
        # ----------------------------------------------
        
        return loss, p, state

    print("Starting Training (Fixed: Clamped log_w + Correct Init)...")
    for i in range(1001):
        loss_val, params, opt_state = step(params, opt_state)
        
        if i % 100 == 0:
            noise_val = jnp.exp(params[1])[0]
            gp_curr = params[0]
            mu_train = gp_curr.posterior_mean(X_train, y_train_flat)
            train_rmse = jnp.sqrt(jnp.mean((mu_train.real - y_train_flat.real)**2))
            print(f"[{i:04d}] NLML: {loss_val.item():.4e} | eps: {noise_val:.2e} | Train RMSE: {train_rmse:.4e}")

    # Eval
    gp_final, _ = params
    mu_flat = gp_final.posterior_mean(X_total, y_train_flat)
    mu_matrix = mu_flat.reshape(X_total.shape[0], 6)
    
    diff = mu_matrix - EB_truth_matrix
    rmse_complex = jnp.sqrt(jnp.mean(diff.conj() * diff).real)
    
    print(f"\nFinal RMSE (Complex): {rmse_complex.item():.4e}")

if __name__ == "__main__":
    main()