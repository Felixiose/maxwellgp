from jaxtyping import Array, Float
import equinox as eqx   
import jax.numpy as jnp 
import jax

from kernel import MaxwellKernel


class GaussianProcess(eqx.Module):
    kernel: MaxwellKernel
    X: Float[Array, "N 3"] = eqx.field(static=True) # Data is static relative to gradients
    num_data: int = eqx.field(static=True)
    log_eps: Float[Array, "1"] # Learned noise parameter

    def __init__(self, kernel: MaxwellKernel, X: Array, log_eps_init: float = -12.0):
        self.kernel = kernel
        self.X = X
        self.num_data = X.shape[0] * 6
        self.log_eps = jnp.array([log_eps_init], dtype=jnp.float64)

    def compute_A_and_Phi(self, jitter=1e-6):
        Phi = self.kernel.feature_map(self.X)
        W_diag = jnp.exp(self.kernel.log_w).astype(jnp.complex128)
        # Low-rank update structure usually safer with jitter on diagonal
        A = jnp.diag(W_diag) + Phi @ Phi.conj().T + jitter * jnp.eye(Phi.shape[0])
        return A, Phi

    def nlml(self, y: Array) -> Array:
        y = y.astype(jnp.complex128)
        A, Phi = self.compute_A_and_Phi()
        L = jax.scipy.linalg.cholesky(A, lower=True)
        
        # alpha = A^{-1} Phi y
        alpha = jax.scipy.linalg.cho_solve((L, True), Phi @ y)

        noise_std = jnp.exp(self.log_eps)[0]
        
        # Data fit term (Negative Log Likelihood part)
        # Note: 1/noise * (y - Phi.T alpha)^2 is roughly the logic, 
        # but using the Matrix Inversion Lemma form from your original code:
        y_norm2 = (y.conj().T @ y).real.squeeze()
        Fy = Phi.conj().T @ alpha
        quad = (Fy.conj().T @ Fy).real.squeeze()
        
        term1 = (0.5 / noise_std) * (y_norm2 - quad)
        term2 = jnp.sum(jnp.log(jnp.diagonal(L).real))
        term3 = 0.5 * jnp.sum(jnp.exp(self.kernel.log_w))
        
        return term1 + term2 + term3

    def posterior_mean(self, X_query: Array, y_train: Array) -> Array:
        y = y_train.astype(jnp.complex128)
        A, Phi_x = self.compute_A_and_Phi()
        Phi_q = self.kernel.feature_map(X_query)
        
        L = jax.scipy.linalg.cholesky(A, lower=True)
        alpha = jax.scipy.linalg.cho_solve((L, True), Phi_x @ y)
        return Phi_q.conj().T @ alpha
