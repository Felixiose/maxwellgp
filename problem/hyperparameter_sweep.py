import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import Array, Float, Complex
import itertools
from GP import (
    MaxwellKernel, GaussianProcess, compute_ground_truth,
    fibonacci_sphere, normalize
)

# Enable x64
jax.config.update("jax_enable_x64", True)

def train_model(n_spectral, lr_map, lr_gp, init_log_eps, epochs, key):
    """Train a single model with given hyperparameters."""
    key_data, key_model = jax.random.split(key)
    
    # Setup Data
    omega_val = 2.0 * jnp.pi
    EE0s = jnp.array([[-2, 0, 1], [1, 1, 0], [1, -1, -1], [3, 2, 1], [-7, 2, 3]], dtype=jnp.float64)
    k0_dirs = jnp.array([[1, 0, 2], [0, 0, 1], [0, -1, 1], [-1, 1, 1], [0, 3, -2]], dtype=jnp.float64)
    
    axis = jnp.linspace(-1, 1, 100, dtype=jnp.float64)
    X1, X2, X3 = jnp.meshgrid(axis, axis, axis, indexing='ij')
    X_total = jnp.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=-1)
    
    EB_truth_matrix = compute_ground_truth(X_total, EE0s, k0_dirs, omega_val)
    
    n_train = 100
    indices = jax.random.permutation(key_data, X_total.shape[0])[:n_train]
    X_train = X_total[indices]
    y_train_flat = EB_truth_matrix[indices].reshape(-1, 1)
    
    # Init Model
    kernel = MaxwellKernel(n_spectral=n_spectral, omega=float(omega_val), key=key_model)
    gp = GaussianProcess(kernel, X_train)
    log_eps = jnp.array([init_log_eps], dtype=jnp.float64)
    
    # Separate optimizers
    opt_map = optax.adam(lr_map)
    opt_gp = optax.adam(lr_gp)
    
    fm_params = eqx.filter(gp.kernel.feature_map, eqx.is_inexact_array)
    opt_state_map = opt_map.init(fm_params)
    
    gp_params = (gp.kernel.log_w, log_eps)
    opt_state_gp = opt_gp.init(gp_params)
    
    params = (gp, log_eps)
    
    @eqx.filter_value_and_grad
    def loss_fn(p):
        model, l_eps = p
        return model.nlml(y_train_flat, noise=jnp.exp(l_eps)[0])
    
    @eqx.filter_jit
    def step(p, state_map, state_gp):
        loss, grads = loss_fn(p)
        
        grad_fm = eqx.filter(grads[0].kernel.feature_map, eqx.is_inexact_array)
        updates_map, state_map = opt_map.update(grad_fm, state_map, p[0].kernel.feature_map)
        new_fm = eqx.apply_updates(p[0].kernel.feature_map, updates_map)
        
        grad_log_w = grads[0].kernel.log_w
        grad_log_eps = grads[1]
        updates_gp, state_gp = opt_gp.update(
            (grad_log_w, grad_log_eps), 
            state_gp, 
            (p[0].kernel.log_w, p[1])
        )
        new_log_w = eqx.apply_updates(p[0].kernel.log_w, updates_gp[0])
        new_log_eps = eqx.apply_updates(p[1], updates_gp[1])
        
        new_kernel = eqx.tree_at(lambda k: k.feature_map, p[0].kernel, new_fm)
        new_kernel = eqx.tree_at(lambda k: k.log_w, new_kernel, new_log_w)
        new_gp = eqx.tree_at(lambda g: g.kernel, p[0], new_kernel)
        
        new_gp = eqx.tree_at(
            lambda g: g.kernel.log_w,
            new_gp,
            jnp.clip(new_gp.kernel.log_w, -20.0, 10.0)
        )
        p = (new_gp, new_log_eps)
        
        return loss, p, state_map, state_gp
    
    # Training loop
    best_loss = float('inf')
    best_rmse = float('inf')
    
    for i in range(epochs):
        loss_val, params, opt_state_map, opt_state_gp = step(params, opt_state_map, opt_state_gp)
        
        if i % 100 == 0 or i == epochs - 1:
            gp_curr = params[0]
            mu_train = gp_curr.posterior_mean(X_train, y_train_flat)
            train_rmse = jnp.sqrt(jnp.mean((mu_train.real - y_train_flat.real)**2))
            
            if loss_val < best_loss:
                best_loss = loss_val.item()
            if train_rmse < best_rmse:
                best_rmse = train_rmse.item()
    
    # Final evaluation
    gp_final, _ = params
    mu_flat = gp_final.posterior_mean(X_total, y_train_flat)
    mu_matrix = mu_flat.reshape(X_total.shape[0], 6)
    
    diff = mu_matrix - EB_truth_matrix
    final_rmse = jnp.sqrt(jnp.mean(diff.conj() * diff).real).item()
    
    final_loss = loss_val.item()
    final_train_rmse = jnp.sqrt(jnp.mean((mu_train.real - y_train_flat.real)**2)).item()
    
    return {
        'final_loss': final_loss,
        'best_loss': best_loss,
        'final_train_rmse': final_train_rmse,
        'best_train_rmse': best_rmse,
        'final_rmse': final_rmse,
    }

def main():
    # Hyperparameter grid
    n_spectral_options = [12, 16, 20, 24, 28, 32]
    lr_options = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    init_log_eps_options = [-12.0, -10.0, -8.0, -6.0]
    epochs = 1000
    
    # Use same key for all runs for reproducibility
    base_key = jax.random.PRNGKey(42)
    
    results = []
    total_combinations = len(n_spectral_options) * len(lr_options) * len(lr_options) * len(init_log_eps_options)
    current = 0
    
    print(f"Starting hyperparameter sweep: {total_combinations} combinations")
    print("=" * 80)
    
    for n_spectral, lr_map, lr_gp, init_log_eps in itertools.product(
        n_spectral_options, lr_options, lr_options, init_log_eps_options
    ):
        current += 1
        key = jax.random.fold_in(base_key, current)
        
        try:
            metrics = train_model(n_spectral, lr_map, lr_gp, init_log_eps, epochs, key)
            
            result = {
                'n_spectral': n_spectral,
                'lr_map': lr_map,
                'lr_gp': lr_gp,
                'init_log_eps': init_log_eps,
                **metrics
            }
            results.append(result)
            
            print(f"[{current}/{total_combinations}] "
                  f"n={n_spectral:2d}, lr_map={lr_map:.4f}, lr_gp={lr_gp:.4f}, "
                  f"log_eps={init_log_eps:5.1f} | "
                  f"Loss: {metrics['final_loss']:.2e}, "
                  f"Train RMSE: {metrics['final_train_rmse']:.4f}, "
                  f"Final RMSE: {metrics['final_rmse']:.4f}")
        except Exception as e:
            print(f"[{current}/{total_combinations}] ERROR: {e}")
            continue
    
    # Sort by final RMSE (best first)
    results.sort(key=lambda x: x['final_rmse'])
    
    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS (by Final RMSE):")
    print("=" * 80)
    print(f"{'Rank':<6} {'n_spec':<8} {'lr_map':<10} {'lr_gp':<10} {'log_eps':<10} "
          f"{'Final Loss':<12} {'Train RMSE':<12} {'Final RMSE':<12}")
    print("-" * 80)
    
    for i, r in enumerate(results[:10], 1):
        print(f"{i:<6} {r['n_spectral']:<8} {r['lr_map']:<10.4f} {r['lr_gp']:<10.4f} "
              f"{r['init_log_eps']:<10.1f} {r['final_loss']:<12.2e} "
              f"{r['final_train_rmse']:<12.4f} {r['final_rmse']:<12.4f}")
    
    # Also show best by train RMSE
    results_by_train = sorted(results, key=lambda x: x['final_train_rmse'])
    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS (by Train RMSE):")
    print("=" * 80)
    print(f"{'Rank':<6} {'n_spec':<8} {'lr_map':<10} {'lr_gp':<10} {'log_eps':<10} "
          f"{'Final Loss':<12} {'Train RMSE':<12} {'Final RMSE':<12}")
    print("-" * 80)
    
    for i, r in enumerate(results_by_train[:10], 1):
        print(f"{i:<6} {r['n_spectral']:<8} {r['lr_map']:<10.4f} {r['lr_gp']:<10.4f} "
              f"{r['init_log_eps']:<10.1f} {r['final_loss']:<12.2e} "
              f"{r['final_train_rmse']:<12.4f} {r['final_rmse']:<12.4f}")
    
    # Best overall
    best = results[0]
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION:")
    print("=" * 80)
    print(f"n_spectral: {best['n_spectral']}")
    print(f"lr_map: {best['lr_map']}")
    print(f"lr_gp: {best['lr_gp']}")
    print(f"init_log_eps: {best['init_log_eps']}")
    print(f"Final Loss: {best['final_loss']:.2e}")
    print(f"Train RMSE: {best['final_train_rmse']:.4f}")
    print(f"Final RMSE: {best['final_rmse']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()

