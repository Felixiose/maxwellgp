import jax.numpy as jnp 
import jax
from jaxtyping import Array, UInt, Float



def normalize(v: Array, axis: int = 1, eps: float = 1e-12) -> Array:
    """Normalize vector to unit length."""
    norms = jnp.linalg.norm(v, axis=axis, keepdims=True)
    return v / (norms + eps)

def UniformUnitSphereS2(n: int, dtype=jnp.float64) -> Float[Array, "n 3"]:
    """
    Deterministically generates N nearly uniformly distributed points on the unit sphere.
    """
    k = jnp.arange(n, dtype=dtype) + 0.5
    phi = jnp.pi * (1.0 + jnp.sqrt(5.0))  # Golden angle approx
    
    z = 1.0 - 2.0 * k / n
    r = jnp.sqrt(jnp.clip(1.0 - z*z, a_min=0.0))
    theta = phi * k
    
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    
    dirs = jnp.stack([x, y, z], axis=-1)
    # Numerical safety: normalize
    norm = jnp.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / norm




if __name__ == "__main__":
    # Create some tests for the utility functions
    key = jax.random.PRNGKey(0)
    n_points = 1000
    points = UniformUnitSphereS2(n_points)
    print("Generated points shape:", points.shape)
    print("First 5 points:\n", points[:5])
    norms = jnp.linalg.norm(points, axis=-1)
    print("Norms of the points (should be 1):\n", norms[:5])

    # plot the points to verify uniform distribution
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')      
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_title('Uniformly Distributed Points on Unit Sphere S2')
    plt.show()  
               