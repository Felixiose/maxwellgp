import torch
import torch.nn as nn

def normalize(v: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    """Normalize vector to unit length."""
    return v / (v.norm(dim=dim, keepdim=True) + eps)

def compute_phase_fields(axis: torch.Tensor, EE0s: torch.Tensor, k0_dirs: torch.Tensor, omega: torch.Tensor):
    """
    Compute phase fields for Maxwell equations.
    Similar to compute_ground_truth in JAX version.
    
    Args:
        axis: 1D tensor of axis values
        EE0s: (n_sources, 3) Electric field amplitudes
        k0_dirs: (n_sources, 3) Propagation directions
        omega: Scalar frequency
        
    Returns:
        EBdata: (N, 6, 1) complex tensor with [E, B] fields
    """
    device = axis.device
    dtype = axis.dtype
    
    # Create grid
    X_total = torch.cartesian_prod(axis, axis, axis)  # (N, 3)
    
    # Normalize k directions
    k_dirs_norm = normalize(k0_dirs.real if k0_dirs.is_complex() else k0_dirs)
    
    # Compute k vectors
    k_vecs = k_dirs_norm * omega  # (n_sources, 3)
    
    # Calculate phases: exp(i * k . x)
    phases = torch.exp(1j * (X_total @ k_vecs.T))  # (N, n_sources)
    
    # Calculate B fields: B = khat x E
    BB0s = torch.cross(k_dirs_norm, EE0s, dim=-1)  # (n_sources, 3)
    
    # Sum over sources
    E_total = phases @ EE0s.to(torch.complex128)  # (N, 3)
    B_total = phases @ BB0s.to(torch.complex128)  # (N, 3)
    
    # Concatenate and reshape to (N, 6, 1)
    EB_combined = torch.cat([E_total, B_total], dim=-1)  # (N, 6)
    EBdata = EB_combined.unsqueeze(-1)  # (N, 6, 1)
    
    return EBdata

def make_initial_data(data_tuple, n_train_pts: int):
    """
    Create training data by randomly sampling points.
    
    Args:
        data_tuple: (X_total, EBdata) tuple
        n_train_pts: Number of training points
        
    Returns:
        X_train: (n_train_pts, 3)
        y_train: (6 * n_train_pts, 1) flattened
    """
    X_total, EBdata = data_tuple
    N = X_total.shape[0]
    
    # Randomly sample indices (using seed 42 for reproducibility)
    torch.manual_seed(42)
    indices = torch.randperm(N, device=X_total.device)[:n_train_pts]
    
    X_train = X_total[indices]  # (n_train_pts, 3)
    EB_train = EBdata[indices]  # (n_train_pts, 6, 1)
    
    # Flatten to (6 * n_train_pts, 1)
    y_train = EB_train.reshape(-1, 1)  # (6 * n_train_pts, 1)
    
    return X_train, y_train

# Placeholder for GaussianPacketDataGenerator if needed
class GaussianPacketDataGenerator:
    """Placeholder class if needed elsewhere."""
    pass

