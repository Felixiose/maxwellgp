import torch
import torch.nn as nn
import torch.optim as optim
import math

verbose = True
# ----------------------------
# Global defaults 
# ----------------------------
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CDTYPE  = torch.complex128
RDTYPE  = torch.float64


# ----------------------------
# Utilities
# ----------------------------
def fibonacci_sphere(n: int,
                     device: torch.device = DEVICE,
                     dtype: torch.dtype = RDTYPE) -> torch.Tensor:
    """
    Deterministic, nearly-uniform points on S^2 (unit sphere).
    Returns: (n, 3) real tensor of unit vectors.
    """
    # Fibonacci lattice on sphere
    k = torch.arange(n, dtype=dtype, device=device) + 0.5 # [0.5, 1.5, 2.5, ...]
    phi = 2.0 * math.pi * (1.0 / ((1 + math.sqrt(5)) / 2.0))  # 2π/φ
    z = 1.0 - 2.0 * k / n
    r = torch.sqrt(torch.clamp(1.0 - z*z, min=0.0))
    theta = phi *  k # torch.ones_like(k)*
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    dirs = torch.stack([x, y, z], dim=-1)  # (n,3)
    # numerical safety: normalize to unit
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    return dirs

def normalize(v: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.norm(dim=dim, keepdim=True) + eps)

# ----------------------------
# feature map: light-cone, two polarizations per k
# ----------------------------
class PolarLightConeFeatureMap(nn.Module):

    def __init__(self,
                 n_spectral: int,
                 omega: torch.Tensor,
                 device: torch.device = DEVICE,
                 init_jitter: float = 0.0):      # small noise to help escape symmetry
        super().__init__()
        
        self.n_spectral = int(n_spectral)
        self.n_pol      = 2
        self.omega      = torch.as_tensor(omega, dtype=RDTYPE, device=device)

        base = fibonacci_sphere(self.n_spectral, device=device, dtype=RDTYPE) 
        if init_jitter > 0.0:
            # optional randomness (set 0.0 for fully deterministic)
            noise = init_jitter * torch.randn_like(base)
            base = base + noise
            base = base / base.norm(dim=1, keepdim=True)
        self.base_dirs_raw = nn.Parameter(base)

    @property
    def kdirs_unit(self) -> torch.Tensor:
        # re-normalize during training for safety
        v = self.base_dirs_raw
        return v / (v.norm(dim=1, keepdim=True) + 1e-12)

    @property
    def n_features(self) -> int:
        return self.n_spectral * self.n_pol

    def two_polarization_eq(self, kdirs_unit: torch.Tensor) -> torch.Tensor:
        """
        SO(3)-friendly polarizations via projection + GS.
        kdirs_unit: (r,3)  ->  pols: (r,2,3), real, each ⟂ k, orthonormal
        """
        r = kdirs_unit.shape[0]
        I = torch.eye(3, dtype=RDTYPE, device=kdirs_unit.device).expand(r,3,3)
        # Project two fixed seeds onto the tangent plane
        P = I - kdirs_unit[:, :, None] * kdirs_unit[:, None, :]           # (r,3,3)
        S = torch.tensor([[1.,0.],[0.,1.],[0.,0.]], dtype=RDTYPE, device=kdirs_unit.device)  # (3,2)
        V = torch.einsum('rij,jk->rik', P, S)                              # (r,3,2)

        # Gram–Schmidt
        e1 = normalize(V[:,:,0], dim=1)
        v2 = V[:,:,1] - (e1 * (V[:,:,1]*e1).sum(dim=1, keepdim=True))
        e2 = normalize(v2, dim=1)
        pols = torch.stack([e1, e2], dim=1)                                # (r,2,3)
        return pols

    def _coefficients_EB(self, kdirs_unit: torch.Tensor) -> torch.Tensor:
        """
        Build per-(k,pol) coefficient vector for (E,B) given π (polarization basis).
        Returns: coeff6 (r, 2, 6) complex
        """
        w = self.omega
        k = kdirs_unit * w                               # (r,3), real
        pols = self.two_polarization_eq(kdirs_unit)              # (r,2,3)

        k_exp = k[:, None, :]                            # (r,1,3)
        cross_k_pi = torch.cross(k_exp, pols, dim=-1)    # (r,2,3)

        E =  - w * cross_k_pi                               # (r,2,3) real
        B =  torch.cross(k_exp, cross_k_pi, dim=-1)      # (r,2,3) real

        coeff6 = torch.cat([E, B], dim=-1)                # (r,2,6)
        
        return coeff6.to(CDTYPE)



    def _phases(self, X: torch.Tensor, kdirs_unit: torch.Tensor) -> torch.Tensor:
        """
        X: (N,3) real
        returns phase: (N, r) complex with e^{i k·x}
        """
        Xr = X.to(RDTYPE)
        w = self.omega
        k = kdirs_unit * w                                # (r,3)
        phase = torch.exp(1j * (Xr @ k.T))                # (N, r)
        return phase


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns Φ of shape (F, 6N) where F = 2 * n_spectral.
        Feature order: (for each k): pol=0, pol=1
        """
        N = X.shape[0]
        kdirs = self.kdirs_unit                           # (r,3)
        coeff6 = self._coefficients_EB(kdirs)             # (r,2,6) complex
        phase  = self._phases(X, kdirs)                   # (N,r) complex

        # Broadcast to (r,2,N,6) then flatten to (2r, 6N)
        phase_rn = phase.T[:, None, :, None]              # (r,1,N,1)
        feat_r2n6 = coeff6[:, :, None, :] * phase_rn      # (r,2,N,6)
        Phi = feat_r2n6.reshape(self.n_spectral*self.n_pol, N, 6).reshape(-1, 6*N)
        return Phi


# ----------------------------
# Weight-space GP (feature GP)
# ----------------------------
class FeatureSpaceGP(nn.Module):
    """
    Gaussian process in feature space:
      y = Φ^H w + ε,  w ~ N(0, diag(exp(log_w)))
    Works with any FeatureMap producing Φ of shape (F, D_out).
    """
    def __init__(self,
                 feature_map,
                 init_log_eps: float = -4.6,   # ~0.01
                 device: torch.device = DEVICE):
        super().__init__()
        
        self.fm = feature_map.to(device)
        F = self.fm.n_features # for spectral 2*

        # diagonal spectral prior per feature
        self.log_w   = nn.Parameter(torch.zeros((F,), dtype=RDTYPE, device=device))
        self.log_eps = nn.Parameter(torch.tensor(init_log_eps, dtype=RDTYPE, device=device))

    @property
    def noise(self) -> torch.Tensor:
        return torch.exp(self.log_eps).to(RDTYPE)

    def _assemble_A(self, Phi: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
        """
        A = diag(exp(log_w)) + Φ Φ^H  (+ jitter I)
        Returns: (F, F) complex Hermitian PD
        """
        F = Phi.shape[0]
        W = torch.diag_embed(torch.exp(self.log_w).to(RDTYPE)).to(CDTYPE)  # (F,F)
        A = W + Phi @ Phi.conj().transpose(-1, -2)  
        if jitter is not None and jitter > 0:
            A = A + torch.diag_embed(torch.full((F,), jitter, dtype=CDTYPE, device=Phi.device))
        return A
    

    def mll(self, Phi: torch.Tensor, y: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
        """
        Negative log marginal likelihood (up to constants).
        """
        with torch.no_grad():
            self.log_w.data.clamp_(-20.0, 10.0)
        y = y.to(CDTYPE)
        A = self._assemble_A(Phi, jitter=jitter)
        # print condition number of A
        L = torch.linalg.cholesky(A)
        # α = A^{-1} Φ y  via two solves
        alpha = torch.cholesky_solve(Phi @ y, L)  # (F,1)
 
        # 0.5/ε*(||y||^2 - ||Φ^H α||^2) + log|L| + 0.5*tr(W)
        eps = self.noise
        y_norm2 = (y.conj().T @ y).real.squeeze()
        Fy = Phi.conj().transpose(-1, -2) @ alpha
        quad = (Fy.conj().T @ Fy).real.squeeze()
        nlml = 0.5/eps * (y_norm2 - quad)
        nlml = nlml + torch.log(torch.diagonal(L).real).sum()
        nlml = nlml + 0.5 * torch.exp(self.log_w).sum()

        return nlml



    def fit(self,
            X: torch.Tensor,
            y: torch.Tensor,
            epochs: int = 500,
            lr_map: float = 1e-2,
            lr_gp:  float = 1e-2,
            jitter: float = 1e-6,
            log_every: int = 100) -> dict:
        """
        Jointly optimizes feature map parameters (k-directions) and GP hyperparams.
        """
        # Two parameter groups: feature map params vs GP params
        params_map = [p for n,p in self.fm.named_parameters() if p.requires_grad]
        params_gp  = [self.log_w, self.log_eps]
        opt = optim.Adam([
            {'params': params_map, 'lr': lr_map},
            {'params': params_gp,  'lr': lr_gp},
        ])

        y = y.to(CDTYPE)
        lml_hist, rmse_hist = [], []

        for epoch in range(epochs):
            Phi  = self.fm(X)                      # (F,6N)
            nlml = self.mll(Phi, y, jitter=jitter)
            
            opt.zero_grad()
            nlml.backward()
            opt.step()

            if ((epoch % log_every) == 0 or epoch == epochs-1) and verbose:
                with torch.no_grad():
                    # posterior mean = Φ^H A^{-1} Φ y
                    A = self._assemble_A(Phi, jitter=jitter)
                    L = torch.linalg.cholesky(A)
                    alpha = torch.cholesky_solve(Phi @ y, L)
                    y_pred = (Phi.conj().transpose(-1, -2) @ alpha)  # (6N,1)
                    rmse = (y_pred.real - y.real).pow(2).mean().sqrt()
                    lml_hist.append(-nlml.item())
                    rmse_hist.append(rmse.item())
                    print(f"[{epoch:04d}] NLML: {nlml.item():.4e} | RMSE: {rmse.item():.4e} | eps: {self.noise.item():.2e}")

        return {'lml': lml_hist, 'rmse': rmse_hist}

    def posterior_mean(self,
                       X_query: torch.Tensor,
                       X_train: torch.Tensor,
                       y_train: torch.Tensor,
                       jitter: float = 1e-6) -> torch.Tensor:
        """
        Compute μ_*(X_query) = Φ_q^H A^{-1} Φ_x y  with current learned params.
        Returns shape (6 N_q, 1) complex.
        """
        y = y_train.to(CDTYPE)
        Phi_x = self.fm(X_train)                        # (F, 6N_x)
        Phi_q = self.fm(X_query)                        # (F, 6N_q)
        A = self._assemble_A(Phi_x, jitter=jitter)
        L = torch.linalg.cholesky(A)
        alpha = torch.cholesky_solve(Phi_x @ y, L)      # (F,1)
        pred  = Phi_q.conj().transpose(-1, -2) @ alpha  # (6N_q,1)
        return pred



if __name__ == "__main__":
    # Your helpers
    from utils import compute_phase_fields, make_initial_data, GaussianPacketDataGenerator

    omega   = torch.tensor(2*torch.pi, dtype=RDTYPE, device=DEVICE)
    EE0s    = torch.tensor([[-2,  0,  1],
                            [ 1,  1,  0],
                            [ 1, -1, -1],
                            [ 3,  2,  1],
                            [-7,  2,  3]], 
                            dtype=RDTYPE, device=DEVICE)
    k0_dirs = torch.tensor([[ 1,  0,  2],
                            [ 0,  0,  1],
                            [ 0, -1,  1],
                            [-1,  1,  1],
                            [ 0,  3, -2],
                            ], dtype=CDTYPE, device=DEVICE)
    
    
    axis    = torch.linspace(-1, 1, 100, dtype=RDTYPE, device=DEVICE)

    X_total = torch.cartesian_prod(axis, axis, axis)               # (N,3)
    EBdata  = compute_phase_fields(axis, EE0s, k0_dirs, omega)     # (N,6,1) 

    n_train_pts = 100
    X_train, y_train = make_initial_data((X_total, EBdata), n_train_pts)  # y: (6N_train,1)
    
    # Build feature map + GP
    n_spectral_points = 24
    fm = PolarLightConeFeatureMap(n_spectral_points, omega, device=DEVICE)
 
    gp = FeatureSpaceGP(fm,init_log_eps=-10, device=DEVICE)
    history = gp.fit(X_train, y_train, epochs=1000, lr_map=1e-3, lr_gp=1e-3, jitter=1e-6, log_every=100)
    prediction = gp.posterior_mean(X_total, X_train, y_train)

    diff = prediction.view_as(EBdata) - EBdata
    rmse = torch.sqrt((diff.conj() * diff).real.mean())
    print(f"RMSE: {rmse.item():.4e}")

    #real rmse
    diff_real = prediction.view_as(EBdata).real - EBdata.real
    rmse_real = torch.sqrt((diff_real.conj() * diff_real).mean())
    print(f"Real RMSE: {rmse_real.item():.4e}")

    #imaginary rmse
    diff_img = prediction.view_as(EBdata).imag - EBdata.imag
    rmse_img = torch.sqrt((diff_img.conj() * diff_img).mean())
    print(f"Imaginary RMSE: {rmse_img.item():.4e}")

    # import plotting
    # plotting.plot_E_magnitude(prediction.detach().view_as(EBdata), 5)
    # plotting.plot_E_magnitude(EBdata, 5)
