import torch
import torch.nn as nn
from torchdiffeq import odeint

# --------------------------- General2ndODE ---------------------------
class General2ndODE(nn.Module):
    """
    z = [x, y, vx, vy, theta, omega, s, l, a]

    - (x, y, vx, vy): 二阶线性动力学 + residual 修正
    - (theta, omega): only for 阻尼摆/rotate（线性化）:
        dtheta/dt = omega
        domega/dt = - (g/L) * theta - gamma * omega + residual 修正
    - (s, l, a):  shape related 一阶线性动力学 + residual 修正
    """
    def __init__(self, hidden=32):
        super().__init__()

        # Linear coeffs for (x,y, vx, vy)
        self.a_x = nn.Parameter(torch.tensor(0.1))
        self.b_x = nn.Parameter(torch.tensor(0.1))
        self.c_x = nn.Parameter(torch.tensor(0.1))
        self.a_y = nn.Parameter(torch.tensor(0.1))
        self.b_y = nn.Parameter(torch.tensor(0.1))
        self.c_y = nn.Parameter(torch.tensor(-8.0))

        # Pendulum params (linearized) for (theta omega)
        self.g_over_L = nn.Parameter(torch.tensor(9.8/5))
        self.gamma = nn.Parameter(torch.tensor(0.1))

        # Linear coeffs for (s, l, a)
        self.alpha_s = nn.Parameter(torch.tensor(0.0))
        self.beta_s  = nn.Parameter(torch.tensor(0.0))
        self.alpha_l = nn.Parameter(torch.tensor(0.0))
        self.beta_l  = nn.Parameter(torch.tensor(0.0))
        self.alpha_a = nn.Parameter(torch.tensor(0.0))
        self.beta_a  = nn.Parameter(torch.tensor(0.0))

        # residual scale
        self.res_scale = nn.Parameter(torch.tensor(0.01), requires_grad=True)

        # Residual MLP for non-linear corrections
        self.residual = nn.Sequential(
            nn.Linear(9, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 6)  # [ax_res, ay_res, domega_res, ds_res, dl_res, da_res]
        )
        # 初始化 residual 为 0
        for m in self.residual:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t, z):
        x, y, vx, vy, theta, omega, s, l, a = torch.split(z, 1, dim=1)

        # Linear 2nd-order for (x,y)
        ax_lin = self.a_x * x + self.b_x * vx + self.c_x
        ay_lin = self.a_y * y + self.b_y * vy + self.c_y

        # Linearized pendulum (theta, omega)
        dtheta = omega
        domega_lin = -self.g_over_L * theta - self.gamma * omega

        # Linear 1st-order for (s, l, a)
        ds_lin = self.alpha_s * s + self.beta_s
        dl_lin = self.alpha_l * l + self.beta_l
        da_lin = self.alpha_a * a + self.beta_a

        # Residual
        res = torch.tanh(self.residual(torch.cat([x, y, vx, vy, theta, omega, s, l, a], dim=1)))
        ax = ax_lin + self.res_scale * res[:, 0:1]
        ay = ay_lin + self.res_scale * res[:, 1:2]
        domega = domega_lin + self.res_scale * res[:, 2:3]
        ds = ds_lin + self.res_scale * res[:, 3:4]
        dl = dl_lin + self.res_scale * res[:, 4:5]
        da = da_lin + self.res_scale * res[:, 5:6]

        dzdt = torch.cat([vx, vy, ax, ay, dtheta, domega, ds, dl, da], dim=1)
        return dzdt


class NewtonODELatent(nn.Module):
    """
    Latent dynamics with General2ndODE
    """
    def __init__(self):
        super().__init__()
        self.func = General2ndODE(hidden=32)

    def forward(self, z0: torch.Tensor, ts: torch.Tensor):
        """
        Args:
            z0: (B,9) -> [x, y, vx, vy, theta, omega, s, l, a]
            ts: (T,) time stamps
        Returns:
            dynamics: (B,T,9)
        """
        dynamics = odeint(self.func, z0, ts, rtol=1e-5, atol=1e-6)
        dynamics = dynamics.permute(1, 0, 2)  # (B,T,9)
        return dynamics

