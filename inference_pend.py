#!/usr/bin/env python
"""
Inference for NewtonODELatent with pendulum pivot initialization
Given pivot, l, theta, omega -> construct z0, then predict trajectory and optical flow.
"""

import torch
import numpy as np
from pathlib import Path
from models.nnd_motion_pend import NewtonODELatent  # 你训练时用的模型文件

# ---------------- Config ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("/home/yuan418/data/project/Newtongen_ICLR/runs_nnd_pend/newton_ode_pend_final.pth")  # 保存的模型
DT = 0.02
T_pred = 48  # 未来48帧
H, W = 240, 360
radius_px = 24
METER_PER_PX = 0.05  # 每像素对应的米

# ---------------- Helper: construct z0 from pivot ----------------
def make_z0(pivot, pend_l, theta, omega, s=0.5, l=0.8, d=0.0):
    """
    Construct initial z0 from pendulum geometry.
    Args:
        pivot: (px, py), 支点世界坐标 (m)
        pend_l: 摆长 (m)
        theta: 初始角度 (rad), 相对竖直方向, 正方向为逆时针
        omega: 初始角速度 (rad/s)
        s, d: 其他一阶动力学变量
    Returns:
        z0: (1, 9) torch tensor [x, y, vx, vy, theta, omega, s, l, d]
    """
    # 球的位置
    x = pivot[0] + pend_l * np.sin(theta)
    y = pivot[1] - pend_l * np.cos(theta)

    # 球的速度
    vx = pend_l * omega * np.cos(theta)
    vy = pend_l * omega * np.sin(theta)

    z0 = torch.tensor([[x, y, vx, vy, theta, omega, s, l, d]],
                      dtype=torch.float32, device=DEVICE)
    return z0

# ---------------- Load model ----------------
model = NewtonODELatent().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded trained model from {MODEL_PATH}")

# ---------------- Example initial condition ----------------
pivot = np.array([9.0, 12.0])   # 支点位置 (m)
pend_l = 8.0                         # 摆长 (m)
theta = -0.2                     # 初始角度 (rad)
omega = 0.4                     # 初始角速度 (rad/s)

z0 = make_z0(pivot, pend_l, theta, omega)
print("Initialized z0:", z0)

# ---------------- Time stamps ----------------
ts = torch.arange(T_pred, dtype=torch.float32, device=DEVICE) * DT

# ---------------- Inference ----------------
with torch.no_grad():
    motion = model(z0, ts)  # (1, T_pred, 9)

motion = motion.squeeze(0).cpu().numpy()  # (T_pred, 9)

# ---------------- Save trajectory ----------------
out_dir = Path("runs_nnd_pend")
out_dir.mkdir(parents=True, exist_ok=True)
torch.save(torch.from_numpy(motion), out_dir / "inference_motion_pend.pt")
np.save(out_dir / "motion_pend_world.npy", motion[:, :4])  # 世界坐标 (x,y,vx,vy)

print(f"Full trajectory saved to {out_dir/'inference_motion.pt'}")

# ================================================================
# -------- Convert to pixel coordinates & optical flow -----------
# ================================================================
traj_world = motion[:, :4]  # (T, 4): [x,y,vx,vy] in meters
traj_px = np.zeros_like(traj_world)

# 世界(m) -> 像素(px)
traj_px[:, 0] = traj_world[:, 0] / METER_PER_PX       # x
traj_px[:, 1] = H - (traj_world[:, 1] / METER_PER_PX) # y (注意翻转)
traj_px[:, 2] = traj_world[:, 2] / METER_PER_PX       # vx
traj_px[:, 3] = -traj_world[:, 3] / METER_PER_PX      # vy

np.save(out_dir / "traj_pixel_pend.npy", traj_px)
print(f"Saved traj_world.npy (meters) and traj_pixel.npy (pixels) to {out_dir}")

# ---------------- Build optical flow ----------------
flows = np.zeros((T_pred, 2, H, W), dtype=np.float32)
Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

for t in range(T_pred - 1):
    cx, cy = traj_px[t, 0], traj_px[t, 1]
    nx, ny = traj_px[t + 1, 0], traj_px[t + 1, 1]
    dx, dy = nx - cx, ny - cy

    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius_px ** 2
    flows[t, 0, mask] = dx
    flows[t, 1, mask] = dy

np.save(out_dir / "flows_dxdy_pend.npy", flows)
print(f"Optical flow saved to {out_dir/'flows_dxdy_pend.npy'}, shape={flows.shape}")

# ---------------- Debug print ----------------
print("\nFirst 10 frames (world, meters):")
for t in range(min(10, T_pred)):
    x, y, vx, vy = traj_world[t]s
    print(f"t={t:02d}: x={x:.3f}m, y={y:.3f}m, vx={vx:.3f}m/f, vy={vy:.3f}m/f")

print("\nFirst 10 frames (pixel, px):")
for t in range(min(10, T_pred)):
    x, y, vx, vy = traj_px[t]
    print(f"t={t:02d}: x={x:.1f}px, y={y:.1f}px, vx={vx:.2f}px/f, vy={vy:.2f}px/f")







