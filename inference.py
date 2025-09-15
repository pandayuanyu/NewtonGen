#!/usr/bin/env python
#!/usr/bin/env python
"""
Inference for NewtonODELatent
Given z0 and time stamps, outputs predicted motion and optical flow.
"""

import torch
import numpy as np
from pathlib import Path
from models.nnd_motion import NewtonODELatent  # 你训练时用的模型文件

# ---------------- Config ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("/home/yuan418/data/project/Newtongen_ICLR/runs_nnd_fall/newton_ode_fall_epoch05000.pth")  # 保存的模型
DT = 0.02
T_pred = 48  # 未来48帧
H, W = 240, 360
radius_px = 24
METER_PER_PX = 0.1  # 每像素对应的米

# ---------------- Load model ----------------
model = NewtonODELatent().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded trained model from {MODEL_PATH}")

# ---------------- Example initial condition z0 ----------------
# 格式 [x, y, vx, vy, s, θ, l]
z0 = torch.tensor([[10.0, 10.0, 1.0, 10.0, 1000.0, 100.0, 100.0]], dtype=torch.float32, device=DEVICE)

# ---------------- Time stamps ----------------
ts = torch.arange(T_pred, dtype=torch.float32, device=DEVICE) * DT

# ---------------- Inference ----------------
with torch.no_grad():
    motion = model(z0, ts)  # (1, T_pred, 7)

motion = motion.squeeze(0).cpu().numpy()  # (T_pred, 7)

# ---------------- Print first few steps ----------------
print("Predicted motion (first 10 frames):")
for t, z in enumerate(motion[:10]):
    x, y, vx, vy, s, theta, l = z.tolist()
    print(f"t={t*DT:.2f}: x={x:.3f}, y={y:.3f}, vx={vx:.3f}, vy={vy:.3f}, s={s:.3f}, θ={theta:.3f}, l={l:.3f}")

# ---------------- Save trajectory ----------------
out_dir = Path("runs_nnd_fall")
out_dir.mkdir(parents=True, exist_ok=True)
torch.save(torch.from_numpy(motion), out_dir / "inference_motion.pt")
np.save(out_dir / "traj_world.npy", motion[:, :4])  # 世界坐标 (x,y,vx,vy)

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

np.save(out_dir / "traj_pixel.npy", traj_px)
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

np.save(out_dir / "flows_dxdy.npy", flows)
print(f"Optical flow saved to {out_dir/'flows_dxdy.npy'}, shape={flows.shape}")

# ---------------- Debug: print first 10 frames ----------------
print("\nFirst 10 frames (world, meters):")
for t in range(min(10, T_pred)):
    x, y, vx, vy = traj_world[t]
    print(f"t={t:02d}: x={x:.3f}m, y={y:.3f}m, vx={vx:.3f}m/f, vy={vy:.3f}m/f")

print("\nFirst 10 frames (pixel, px):")
for t in range(min(10, T_pred)):
    x, y, vx, vy = traj_px[t]
    print(f"t={t:02d}: x={x:.1f}px, y={y:.1f}px, vx={vx:.2f}px/f, vy={vy:.2f}px/f")



