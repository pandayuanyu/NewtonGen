import torch
import numpy as np
from pathlib import Path
from models.nnd_motion_pend import NewtonODELatent  # 你训练时用的模型文件

# ---------------- Config ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("/home/yuan418/data/project/Newtongen_ICLR/runs_nnd_pend/newton_ode_pend_final.pth")
DT = 0.02
T_pred = 48
H, W = 240, 360
METER_PER_PX = 0.05  # 每像素对应的米

# ---------------- Helper: construct z0 from pivot ----------------
def make_z0(pivot, pend_l, theta, omega, s=0.5, l=0.8, d=0.0):
    """
    Construct initial z0 from pendulum geometry.
    z0 = [x, y, vx, vy, theta, omega, s, l, d]
    """
    x = pivot[0] + pend_l * np.sin(theta)
    y = pivot[1] - pend_l * np.cos(theta)
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
pend_l = 8.0                    # 单摆摆长 (m)
theta = -0.2                     # 初始角度 (rad)
omega = 0.4                     # 初始角速度 (rad/s)
s_init = 0.1                     # 面积控制参数 s

z0 = make_z0(pivot, pend_l, theta, omega, s=s_init)
print("Initialized z0:", z0)

# ---------------- Time stamps ----------------
ts = torch.arange(T_pred, dtype=torch.float32, device=DEVICE) * DT

# ---------------- Inference ----------------
with torch.no_grad():
    motion = model(z0, ts)  # (1, T_pred, 9)
    print("motion",motion)
motion = motion.squeeze(0).cpu().numpy()  # (T_pred, 9)

# ---------------- Save trajectory ----------------
out_dir = Path("runs_nnd_pend")
out_dir.mkdir(parents=True, exist_ok=True)
torch.save(torch.from_numpy(motion), out_dir / "inference_motion_pend.pt")
np.save(out_dir / "motion_pend_world.npy", motion[:, :9])
print(f"Full trajectory saved to {out_dir/'inference_motion_pend.pt'}")

# ---------------- Convert to pixel coordinates ----------------
traj_world = motion[:, :4]  # x,y,vx,vy
traj_px = np.zeros_like(traj_world)
traj_px[:, 0] = traj_world[:, 0] / METER_PER_PX
traj_px[:, 1] = H - (traj_world[:, 1] / METER_PER_PX)
traj_px[:, 2] = traj_world[:, 2] / METER_PER_PX
traj_px[:, 3] = -traj_world[:, 3] / METER_PER_PX
np.save(out_dir / "traj_pixel_pend.npy", traj_px)

# ---------------- Shape mask function ----------------
def make_mask(shape, X, Y, cx, cy, scale):
    """
    shape: "circle","square","ellipse","triangle","diamond","pentagon","hexagon"
    scale: size factor from s
    """
    if shape == "circle":
        return (X - cx) ** 2 + (Y - cy) ** 2 <= scale**2
    elif shape == "square":
        return (np.abs(X - cx) <= scale) & (np.abs(Y - cy) <= scale)
    elif shape == "ellipse":
        return ((X - cx)**2)/(scale**2) + ((Y - cy)**2)/(scale*0.6)**2 <= 1
    elif shape == "triangle":
        Xr, Yr = X - cx, Y - cy
        h = np.sqrt(3) * scale
        return (Yr >= -h/2) & (Yr <= h/2) & (np.abs(Xr) <= (h/2 - Yr/np.sqrt(3)))
    elif shape == "diamond":
        return np.abs(X - cx) + np.abs(Y - cy) <= scale
    elif shape in ["pentagon","hexagon"]:
        n = 5 if shape=="pentagon" else 6
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        verts = np.stack([cx + scale*np.cos(angles), cy + scale*np.sin(angles)], axis=-1)
        path = Path(verts)
        coords = np.stack([X.ravel(), Y.ravel()], axis=-1)
        return path.contains_points(coords).reshape(X.shape)
    else:
        raise ValueError(f"Unknown shape: {shape}")

# ---------------- Build optical flow with shape -----------------
flows = np.zeros((T_pred, 2, H, W), dtype=np.float32)
Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
chosen_shape = "triangle"
s_param = np.sqrt(motion[:, 6])  # s 参数控制尺寸

for t in range(T_pred - 1):
    cx, cy = traj_px[t, 0], traj_px[t, 1]
    nx, ny = traj_px[t + 1, 0], traj_px[t + 1, 1]
    dx, dy = nx - cx, ny - cy
    scale = s_param[t] / METER_PER_PX  # 由 s 转换为像素单位
    mask = make_mask(chosen_shape, X, Y, cx, cy, scale)
    flows[t, 0, mask] = dx
    flows[t, 1, mask] = dy

np.save(out_dir / "flows_dxdy_pend.npy", flows)
print(f"Optical flow saved, shape={flows.shape}")



