
"""
Train NewtonODELatent using precomputed .pt
Supports mini-batch training, weighted loss, and periodic checkpoints.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from pathlib import Path
from models.nnd_motion_pend import NewtonODELatent  # nnd.py 文件路径

# ---------------- Hyperparameters ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100000
LR_INIT = 1e-4
BATCH_SIZE = 64
SAVE_EVERY = 1000
OUT_DIR = Path("runs_nnd_pend")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Load motion data ----------------
motion_batch = torch.load(
    "/home/yuan418/data/project/Newtongen_ICLR/physicsl_label/physical_pendulum50.pt"
)  # shape (B_total,T,C)
B_total, T, C = motion_batch.shape
print(f"Loaded motion_batch.pt: shape {motion_batch.shape}")

# 取前7维 [x,y,vx,vy,s,theta,l]
motion_batch = motion_batch[:, :, :9]
z0_data = motion_batch[:, 0, :]  # 初始状态 (B_total, 9)

# ---------------- Time tensor ----------------
dt = 0.01
ts = torch.arange(T, dtype=torch.float32) * dt  # shape [T]

# ---------------- Model & optimizer ----------------
model = NewtonODELatent().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR_INIT)
scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS // 5, eta_min=1e-6
)

motion_batch = motion_batch.to(DEVICE)
z0_data = z0_data.to(DEVICE)
ts = ts.to(DEVICE)

# ---------------- Weighted MSE function ----------------
weights = torch.tensor([0.01, 0.01, 0.01, 0.01,   # (x,y,vx,vy)
                        10.0, 10.0,             # (theta, omega)
                        0.01, 0.01, 0.01],      # (s,l, d)
                       device=DEVICE)

def weighted_mse(pred, target):
    diff = (pred - target) ** 2
    diff = diff * weights  # 按维度加权
    return diff.mean()

# ---------------- Training loop ----------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(B_total)

    epoch_loss = 0.0
    for i in range(0, B_total, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        z0_batch = z0_data[idx]              # (batch_size,9)
        motion_batch_b = motion_batch[idx]   # (batch_size,T,9)

        # Forward
        motion_pred = model(z0_batch, ts)    # (batch_size, T, 9)
        loss = weighted_mse(motion_pred, motion_batch_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * z0_batch.size(0)

    scheduler.step()
    epoch_loss /= B_total

    # ---- logging & checkpoint ----
    if epoch % SAVE_EVERY == 0 or epoch == 1:
        print(f"[Epoch {epoch:05d}/{EPOCHS}] loss={epoch_loss:.6f}")
        func = model.func
        print("Current ODE parameters:")
        print(f"g_over_L={func.g_over_L.item():.4f}, gamma={func.gamma.item():.4f}")

        # 保存 checkpoint
        ckpt_path = OUT_DIR / f"newton_ode_pend_epoch{epoch:05d}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

# ---------------- Save final model ----------------
final_path = OUT_DIR / "newton_ode_pend_final.pth"
torch.save(model.state_dict(), final_path)
print(f"Training complete. Final model saved at {final_path}")

