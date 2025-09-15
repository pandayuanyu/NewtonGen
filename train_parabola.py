
"""
Train NewtonODELatent using precomputed .pt
Supports mini-batch training, weighted loss, and periodic checkpoints.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from pathlib import Path
from models.nnd import NewtonODELatent  # nnd.py 文件路径

# ---------------- Hyperparameters ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100000
LR_INIT = 1e-4
BATCH_SIZE = 64
SAVE_EVERY = 1000
OUT_DIR = Path("runs/parabola")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Load latent dynamics data ----------------
dynamics_batch = torch.load(
    "/home/yuan418/data/project/Newtongen_ICLR/physicsl_label/unified/labels/physical_parabola.pt"
)  # shape (B_total,T,C)
B_total, T, C = dynamics_batch.shape
print(f"Loaded dynamics_batch.pt: shape {dynamics_batch.shape}")

#[x,y,vx,vy,theta,omega,s,l]
dynamics_batch = dynamics_batch[:, :, :9]
z0_data = dynamics_batch[:, 0, :]  # 初始状态 (B_total, 9) (x y vx vy theta omega s l a)

# ---------------- Time tensor ----------------
dt = 0.01
ts = torch.arange(T, dtype=torch.float32) * dt  # shape [T]

# ---------------- Model & optimizer ----------------
model = NewtonODELatent().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR_INIT)
scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS // 5, eta_min=1e-6
)

dynamics_batch = dynamics_batch.to(DEVICE)
z0_data = z0_data.to(DEVICE)
ts = ts.to(DEVICE)

# ---------------- Weighted MSE function ----------------
weights = torch.tensor([1.01, 1.01, 1.01, 1.01,   # (x,y,vx,vy)
                        0.01, 0.01,             # (theta, omega)
                        0.1, 0.1, 0.1],      # (s, l, a)
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
        dynamics_batch_b = dynamics_batch[idx]   # (batch_size,T,9)

        # Forward
        dynamics_pred = model(z0_batch, ts)    # (batch_size, T, 9)
        loss = weighted_mse(dynamics_pred, dynamics_batch_b)

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
        print(f"a_x={func.a_x.item():.4f}, b_x={func.b_x.item():.4f}, c_x={func.c_x.item():.4f}")
        print(f"a_y={func.a_y.item():.4f}, b_y={func.b_y.item():.4f}, c_y={func.c_y.item():.4f}")
        print(f"g_over_L={func.g_over_L.item():.4f}, gamma={func.gamma.item():.4f}")
        print(f"alpha_s={func.alpha_s.item():.4f}, beta_s={func.beta_s.item():.4f}")
        print(f"alpha_l={func.alpha_l.item():.4f}, beta_l={func.beta_l.item():.4f}")
        print(f"alpha_a={func.alpha_a.item():.4f}, beta_a={func.beta_a.item():.4f}")




        # 保存 checkpoint
        ckpt_path = OUT_DIR / f"learnedODE_parabola_epoch{epoch:05d}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

# ---------------- Save final model ----------------
final_path = OUT_DIR / "earnedODE_parabola_final.pth"
torch.save(model.state_dict(), final_path)
print(f"Training complete. Final model saved at {final_path}")

