"""
Train NewtonODELatent using precomputed .pt
Supports mini-batch training, weighted loss, periodic checkpoints,
and validates with plots (3x3 combined + individual per parameter) with fixed y-axis per parameter.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from pathlib import Path
from models.nnd import NewtonODELatent
import matplotlib.pyplot as plt

# ---------------- Hyperparameters ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100000
LR_INIT = 1e-4
BATCH_SIZE = 64
SAVE_EVERY = 1000
OUT_DIR = Path("runs/spring")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Load latent dynamics data ----------------
dynamics_batch = torch.load(
    "/home/yuan418/data/project/Newtongen_ICLR/physicsl_label/unified/labels/physical_spring.pt"
)  # shape (B_total,T,C)
B_total, T, C = dynamics_batch.shape
print(f"Loaded training dynamics_batch.pt: shape {dynamics_batch.shape}")

# [x,y,vx,vy,theta,omega,s,l,a]
dynamics_batch = dynamics_batch[:, :, :9]
z0_data = dynamics_batch[:, 0, :]  # 初始状态 (B_total, 9)

# ---------------- Time tensor ----------------
dt = 0.01
ts = torch.arange(T, dtype=torch.float32) * dt  # shape [T]

# ---------------- Model & optimizer ----------------
model = NewtonODELatent().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR_INIT)
scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS // 2, eta_min=1e-6
)

dynamics_batch = dynamics_batch.to(DEVICE)
z0_data = z0_data.to(DEVICE)
ts = ts.to(DEVICE)

# ---------------- Weighted MSE function ----------------
weights = torch.tensor([1.01, 1.01, 1.01, 1.01,   # (x,y,vx,vy)
                        0.01, 0.01,              # (theta, omega)
                        0.1, 0.1, 0.1],          # (s, l, a)
                       device=DEVICE)

def weighted_mse(pred, target):
    diff = (pred - target) ** 2
    diff = diff * weights
    return diff.mean()

param_names = ["x", "y", "vx", "vy", "theta", "omega", "s", "l", "a"]

# ---------------- Custom y-axis ranges for each z ----------------
y_ranges = {
    "x": (0, 18),
    "y": (0, 12),
    "vx": (-1, 15),
    "vy": (-15, 1),
    "theta": (-3.14, 3.14),
    "omega": (-3, 3),
    "s": (0, 2),
    "l": (0, 5),
    "a": (0, 2),
}

# ---------------- Training loop ----------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(B_total)

    epoch_loss = 0.0
    for i in range(0, B_total, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        z0_batch = z0_data[idx]                  # (batch_size,9)
        dynamics_batch_b = dynamics_batch[idx]   # (batch_size,T,9)

        # Forward
        dynamics_pred = model(z0_batch, ts)      # (batch_size, T, 9)
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

        # ---------------- Validation ----------------
        model.eval()
        with torch.no_grad():
            dynamics_batch_val = torch.load(
                "/home/yuan418/data/project/Newtongen_ICLR/physicsl_label/unified/labels_val/physical_gt_spring.pt"
            )
            dynamics_batch_val = dynamics_batch_val[:, :, :9].to(DEVICE)
            z0_val = dynamics_batch_val[:, 0, :]

            # 推理
            dynamics_pred_val = model(z0_val, ts)  # (B_val, T, 9)

            # 取第一个样本做保存和可视化
            pred_np = dynamics_pred_val[0].cpu().numpy()
            gt_np   = dynamics_batch_val[0].cpu().numpy()

            # 保存 txt
            txt_path = OUT_DIR / f"val_pred_epoch{epoch:05d}.txt"
            with open(txt_path, "w") as f:
                f.write("t " + " ".join([f"pred_{i}" for i in range(9)]) + " " +
                        " ".join([f"gt_{i}" for i in range(9)]) + "\n")
                for t_idx in range(pred_np.shape[0]):
                    line = f"{ts[t_idx].item():.4f} " + \
                           " ".join([f"{v:.6f}" for v in pred_np[t_idx]]) + " " + \
                           " ".join([f"{v:.6f}" for v in gt_np[t_idx]]) + "\n"
                    f.write(line)
            print(f"Validation results saved to {txt_path}")

            # ---- 3x3 combined plot ----
            fig, axes = plt.subplots(3, 3, figsize=(15, 10))
            for i in range(9):
                ax = axes[i // 3, i % 3]
                ax.plot(ts.cpu(), gt_np[:, i], label="GT")
                ax.plot(ts.cpu(), pred_np[:, i], "--", label="Pred")
                ax.set_title(param_names[i])
                ax.legend()
                # 使用预定义 y 轴范围
                if param_names[i] in y_ranges:
                    ax.set_ylim(y_ranges[param_names[i]])
            fig.tight_layout()
            fig_path = OUT_DIR / f"val_plot_epoch{epoch:05d}.png"
            fig.savefig(fig_path)
            plt.close(fig)
            print(f"Validation plot saved to {fig_path}")

            # ---- individual parameter plots ----
            single_plot_dir = OUT_DIR / f"val_single_plots_epoch{epoch:05d}"
            single_plot_dir.mkdir(parents=True, exist_ok=True)
            for i in range(9):
                plt.figure(figsize=(6,4))
                plt.plot(ts.cpu(), gt_np[:, i], label="GT")
                plt.plot(ts.cpu(), pred_np[:, i], "--", label="Pred")
                plt.title(param_names[i])
                plt.xlabel("time")
                plt.ylabel(param_names[i])
                plt.legend()
                if param_names[i] in y_ranges:
                    plt.ylim(y_ranges[param_names[i]])
                plt.tight_layout()
                plt.savefig(single_plot_dir / f"{param_names[i]}.png")
                plt.close()
            print(f"Individual parameter plots saved to {single_plot_dir}")

        # ---------------- Save checkpoint ----------------
        ckpt_path = OUT_DIR / f"learnedODE_spring_epoch{epoch:05d}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

# ---------------- Save final model ----------------
final_path = OUT_DIR / "learnedODE_spring_final.pth"
torch.save(model.state_dict(), final_path)
print(f"Training complete. Final model saved at {final_path}")
