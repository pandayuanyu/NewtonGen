#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motion-type driven Physics metrics (examples):
  --motion_type=1 → [ y,  ax, energy ]
  --motion_type=2 → [ y, -ax, energy ]             (deceleration)
  --motion_type=3 → [ y,  vx, energy ]
  --motion_type=4 → [ vx, ay, energy, energy_x_only ]
      where energy_x_only = 0.5*vx^2 + 9.8*y
  --motion_type=5 → rotational: [ omega_center, radius2, speed2 ]
      where
        theta_center(t) ∈ [0, 2π) = mod(atan2(y(t) - H/2*mpp, x(t) - W/2*mpp) + 2π, 2π)
        omega_center(t) = central-diff( unwrap(theta_center) ) / Δt
        radius2(t)      = (x(t) - W/2*mpp)^2 + (y(t) - H/2*mpp)^2
        speed2(t)       = vx(t)^2 + vy(t)^2
  --motion_type=6 → [ energy ]
  --motion_type=7 → [ omega, energy ]                   # omega = dθ/dt of long axis
  --motion_type=8 → [ vx, ay, omega, energy ]
  --motion_type=9 → [ ax, ay, energy ]
  --motion_type=10 → [ dl ]                             # dl = l[t+1] - l[t]
  --motion_type=11 → [ dl, ds ]                         # ds = s[t+1] - s[t]
  --motion_type=12 → : define below in get_metric_specs()

Per-video PIS-Norm (for each metric):
  pis_norm(metric) = 1 / (1 + std(metric) / (mean(metric) + eps))

Outputs (with CSV companions):
  1) features_pt: {filename: {"frame_indices": List[int], "features": Tensor[6, 9], "columns": [...]}}
     → CSV (per-frame): filename, frame_index, x,y,vx,vy,theta,omega,s,l,a
  2) physics_pt:  {filename: {"frame_indices": List[int], "physics": Tensor[6, M], "columns": [...], "formulae": {...}}}
     → CSV (per-frame): filename, frame_index, <metric1>, <metric2>, ...
  3) pisnorm_pt:  {filename: {"pis_norm": Tensor[M], "mean": Tensor[M], "std": Tensor[M], "metrics": [...], "eps": float}}
     → CSV (per-video): filename, for each metric: mean_*, std_*, pis_norm_*, then eps

Features (9 cols) per frame, in SI units (meters, seconds, radians):
  [ x, y, vx, vy, theta, omega, s, l, a ]
    x, y     : smoothed center (y flipped upward)
    vx, vy   : velocities via central differences
    theta    : orientation of long axis in [0, π), unwrapped for ω
    omega    : angular velocity (rad/s)
    s, l     : short/long side lengths of min-area rectangle
    a        : contour area

Notes
-----
• Smoothing uses an odd window (default 5).
• We compute all frames, then trim 3 frames at each end to avoid diff edge effects.
• We always *return exactly 6 frames* per video by sampling the middle of the valid region; if too few valid frames, we sample with replacement to length 6.
• If a frame has no contour, we insert zeros for that frame.
• delta_t: if not given, we read per-video FPS and set delta_t = 1/FPS.

"""

import os
import math
import argparse
import csv
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# --------------------------- Helpers ---------------------------

def get_video_fps(video_path: str, fallback: float = 30.0) -> float:
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else fallback
    finally:
        cap.release()


def read_mask_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Convert to grayscale if needed
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        # Threshold to binary mask (white object on black background)
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        frames.append(mask)
    cap.release()
    return frames


def moving_average_2d(x: np.ndarray, window: int) -> np.ndarray:
    assert window % 2 == 1, "smooth_window must be odd"
    N = len(x)
    out = np.copy(x)
    half = window // 2
    for i in range(N):
        s = max(0, i - half)
        e = min(N, i + half + 1)
        out[i] = np.mean(x[s:e], axis=0)
    return out


def unwrap_theta(theta: np.ndarray) -> np.ndarray:
    """Unwrap angles defined on [0, π) by doubling → unwrap → halving."""
    return 0.5 * np.unwrap(2.0 * theta)


def pick_six_indices(valid_len: int) -> List[int]:
    """Pick 6 indices from [0, valid_len-1], centered; if too short, sample with replacement."""
    if valid_len <= 0:
        return [0, 0, 0, 0, 0, 0]
    if valid_len >= 6:
        start = (valid_len - 6) // 2
        return list(range(start, start + 6))
    # fewer than 6 → spread across range and round
    pts = np.linspace(0, valid_len - 1, num=6)
    return [int(round(p)) for p in pts]


# --------------------------- Geometry & Physics ---------------------------

def per_frame_geometry(masks: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """Raw geometry arrays over all frames (pixel units except area in px^2)."""
    centers = []  # (cx, cy) in px (cy measured from top; we'll flip later)
    areas = []
    long_axes = []
    short_axes = []
    thetas = []   # radians in [0, π)

    if not masks:
        return {
            "centers": np.zeros((0, 2), dtype=float),
            "areas": np.zeros((0,), dtype=float),
            "long_axes": np.zeros((0,), dtype=float),
            "short_axes": np.zeros((0,), dtype=float),
            "thetas": np.zeros((0,), dtype=float),
        }

    H, W = masks[0].shape[:2]

    for mask in masks:
        # Ensure binary 0/255 uint8
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        mask_bin = (mask > 127).astype(np.uint8) * 255

        cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0:
            centers.append((0.0, 0.0))
            areas.append(0.0)
            long_axes.append(0.0)
            short_axes.append(0.0)
            thetas.append(0.0)
            continue

        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = M["m10"] / M["m00"] if M["m00"] > 0 else 0.0
        cy = M["m01"] / M["m00"] if M["m00"] > 0 else 0.0
        cy_up = H - cy  # flip so up is +y

        area_px = float(cv2.contourArea(cnt))

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.float32)
        edges = [box[1] - box[0], box[2] - box[1], box[3] - box[2], box[0] - box[3]]
        lengths = [float(np.linalg.norm(e)) for e in edges]
        max_idx = int(np.argmax(lengths))
        long_edge = edges[max_idx]
        angle = math.atan2(-long_edge[1], long_edge[0])
        if angle < 0:
            angle += math.pi  # [0, π)

        centers.append((cx, cy_up))
        areas.append(area_px)
        long_axes.append(lengths[max_idx])
        short_axes.append(lengths[int(np.argmin(lengths))])
        thetas.append(angle)

    return {
        "centers": np.asarray(centers, dtype=float),
        "areas": np.asarray(areas, dtype=float),
        "long_axes": np.asarray(long_axes, dtype=float),
        "short_axes": np.asarray(short_axes, dtype=float),
        "thetas": np.asarray(thetas, dtype=float),
    }


def compute_features_from_masks(masks: List[np.ndarray], mpp: float, delta_t: float, smooth_window: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Full-length feature arrays (not trimmed), plus raw arrays."""
    raw = per_frame_geometry(masks)

    centers = raw["centers"]            # px
    areas_px2 = raw["areas"]            # px^2
    long_axes_px = raw["long_axes"]     # px
    short_axes_px = raw["short_axes"]   # px
    theta = raw["thetas"]               # rad [0, π)

    if len(centers) == 0:
        return np.zeros((0, 9), dtype=float), raw

    centers_s = moving_average_2d(centers, smooth_window)

    # velocities in px/frame via central differences
    vel_px = np.zeros_like(centers_s)
    if len(centers_s) >= 3:
        vel_px[1:-1] = (centers_s[2:] - centers_s[:-2]) / 2.0

    # Convert to SI units
    centers_m = centers_s * mpp
    vel_m_per_frame = vel_px * mpp
    vel_m_per_s = vel_m_per_frame / delta_t

    # Orientation unwrap + angular velocity
    theta_u = unwrap_theta(theta)
    omega = np.zeros_like(theta_u)
    if len(theta_u) >= 3:
        omega[1:-1] = (theta_u[2:] - theta_u[:-2]) / (2.0 * delta_t)

    s_m = short_axes_px * mpp
    l_m = long_axes_px * mpp
    a_m2 = areas_px2 * (mpp ** 2)

    feats = np.concatenate([
        centers_m,             # x, y
        vel_m_per_s,           # vx, vy
        theta_u[:, None],      # theta
        omega[:, None],        # omega
        s_m[:, None],          # s
        l_m[:, None],          # l
        a_m2[:, None],         # a
    ], axis=1)

    return feats, raw


def trim_valid_region(feats: np.ndarray) -> np.ndarray:
    """Drop 3 frames at each end (derivative edge effects)."""
    if len(feats) <= 6:
        return feats.copy()
    return feats[3:-3]


def select_six_rows(feats_mid: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Return exactly 6 rows: pick a centered block if possible; otherwise sample with replacement.
    Also return the selected local indices (within the trimmed region)."""
    n = len(feats_mid)
    idx6 = pick_six_indices(n if n > 0 else 1)
    idx6 = [min(max(i, 0), max(n - 1, 0)) for i in idx6]
    return feats_mid[idx6], idx6


# --------------------------- Physics metrics config ---------------------------

def get_metric_specs(motion_type: int):
    """
    Return a list of (name, formula_str, compute_fn) where compute_fn takes a dict of arrays:
      arrays = { 'x','y','vx','vy','ax','ay','omega','omega_c','r2','speed2','theta_center','dl','ds' }
    and returns a numpy array shape (T,) for the selected frame indices.

    Define new motion types as needed.
    """
    def energy(arr):
        # Global energy definition: 0.5*(vx^2 + vy^2) + 9.8*y
        return 0.5*(arr['vx']**2 + arr['vy']**2) + 9.8*arr['y']

    if motion_type == 1:
        return [
            ("y", "y", lambda a: a['y']),
            ("ax", "ax", lambda a: a['ax']),
            ("ay", "ay", lambda a: a['ay']),
            ("energy", "0.5*(vx^2+vy^2)+9.8*y", energy),
        ]
    elif motion_type == 2:
        return [
            ("y", "y", lambda a: a['y']),
            ("neg_ax", "-ax", lambda a: -a['ax']),
            ("ay", "ay", lambda a: a['ay']),  
            ("energy", "0.5*(vx^2+vy^2)+9.8*y", energy),
        ]
    elif motion_type == 3:
        return [
            ("y", "y", lambda a: a['y']),
            ("vx", "vx", lambda a: a['vx']),
            ("energy", "0.5*(vx^2+vy^2)+9.8*y", energy),
        ]
    elif motion_type == 4:
        return [
            ("vx", "vx", lambda a: a['vx']),
            ("ay", "ay", lambda a: a['ay']),
            ("energy", "0.5*(vx^2+vy^2)+9.8*y", energy),
            ("energy_x_only", "0.5*vx^2+9.8*y", lambda a: 0.5*(a['vx']**2) + 9.8*a['y']),
        ]
    elif motion_type == 5:
        # Rotational metrics around image center
        return [
            ("omega_center", "d/dt unwrap(mod(atan2(y-yc,x-xc)+2π,2π))", lambda a: a['omega_c']),
            ("radius2", "(x-xc)^2 + (y-yc)^2", lambda a: a['r2']),
            ("speed2", "vx^2 + vy^2", lambda a: a['speed2']),
        ]
    elif motion_type == 6:
        return [
        ("ax", "ax", lambda a: a['ax']),
        ("ay", "ay", lambda a: a['ay']),
        ("energy", "0.5*(vx^2+vy^2)+9.8*y", energy),
        ]
    
    elif motion_type == 7:
        return [
            ("omega", "dθ/dt (long axis)", lambda a: a['omega']),
            ("energy", "0.5*(vx^2+vy^2)+9.8*y", energy),
        ]
    elif motion_type == 8:
        return [
            ("vx", "vx", lambda a: a['vx']),
            ("ay", "ay", lambda a: a['ay']),
            ("omega", "dθ/dt (long axis)", lambda a: a['omega']),
            ("energy", "0.5*(vx^2+vy^2)+9.8*y", energy),
        ]
    elif motion_type == 9:
        return [
            ("ax", "ax", lambda a: a['ax']),
            ("ay", "ay", lambda a: a['ay']),
            ("energy", "0.5*(vx^2+vy^2)+9.8*y", energy),
        ]
    elif motion_type == 10:
        return [
            ("dl", "l[t+1]-l[t]", lambda a: a['dl']),
        ]
    elif motion_type == 11:
        return [
            ("dl", "l[t+1]-l[t]", lambda a: a['dl']),
            ("ds", "s[t+1]-s[t]", lambda a: a['ds']),
        ]
    elif motion_type == 12:
        return [
            ("ax", "ax", lambda a: a['ax']),
            ("ay", "ay", lambda a: a['ay']),
            ("energy", "0.5*(vx^2+vy^2)+9.8*y", energy),
        ]
    elif motion_type == 13:
        return [
            ("vx", "vx", lambda a: a['vx']),
            ("vy", "vy", lambda a: a['vy']),
            ("dl", "l[t+1]-l[t]", lambda a: a['dl']),
            ("energy", "0.5*(vx^2+vy^2)+9.8*y", energy),
        ]

    else:
        raise ValueError(f"Unsupported --motion_type={motion_type}. Please add a case in get_metric_specs().")


# --------------------------- CSV Writers ---------------------------

def derive_csv_path(pt_path: str, override: Optional[str]) -> str:
    if override:
        return override
    base, _ = os.path.splitext(pt_path)
    return base + ".csv"


def write_features_csv(path: str, features_out: Dict[str, Dict[str, object]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "frame_index", "x", "y", "vx", "vy", "theta", "omega", "s", "l", "a"])
        for fname, payload in features_out.items():
            feats = payload["features"].tolist() if isinstance(payload["features"], torch.Tensor) else payload["features"]
            frame_idx = payload["frame_indices"]
            for i, row in enumerate(feats):
                w.writerow([fname, frame_idx[i]] + list(map(float, row)))


def write_physics_csv(path: str, physics_out: Dict[str, Dict[str, object]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Build header dynamically from any one entry
    any_payload = next(iter(physics_out.values())) if physics_out else None
    cols = any_payload.get("columns", []) if any_payload else []
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "frame_index"] + cols)
        for fname, payload in physics_out.items():
            physics = payload["physics"].tolist() if isinstance(payload["physics"], torch.Tensor) else payload["physics"]
            frame_idx = payload["frame_indices"]
            for i, row in enumerate(physics):
                w.writerow([fname, frame_idx[i]] + [float(x) for x in row])


def write_pisnorm_csv(path: str, pisnorm_out: Dict[str, Dict[str, object]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Build header dynamically from metrics list
    any_payload = next(iter(pisnorm_out.values())) if pisnorm_out else None
    metrics = any_payload.get("metrics", []) if any_payload else []
    header = ["filename"]
    for m in metrics:
        header += [f"mean_{m}", f"std_{m}", f"pis_norm_{m}"]
    header += ["eps"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for fname, payload in pisnorm_out.items():
            mean = payload["mean"].tolist() if isinstance(payload["mean"], torch.Tensor) else payload["mean"]
            std = payload["std"].tolist() if isinstance(payload["std"], torch.Tensor) else payload["std"]
            pisn = payload["pis_norm"].tolist() if isinstance(payload["pis_norm"], torch.Tensor) else payload["pis_norm"]
            row = [fname]
            for i in range(len(mean)):
                row += [float(mean[i]), float(std[i]), float(pisn[i])]
            row += [float(payload["eps"])]
            w.writerow(row)


# --------------------------- Main ---------------------------

def process_video(video_path: str, mpp: float, delta_t: Optional[float], smooth_window: int):
    masks = read_mask_frames(video_path)
    if delta_t is None:
        fps = get_video_fps(video_path)
        delta_t_eff = 1.0 / fps
    else:
        delta_t_eff = delta_t

    feats_full, _ = compute_features_from_masks(masks, mpp=mpp, delta_t=delta_t_eff, smooth_window=smooth_window)

    # Full timeline signals in meters / m/s (and derived)
    if feats_full.shape[0] == 0:
        x_full = np.zeros((0,), dtype=float)
        y_full = np.zeros((0,), dtype=float)
        vx_full = np.zeros((0,), dtype=float)
        vy_full = np.zeros((0,), dtype=float)
        omega_full = np.zeros((0,), dtype=float)
        s_full = np.zeros((0,), dtype=float)
        l_full = np.zeros((0,), dtype=float)
    else:
        x_full  = feats_full[:, 0]
        y_full  = feats_full[:, 1]
        vx_full = feats_full[:, 2]
        vy_full = feats_full[:, 3]
        omega_full = feats_full[:, 5]
        s_full = feats_full[:, 6]
        l_full = feats_full[:, 7]

    # Accelerations (central difference); endpoints left at 0
    ax_full = np.zeros_like(vx_full)
    ay_full = np.zeros_like(vy_full)
    if vx_full.shape[0] >= 3:
        ax_full[1:-1] = (vx_full[2:] - vx_full[:-2]) / (2.0 * delta_t_eff)
    if vy_full.shape[0] >= 3:
        ay_full[1:-1] = (vy_full[2:] - vy_full[:-2]) / (2.0 * delta_t_eff)

    # Forward differences for lengths (dl, ds)
    dl_full = np.zeros_like(l_full)
    ds_full = np.zeros_like(s_full)
    if l_full.shape[0] >= 2:
        dl_full[:-1] = l_full[1:] - l_full[:-1]
    if s_full.shape[0] >= 2:
        ds_full[:-1] = s_full[1:] - s_full[:-1]

    # Image center in meters (origin bottom-left) for motion_type=5
    if masks:
        H, W = masks[0].shape[:2]
    else:
        H, W = 0, 0
    cx_m = (W * 0.5) * mpp
    cy_m = (H * 0.5) * mpp

    # Angle to image center and its angular velocity (θ in [0, 2π))
    if x_full.shape[0] > 0:
        theta_c_base = np.arctan2(y_full - cy_m, x_full - cx_m)   # [-π, π]
        theta_c_02pi = np.mod(theta_c_base + 2.0*np.pi, 2.0*np.pi)  # [0, 2π)
        theta_c_u = np.unwrap(theta_c_02pi)
        omega_c_full = np.zeros_like(theta_c_u)
        if theta_c_u.shape[0] >= 3:
            omega_c_full[1:-1] = (theta_c_u[2:] - theta_c_u[:-2]) / (2.0 * delta_t_eff)
    else:
        theta_c_02pi = np.zeros((0,), dtype=float)
        omega_c_full = np.zeros((0,), dtype=float)

    # Radius^2 and speed^2
    r2_full = (x_full - cx_m) ** 2 + (y_full - cy_m) ** 2
    speed2_full = vx_full ** 2 + vy_full ** 2

    feats_mid = trim_valid_region(feats_full)
    feats_6, local_idx = select_six_rows(feats_mid)

    base = 3 if len(feats_full) > 6 else 0
    frame_indices = [base + i for i in local_idx]

    # Helper to pick frames
    def pick(arr):
        return arr[frame_indices] if arr.shape[0] > 0 else np.zeros((len(frame_indices),), dtype=float)

    arrays_sel = {
        'x':  pick(x_full),
        'y':  pick(y_full),
        'vx': pick(vx_full),
        'vy': pick(vy_full),
        'ax': pick(ax_full),
        'ay': pick(ay_full),
        'omega': pick(omega_full),           # long-axis angular velocity
        'omega_c': pick(omega_c_full),       # center-angle angular velocity
        'r2': pick(r2_full),
        'speed2': pick(speed2_full),
        'theta_center': pick(theta_c_02pi),
        'dl': pick(dl_full),
        'ds': pick(ds_full),
    }

    return feats_6, frame_indices, delta_t_eff, arrays_sel


def main():
    ap = argparse.ArgumentParser(description="Physical features + motion-type Physics scores + PIS-Norm from masked videos (6 frames/video)")
    ap.add_argument("--input_dir", required=True, help="Folder containing masked .mp4 videos (white on black)")
    ap.add_argument("--features_pt", required=True, help="Path to save features dict .pt")
    ap.add_argument("--physics_pt", required=True, help="Path to save Physics-scores dict .pt (per-frame)")
    ap.add_argument("--pisnorm_pt", required=True, help="Path to save PIS-Norm dict .pt (per-video)")

    # CSV (optional): if omitted, will write alongside the .pt with .csv extension
    ap.add_argument("--features_csv", default=None, help="CSV path for features (per-frame)")
    ap.add_argument("--physics_csv", default=None, help="CSV path for Physics scores (per-frame)")
    ap.add_argument("--pisnorm_csv", default=None, help="CSV path for PIS-Norm (per-video)")

    ap.add_argument("--motion_type", type=int, default=1, help="Selects which physics metrics to compute (1..12)")

    ap.add_argument("--mpp", type=float, default=1/12, help="Meters per pixel")
    ap.add_argument("--delta_t", type=float, default=None, help="Seconds between frames; if omitted, uses 1/FPS")
    ap.add_argument("--smooth_window", type=int, default=5, help="Odd window size for center smoothing")
    ap.add_argument("--eps", type=float, default=1e-8, help="Small constant in PIS-Norm: 1/(1 + std/(mean+eps))")

    args = ap.parse_args()

    assert args.smooth_window % 2 == 1, "--smooth_window must be odd"

    os.makedirs(os.path.dirname(args.features_pt), exist_ok=True)
    os.makedirs(os.path.dirname(args.physics_pt), exist_ok=True)
    os.makedirs(os.path.dirname(args.pisnorm_pt), exist_ok=True)

    features_out: Dict[str, Dict[str, object]] = {}
    physics_out: Dict[str, Dict[str, object]] = {}
    pisnorm_out: Dict[str, Dict[str, object]] = {}

    videos = [f for f in sorted(os.listdir(args.input_dir)) if f.lower().endswith('.mp4')]
    if not videos:
        raise SystemExit(f"No .mp4 files found in {args.input_dir}")

    # Configure metrics for this motion type
    specs = get_metric_specs(args.motion_type)
    metric_names = [name for name, _, _ in specs]

    for fname in videos:
        vpath = os.path.join(args.input_dir, fname)
        feats6, frame_idx6, dt_eff, arrays_sel = process_video(
            vpath, mpp=args.mpp, delta_t=args.delta_t, smooth_window=args.smooth_window
        )

        # Ensure exactly 6 rows for features
        if feats6.shape[0] != 6:
            if feats6.shape[0] == 0:
                feats6 = np.zeros((1, 9), dtype=float)
                frame_idx6 = [0]
            last_f = feats6[-1:]
            while feats6.shape[0] < 6:
                feats6 = np.vstack([feats6, last_f])
                frame_idx6.append(frame_idx6[-1])
            feats6 = feats6[:6]
            frame_idx6 = frame_idx6[:6]

        feats_tensor = torch.tensor(feats6, dtype=torch.float32)

        # Build physics matrix (6, M) according to specs
        physics_cols = []
        formulas = {}
        for name, formula_str, fn in specs:
            formulas[name] = formula_str
            physics_cols.append(fn(arrays_sel))
        physics_mat = np.stack(physics_cols, axis=1) if len(physics_cols) else np.zeros((len(frame_idx6), 0), dtype=float)
        physics_tensor = torch.tensor(physics_mat, dtype=torch.float32)

        features_out[fname] = {
            "frame_indices": frame_idx6,
            "features": feats_tensor,
            "columns": ["x", "y", "vx", "vy", "theta", "omega", "s", "l", "a"],
            "meta": {"mpp": args.mpp, "delta_t": dt_eff, "smooth_window": args.smooth_window},
        }

        physics_out[fname] = {
            "frame_indices": frame_idx6,
            "physics": physics_tensor,  # Tensor[6, M]
            "columns": metric_names,
            "formulae": formulas,
        }

        # PIS-Norm per video for each metric
        if physics_tensor.numel() == 0:
            mean_vec = np.zeros((0,), dtype=float)
            std_vec = np.zeros((0,), dtype=float)
            pisn_vec = np.zeros((0,), dtype=float)
        else:
            ph = physics_tensor.detach().cpu().numpy().astype(float)  # (6, M)
            mean_vec = np.mean(ph, axis=0)
            std_vec = np.std(ph, axis=0)
            pisn_vec = 1.0 / (1.0 + (std_vec / (np.abs(mean_vec) + args.eps)))

        pisnorm_out[fname] = {
            "pis_norm": torch.tensor(pisnorm_vec := pisn_vec, dtype=torch.float32),
            "mean": torch.tensor(mean_vec, dtype=torch.float32),
            "std": torch.tensor(std_vec, dtype=torch.float32),
            "metrics": metric_names,
            "eps": float(args.eps),
        }

        print(
            f"Processed {fname}: 6 frames → features + {len(metric_names)} physics metrics + PIS-Norms "
            f"(Δt={dt_eff:.6f}s, metrics={metric_names}, PIS-Norms={pisn_vec})"
        )

    # Save PT
    torch.save(features_out, args.features_pt)
    torch.save(physics_out, args.physics_pt)
    torch.save(pisnorm_out, args.pisnorm_pt)

    # Save CSVs (derive paths if not provided)
    features_csv_path = derive_csv_path(args.features_pt, getattr(args, 'features_csv', None))
    physics_csv_path = derive_csv_path(args.physics_pt, getattr(args, 'physics_csv', None))
    pisnorm_csv_path = derive_csv_path(args.pisnorm_pt, getattr(args, 'pisnorm_csv', None))

    write_features_csv(features_csv_path, features_out)
    write_physics_csv(physics_csv_path, physics_out)
    write_pisnorm_csv(pisnorm_csv_path, pisnorm_out)

    print(
        f"Saved features:   {args.features_pt}"
        f"Saved physics:    {args.physics_pt}"
        f"Saved PIS-Norm:   {args.pisnorm_pt}"
        f"CSV (features):   {features_csv_path}"
        f"CSV (physics):    {physics_csv_path}"
        f"CSV (PIS-Norm):   {pisnorm_csv_path}"
    )


if __name__ == "__main__":
    main()

