#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end SAM2 pipeline:
1) Extract first frames for all videos in a folder.
2) (Optional) Click to record object coordinates on each first frame -> CSV.
3) Run SAM2 segmentation on every video using those clicks -> per-video mask MP4s.

Usage (cmd.exe example):
  conda activate VideoGen
  python sam2_pipeline.py --videos ./data/newton_parabola/videos --output ./masked_videos

Disable interactive clicking (use an existing CSV):
  python sam2_pipeline.py --videos ./data/newton_parabola/videos --output ./masked_videos --no-clicks --csv E:/data/first_frames/click_coords.csv
"""

import os
import csv
import cv2
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

# Hugging Face Transformers (SAM2)
from transformers import Sam2VideoModel, Sam2VideoProcessor, infer_device
from transformers.video_utils import load_video

# -----------------------------
# Environment (edit if you want)
# -----------------------------
os.environ.setdefault("TRANSFORMERS_CACHE", r"")
os.environ.setdefault("TORCH_HOME",        r"")

# -----------------------------
# Constants
# -----------------------------
MODEL_ID   = "facebook/sam2.1-hiera-tiny"
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
IMG_EXTS   = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# =============================
# Helpers: discovery + extraction
# =============================
def is_video(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS

def extract_first_frames(videos_dir: Path, out_frames_dir: Path, recursive: bool = False) -> int:
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    files = (videos_dir.rglob("*") if recursive else videos_dir.iterdir())
    videos = sorted([p for p in files if is_video(p)])
    n_ok = 0
    for vid in videos:
        out_png = out_frames_dir / f"{vid.stem}_frame0.png"
        cap = cv2.VideoCapture(str(vid))
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            cap.release()
            if ok and cv2.imwrite(str(out_png), frame):
                print(f"[frame0] {vid.name} -> {out_png.name}")
                n_ok += 1
                continue
        # Fallback via ffmpeg if available
        if _ffmpeg_first_frame(vid, out_png):
            print(f"[frame0-ffmpeg] {vid.name} -> {out_png.name}")
            n_ok += 1
        else:
            print(f"[WARN] Could not read first frame: {vid.name}")
    return n_ok

def _ffmpeg_first_frame(video_path: Path, out_png: Path) -> bool:
    import shutil, subprocess
    if shutil.which("ffmpeg") is None:
        return False
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", "select=eq(n,0)", "-vsync", "0", str(out_png)
    ]
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0 and out_png.exists()
    except Exception:
        return False

# =============================
# Helpers: click-to-label UI
# =============================
def _list_images(folder: Path, only_frame0: bool = True):
    imgs = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            if (not only_frame0) or p.stem.lower().endswith("_frame0"):
                imgs.append(p)
    return imgs

def _draw_points(img, pts):
    vis = img.copy()
    for i, (x, y) in enumerate(pts):
        cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(vis, str(i), (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                    cv2.LINE_AA)
    return vis

def run_click_labeler(frames_dir: Path, out_csv: Path) -> int:
    imgs = _list_images(frames_dir, only_frame0=True)
    if not imgs:
        print("[Labeler] No images found to label.")
        return 0

    rows = []  # (filename, point_idx, x, y)
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)

    for idx, img_path in enumerate(imgs, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[Labeler][SKIP] Could not read {img_path.name}")
            continue

        points = []

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((int(x), int(y)))

        cv2.setMouseCallback("image", on_click)

        key = -1
        while True:
            vis = _draw_points(img, points)
            title = (f"[{idx}/{len(imgs)}] {img_path.name}  |  "
                     f"Clicks: {len(points)}   (Enter/Space/N=Next,  U=Undo,  R=Reset,  Q/Esc=Quit)")
            cv2.imshow("image", vis)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 32, ord('n')):  # Enter, Space, 'n'
                for pi, (x, y) in enumerate(points):
                    rows.append((img_path.name, pi, x, y))
                if not points:
                    print(f"[Labeler] No clicks saved for {img_path.name}")
                break
            elif key in (ord('u'), 8):  # 'u' or Backspace
                if points:
                    points.pop()
            elif key == ord('r'):       # 'r' reset
                points.clear()
            elif key in (27, ord('q')): # Esc or 'q'
                for pi, (x, y) in enumerate(points):
                    rows.append((img_path.name, pi, x, y))
                idx = len(imgs)
                break

        if key in (27, ord('q')):
            print("[Labeler] Early quit requested.")
            break

    cv2.destroyAllWindows()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "point_idx", "x", "y"])
        w.writerows(rows)

    print(f"[Labeler] Saved {len(rows)} points -> {out_csv}")
    return len(rows)

# =============================
# Helpers: CSV + mapping to video
# =============================
def load_points_csv(csv_path: Path):
    pts = defaultdict(list)  # key -> list[(point_idx, x, y)]
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"filename", "x", "y"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must have columns: filename, point_idx(optional), x, y")
        for row in reader:
            fn = row["filename"].strip()
            pi = int(row["point_idx"]) if (row.get("point_idx") and row["point_idx"].strip()) else len(pts[fn])
            x = int(float(row["x"]))
            y = int(float(row["y"]))
            pts[fn].append((pi, x, y))
    for k in pts:
        pts[k].sort(key=lambda t: t[0])
    return pts

def map_points_to_video(video_path: Path, points_by_key: dict):
    # allow: video name, stem, stem_frame0.png/jpg/...
    candidates = [
        video_path.name,
        video_path.stem,
        f"{video_path.stem}_frame0.png",
        f"{video_path.stem}_frame0.jpg",
        f"{video_path.stem}_frame0.jpeg",
        f"{video_path.stem}_frame0.bmp",
        f"{video_path.stem}_frame0.tif",
        f"{video_path.stem}_frame0.tiff",
    ]
    for key in candidates:
        if key in points_by_key:
            return [(x, y) for _, x, y in points_by_key[key]]
    # case-insensitive fallback
    for key in points_by_key:
        kl = key.lower()
        if kl == video_path.name.lower() or kl == video_path.stem.lower():
            return [(x, y) for _, x, y in points_by_key[key]]
        if kl.startswith(video_path.stem.lower()) and "frame0" in kl:
            return [(x, y) for _, x, y in points_by_key[key]]
    return []

def make_nested_points(points_xy):
    # SAM2 expects: input_points = [[ [ [x1,y1], [x2,y2], ... ] ]], input_labels = [[[1,1,...]]]
    if not points_xy:
        return [[[]]], [[[]]]
    return [[ [ [int(x), int(y)] for (x, y) in points_xy ] ]], [[ [1]*len(points_xy) ]]

def find_fps(video_path: str, info_dict):
    fps = None
    if isinstance(info_dict, dict):
        fps = info_dict.get("fps", None)
    if fps is None:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()
    return float(fps)

# =============================
# SAM2 processing
# =============================
def make_mask_videos(videos_dir: Path, csv_path: Path, out_dir: Path, backend: str = "opencv"):
    out_dir.mkdir(parents=True, exist_ok=True)

    points_by_key = load_points_csv(csv_path)
    videos = sorted([p for p in videos_dir.iterdir() if is_video(p)])
    if not videos:
        print(f"[WARN] No videos found in: {videos_dir}")
        return 0

    device = infer_device()
    dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model      = Sam2VideoModel.from_pretrained(MODEL_ID).to(device, dtype=dtype)
    processor  = Sam2VideoProcessor.from_pretrained(MODEL_ID)

    n_done = 0
    for vid in videos:
        pts_xy = map_points_to_video(vid, points_by_key)
        if not pts_xy:
            print(f"[SKIP] No points for {vid.name}")
            continue

        input_points, input_labels = make_nested_points(pts_xy)
        ann_frame_idx = 0
        obj_id = 1

        video_frames, info = load_video(str(vid), backend=backend)
        fps = find_fps(str(vid), info)

        inference_session = processor.init_video_session(
            video=video_frames, inference_device=device, dtype=dtype
        )

        processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=obj_id,
            input_points=input_points,
            input_labels=input_labels,
        )

        with torch.no_grad():
            outputs = model(inference_session=inference_session, frame_idx=ann_frame_idx)

        first_mask = processor.post_process_masks(
            [outputs.pred_masks],
            original_sizes=[[inference_session.video_height, inference_session.video_width]],
            binarize=False,
        )[0].squeeze()

        h, w = int(first_mask.shape[0]), int(first_mask.shape[1])
        out_mp4 = out_dir / f"{vid.stem}_mask.mp4"
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        writer  = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h), isColor=False)

        first_img = (first_mask.to(torch.float32).cpu().numpy() * 255.0)
        writer.write(np.clip(first_img, 0, 255).astype(np.uint8))

        with torch.no_grad():
            for s in model.propagate_in_video_iterator(inference_session):
                res_mask = processor.post_process_masks(
                    [s.pred_masks],
                    original_sizes=[[inference_session.video_height, inference_session.video_width]],
                    binarize=False,
                )[0].squeeze()
                mask_img = (res_mask.to(torch.float32).cpu().numpy() * 255.0)
                writer.write(np.clip(mask_img, 0, 255).astype(np.uint8))

        writer.release()
        print(f"[OK] {vid.name} -> {out_mp4.name}")
        n_done += 1

    print(f"[SUMMARY] Saved {n_done} mask videos in {out_dir}")
    return n_done

# =============================
# The requested wrapper
# =============================
def run_sam2_pipeline(
    videos_dir: str,
    masked_videos_dir: str | None = None,
    use_clicks: bool = True,
    csv_out_path: str | None = None,
    recursive: bool = False,
    backend: str = "opencv",
) -> Path:
    """
    Input:
      videos_dir: path to folder of input videos.
    Output:
      returns Path to the masked_videos folder (created if needed).

    Optional:
      use_clicks: whether to open the click UI to annotate first frames (default True).
      csv_out_path: where to write/read the CSV (filename,point_idx,x,y). If None,
                    defaults to <videos_dir>/first_frames/click_coords.csv.
      recursive: recurse into subfolders to find videos (default False).
      backend:   video loading backend for transformers (default 'opencv').
    """
    vdir = Path(videos_dir)
    if not vdir.is_dir():
        raise FileNotFoundError(f"Not a folder: {vdir}")

    frames_dir = vdir / "first_frames"
    n_frames = extract_first_frames(vdir, frames_dir, recursive=recursive)
    if n_frames == 0:
        print("[WARN] No first frames extracted. Aborting.")
        return (Path(masked_videos_dir) if masked_videos_dir else vdir / "masked_videos")

    # CSV location (created by this function if clicking; otherwise must exist)
    csv_path = Path(csv_out_path) if csv_out_path else (frames_dir / "click_coords.csv")

    if use_clicks:
        print("[INFO] Launching click UI to record coordinates...")
        run_click_labeler(frames_dir, csv_path)
    else:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV not found: {csv_path}\n"
                "Either enable 'use_clicks=True' to create it interactively, "
                "or provide an existing CSV via 'csv_out_path'."
            )

    masks_dir = Path(masked_videos_dir) if masked_videos_dir else (vdir / "masked_videos")
    masks_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Running SAM2 segmentation to generate mask videos...")
    make_mask_videos(vdir, csv_path, masks_dir, backend=backend)

    return masks_dir

# =============================
# CLI
# =============================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Wrapper: extract frames -> (optional) click points -> make mask videos with SAM2."
    )
    ap.add_argument("--videos", "-v", required=True, type=Path, help="Folder of input videos.")
    ap.add_argument("--output", "-o", type=Path, default=None, help="Folder to save mask videos. Default: <videos>/masked_videos")
    ap.add_argument("--csv", "-c", type=Path, default=None, help="Path to CSV to create/use (filename,point_idx,x,y). Default: <videos>/first_frames/click_coords.csv")
    ap.add_argument("--no-clicks", action="store_true", help="Do NOT open click UI; expect an existing CSV (use --csv).")
    ap.add_argument("--recursive", "-r", action="store_true", help="Recurse into subfolders for videos.")
    ap.add_argument("--backend", default="opencv", choices=["opencv", "pyav", "decord", "torchcodec"], help="Video loading backend.")
    return ap.parse_args()

def main():
    args = parse_args()
    out_dir = run_sam2_pipeline(
        videos_dir=str(args.videos),
        masked_videos_dir=(str(args.output) if args.output else None),
        use_clicks=(not args.no_clicks),
        csv_out_path=(str(args.csv) if args.csv else None),
        recursive=args.recursive,
        backend=args.backend,
    )
    print(f"[DONE] Masked videos folder: {out_dir}")

if __name__ == "__main__":
    main()
