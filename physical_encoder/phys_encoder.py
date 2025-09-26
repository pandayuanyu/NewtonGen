import os

import re
import math
import cv2
import torch
import numpy as np
from transformers import Sam2VideoModel, Sam2VideoProcessor, infer_device
from transformers.video_utils import load_video

# --------------------------- PhysicalEncoder ---------------------------
class PhysicalEncoder:
    def __init__(self,
                 model_name="facebook/sam2.1-hiera-tiny",
                 device=None,
                 save_video=False,
                 mask_outdir="mask_videos",
                 smooth_window=3,
                 meter_per_pixel=1/12,
                 delta_t=1/100):
        self.device = device if device is not None else infer_device()
        self.save_video = save_video
        self.mask_outdir = mask_outdir
        os.makedirs(self.mask_outdir, exist_ok=True)

        self.smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        self.meter_per_pixel = meter_per_pixel
        self.delta_t = delta_t

        self.model = Sam2VideoModel.from_pretrained(model_name).to(self.device, dtype=torch.bfloat16)
        self.processor = Sam2VideoProcessor.from_pretrained(model_name)

    def segment_video(self, video_path, click_point=None):
        video_frames, _ = load_video(video_path)
        inference_session = self.processor.init_video_session(
            video=video_frames,
            inference_device=self.device,
            dtype=torch.bfloat16,
        )

        if click_point is not None:
            points = [[[[click_point[0], click_point[1]]]]]
            labels = [[[1]]]
            self.processor.add_inputs_to_inference_session(
                inference_session=inference_session,
                frame_idx=0,
                obj_ids=1,
                input_points=points,
                input_labels=labels,
            )
            outputs = self.model(inference_session=inference_session, frame_idx=0)
            _ = self.processor.post_process_masks(
                [outputs.pred_masks],
                original_sizes=[[inference_session.video_height, inference_session.video_width]],
                binarize=False
            )[0]

        video_segments = {}
        mask_frames = []
        for sam2_video_output in self.model.propagate_in_video_iterator(inference_session):
            mask_tensor = self.processor.post_process_masks(
                [sam2_video_output.pred_masks],
                original_sizes=[[inference_session.video_height, inference_session.video_width]],
                binarize=False
            )[0]
            video_segments[sam2_video_output.frame_idx] = mask_tensor

            mask = (mask_tensor.squeeze().to(torch.float32).cpu().numpy() > 0.5).astype(np.uint8) * 255
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_frames.append(mask_bgr)

        if self.save_video and len(mask_frames) > 0:
            H, W, _ = mask_frames[0].shape
            save_name = os.path.splitext(os.path.basename(video_path))[0] + "_mask.mp4"
            save_path = os.path.join(self.mask_outdir, save_name)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(save_path, fourcc, 30, (W, H))
            for f in mask_frames:
                out.write(f)
            out.release()
            print(f"[INFO] Saved mask video -> {save_path}")

        return video_segments

    def smooth_centers(self, centers):
        N = len(centers)
        half_w = self.smooth_window // 2
        centers_smooth = np.copy(centers)
        for i in range(N):
            start = max(0, i - half_w)
            end = min(N, i + half_w + 1)
            centers_smooth[i] = np.mean(centers[start:end], axis=0)
        return centers_smooth

    def extract_physical_features(self, video_masks):
        centers, areas, long_axes, short_axes, thetas = [], [], [], [], []
        H, W = video_masks[sorted(video_masks.keys())[0]].squeeze().shape

        for idx in sorted(video_masks.keys()):
            mask_tensor = video_masks[idx].squeeze()
            mask = (mask_tensor.to(torch.float32).cpu().numpy() > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                centers.append((0, 0))
                areas.append(0)
                long_axes.append(0)
                short_axes.append(0)
                thetas.append(0)
                continue

            cnt = max(contours, key=cv2.contourArea)
            M = cv2.moments(cnt)
            cx = M["m10"] / M["m00"] if M["m00"] > 0 else 0
            cy = M["m01"] / M["m00"] if M["m00"] > 0 else 0
            cy = H - cy
            centers.append((cx, cy))
            area = cv2.contourArea(cnt)
            areas.append(area)

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.float32)
            edges = [box[1] - box[0], box[2] - box[1], box[3] - box[2], box[0] - box[3]]
            lengths = [np.linalg.norm(e) for e in edges]

            max_idx = np.argmax(lengths)
            min_idx = np.argmin(lengths)
            long_edge = edges[max_idx]

            angle = math.atan2(-long_edge[1], long_edge[0])
            if angle < 0:
                angle += math.pi

            long_axes.append(lengths[max_idx])
            short_axes.append(lengths[min_idx])
            thetas.append(angle)

        centers = np.array(centers)
        areas = np.array(areas)
        long_axes = np.array(long_axes)
        short_axes = np.array(short_axes)
        thetas = np.array(thetas)

        centers_smooth = self.smooth_centers(centers)

        velocities_px = np.zeros_like(centers_smooth)
        velocities_px[1:-1] = (centers_smooth[2:] - centers_smooth[:-2]) / 2
        centers_m = centers_smooth * self.meter_per_pixel
        velocities_m = velocities_px * self.meter_per_pixel / self.delta_t

        theta_rad = thetas
        omega = np.zeros_like(theta_rad)
        omega[1:-1] = (theta_rad[2:] - theta_rad[:-2]) / (2 * self.delta_t)

        s = short_axes * self.meter_per_pixel
        l = long_axes * self.meter_per_pixel
        a = areas * (self.meter_per_pixel ** 2)

        centers_mid = centers_m[3:-3]
        velocities_mid = velocities_m[3:-3]
        theta_mid = theta_rad[3:-3]
        omega_mid = omega[3:-3]
        s_mid = s[3:-3]
        l_mid = l[3:-3]
        a_mid = a[3:-3]

        feats = np.concatenate(
            [centers_mid,
             velocities_mid,
             theta_mid[:, None],
             omega_mid[:, None],
             s_mid[:, None],
             l_mid[:, None],
             a_mid[:, None]],
            axis=1
        )

        return torch.tensor(feats, dtype=torch.float32)

    def __call__(self, video_path, click_point=None):
        video_masks = self.segment_video(video_path, click_point=click_point)
        feats = self.extract_physical_features(video_masks)
        return feats

if __name__ == "__main__":
    video_dir = ""
    out_path = ""

    encoder = PhysicalEncoder(save_video=True, mask_outdir="masks/mask_videos_3dmove",
                              smooth_window=5, meter_per_pixel=0.00625, delta_t=1/100)

    all_feats = []

    for fname in sorted(os.listdir(video_dir)):
        if fname.lower().endswith(".mp4"):
            video_path = os.path.join(video_dir, fname)
            m = re.search(r"px_(\d+)_py_(\d+)", fname)
            if m:
                px = int(m.group(1))
                py = int(m.group(2))
                click_point = [px, py]
            else:
                click_point = None
            print(f"Processing {video_path}, click_point={click_point}")
            feats = encoder(video_path, click_point=click_point)

            print("single feat", feats.shape)

            num_show = min(15, feats.shape[0])
            print(f"[INFO] Extracted physical labels :")
            print("x, y, vx, vy, theta(rad), omega(rad/s), s(short), l(long), a(area)")
            print(feats[:num_show])

            all_feats.append(feats)

    all_feats_tensor = torch.stack(all_feats, dim=0)
    torch.save(all_feats_tensor, out_path)
    print(f"Saved batch features: shape={all_feats_tensor.shape}, path={out_path}")
