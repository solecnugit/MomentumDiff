import argparse
import gc
import glob
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmdet.datasets import build_dataset
from pyquaternion import Quaternion
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from tools.visualization.bev_render import BEVRender
from tools.visualization.cam_render import CamRender


plot_choices = dict(
    draw_pred=True,
    det=True,
    track=True,
    motion=True,
    map=True,
    planning=True,
)
START = 0
END = 200
INTERVAL = 1


class Visualizer:
    def __init__(self, args, plot_choices):
        self.out_dir = args.out_dir
        self.combine_dir = os.path.join(self.out_dir, "combine")
        os.makedirs(self.combine_dir, exist_ok=True)

        cfg = Config.fromfile(args.config)
        self.dataset = build_dataset(cfg.data.val)
        self.results = self._to_cpu(mmcv.load(args.result_path))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.bev_render = BEVRender(plot_choices, self.out_dir)
        self.cam_render = CamRender(plot_choices, self.out_dir)

    def _to_cpu(self, obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {key: self._to_cpu(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._to_cpu(value) for value in obj]
        if isinstance(obj, tuple):
            return tuple(self._to_cpu(value) for value in obj)
        return obj

    def add_vis(self, index):
        data = self.dataset.get_data_info(index)
        result = self.results[index]
        if isinstance(result, dict) and "img_bbox" in result:
            result = result["img_bbox"]

        cam_pred_path = self.cam_render.render(data, result, index)
        history_plans = self.get_history_plans(index, history_len=2)
        bev_gt_path, bev_pred_path = self.bev_render.render(
            data, result, index, history_plans=history_plans
        )
        self.combine(bev_gt_path, bev_pred_path, cam_pred_path, index)

    def combine(self, bev_gt_path, bev_pred_path, cam_pred_path, index):
        bev_gt = cv2.imread(bev_gt_path)
        bev_pred = cv2.imread(bev_pred_path)
        cam_pred = cv2.imread(cam_pred_path)
        if bev_gt is None or bev_pred is None or cam_pred is None:
            return

        target_height = cam_pred.shape[0]

        def resize_to_height(img, height):
            if img.shape[0] == height:
                return img
            scale = height / img.shape[0]
            width = int(img.shape[1] * scale)
            return cv2.resize(img, (width, height))

        bev_gt = resize_to_height(bev_gt, target_height)
        bev_pred = resize_to_height(bev_pred, target_height)
        merged = cv2.hconcat([cam_pred, bev_pred, bev_gt])
        save_path = os.path.join(self.combine_dir, f"{index:04d}.jpg")
        cv2.imwrite(save_path, merged)

    def image2video(self, fps=12, downsample=4):
        image_paths = sorted(glob.glob(os.path.join(self.combine_dir, "*.jpg")))
        if not image_paths:
            return

        out_path = os.path.join(self.out_dir, "video.mp4")
        writer = None
        try:
            for image_path in tqdm(image_paths):
                image = cv2.imread(image_path)
                if image is None:
                    continue
                height, width = image.shape[:2]
                image = cv2.resize(
                    image,
                    (max(1, width // downsample), max(1, height // downsample)),
                    interpolation=cv2.INTER_AREA,
                )
                size = (image.shape[1], image.shape[0])
                if writer is None:
                    writer = cv2.VideoWriter(
                        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size
                    )
                writer.write(image)
                del image
        finally:
            if writer is not None:
                writer.release()

    def get_history_plans(self, index, history_len=2):
        plans = []
        curr_info = self.dataset.data_infos[index]
        curr_scene = curr_info["scene_token"]
        curr_t = np.array(curr_info["ego2global_translation"])
        curr_r = Quaternion(curr_info["ego2global_rotation"])
        curr_r_inv = curr_r.inverse

        for step in range(1, history_len + 1):
            prev_idx = index - step
            if prev_idx < 0:
                break

            prev_info = self.dataset.data_infos[prev_idx]
            if prev_info["scene_token"] != curr_scene:
                break
            if prev_idx >= len(self.results):
                break

            result = self.results[prev_idx]
            if isinstance(result, dict) and "img_bbox" in result:
                result = result["img_bbox"]
            if "planning" not in result:
                continue

            scores = result["planning_score"]
            trajs = result["planning"]
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()
            if hasattr(trajs, "cpu"):
                trajs = trajs.cpu().numpy()
            if trajs.ndim == 4:
                trajs = trajs.reshape(-1, trajs.shape[-2], 2)
            if scores.ndim > 1:
                scores = scores.flatten()

            best_traj = trajs[np.argmax(scores)]
            best_traj_nusc = np.stack([best_traj[:, 1], -best_traj[:, 0]], axis=-1)
            best_traj_nusc = np.concatenate([np.zeros((1, 2)), best_traj_nusc], axis=0)

            prev_t = np.array(prev_info["ego2global_translation"])
            prev_r = Quaternion(prev_info["ego2global_rotation"])
            traj_3d = np.hstack([best_traj_nusc, np.zeros((len(best_traj_nusc), 1))])
            pts_global = np.dot(traj_3d, prev_r.rotation_matrix.T) + prev_t
            pts_local = np.dot(pts_global - curr_t, curr_r_inv.rotation_matrix.T)
            final_plan = np.stack([-pts_local[:, 1], pts_local[:, 0]], axis=-1)
            plans.append(final_plan[:, :2])

        return plans


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize ground truth and predictions")
    parser.add_argument("config", help="Config file path")
    parser.add_argument("--result-path", required=True, help="Prediction result pkl")
    parser.add_argument("--out-dir", default="vis", help="Output directory")
    parser.add_argument("--start", type=int, default=START)
    parser.add_argument("--end", type=int, default=END)
    parser.add_argument("--interval", type=int, default=INTERVAL)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--video-downsample", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    visualizer = Visualizer(args, plot_choices)

    for idx in tqdm(range(args.start, args.end, args.interval)):
        if idx >= len(visualizer.results):
            break
        visualizer.add_vis(idx)
        plt.close("all")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.skip_video:
        visualizer.image2video(downsample=args.video_downsample)


if __name__ == "__main__":
    main()
