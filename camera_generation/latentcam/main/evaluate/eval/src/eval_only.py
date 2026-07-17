import os
import torch
import argparse
import pandas as pd
import numpy as np
import json
from typing import Dict, List
from torchtyping import TensorType
from src.metrics.modules.caption import CaptionMetrics
from src.metrics.modules.fcd import FrechetCLaTrDistance
from src.metrics.modules.prdc import ManifoldMetrics
from src.metrics.modules.clatr_score import CLaTrScore
from utils.random_utils import set_random_seed

Traj = TensorType["num_samples", "num_cams", 4, 4]  # 摄像机轨迹
Feat = TensorType["num_samples", "num_feats"]  # 特征
Mask = TensorType["num_samples", "num_cams"]  # 掩码（用于文本描述）

class MetricCallback:
    def __init__(self, num_cams: int, num_classes: int, device: str):
        self.num_cams = num_cams

        self.caption_metrics = {
            # "train": CaptionMetrics(num_classes),
            "test": CaptionMetrics(num_classes),
        }
        self.clatr_fd = {
            # "train": FrechetCLaTrDistance(),
            "test": FrechetCLaTrDistance(),
        }
        self.clatr_prdc = {
            # "train": ManifoldMetrics(distance="euclidean"),
            "test": ManifoldMetrics(distance="euclidean"),
        }
        self.clatr_score = {
            # "train": CLaTrScore(),
            "test": CLaTrScore(),
        }

        self.device = device
        self._move_to_device(device)

    def _move_to_device(self, device: str) -> None:
        for stage in ["test"]:
            self.clatr_fd[stage].to(device)
            self.clatr_prdc[stage].to(device)
            self.clatr_score[stage].to(device)

    def update_caption_metrics(self, stage: str, pred: Traj, ref: Traj, mask: Mask):
        print("update_caption_metrics")
        self.caption_metrics[stage].update(pred, ref, mask)

    def compute_caption_metrics(self, stage: str) -> Dict[str, float]:
        precision, recall, fscore = self.caption_metrics[stage].compute()
        self.caption_metrics[stage].reset()
        return {
            "captions/precision": precision,
            "captions/recall": recall,
            "captions/fscore": fscore,
        }

    def update_clatr_metrics(self, stage: str, pred: Feat, ref: Feat, text: Feat):
        print("update_clatr_metrics")
        self.clatr_score[stage].update(pred, text)
        self.clatr_prdc[stage].update(ref, pred)
        self.clatr_fd[stage].update(ref, pred)

    def compute_clatr_metrics(self, stage: str) -> Dict[str, float]:
        clatr_score = self.clatr_score[stage].compute()
        self.clatr_score[stage].reset()

        clatr_p, clatr_r, clatr_d, clatr_c = self.clatr_prdc[stage].compute()
        self.clatr_prdc[stage].reset()

        fcd = self.clatr_fd[stage].compute()
        self.clatr_fd[stage].reset()

        return {
            "clatr/clatr_score": clatr_score,
            "clatr/precision": clatr_p,
            "clatr/recall": clatr_r,
            "clatr/density": clatr_d,
            "clatr/coverage": clatr_c,
            "clatr/fcd": fcd,
        }


def compute_metrics(
    num_cams: int,
    num_classes: int,
    device: str,
    pred_trajectories: TensorType["num_samples", "num_cams", 4, 4],
    ref_trajectories: TensorType["num_samples", "num_cams", 4, 4],
    pred_feats: TensorType["num_samples", "num_feats"],
    ref_feats: TensorType["num_samples", "num_feats"],
    text_feats: TensorType["num_samples", "num_feats"],
    mask: Mask  # 掩码
) -> Dict[str, float]:

    metric_callback = MetricCallback(num_cams=num_cams, num_classes=num_classes, device=device)
    for stage in ["test"]:
        metric_callback.update_caption_metrics(stage, pred_trajectories, ref_trajectories, mask)
    caption_metrics = metric_callback.compute_caption_metrics("test")

    for stage in ["test"]:
        metric_callback.update_clatr_metrics(stage, pred_feats, ref_feats, text_feats)
    clatr_metrics = metric_callback.compute_clatr_metrics("test")

    metrics = {**caption_metrics, **clatr_metrics}
    return metrics, metric_callback

def get_feats(batch):
    pred_feats = batch["m_pred_latents"].to(device)
    ref_feat = batch["m_ref_latents"].to(device)
    text_feats = batch["t_latents"].to(device)
    pred_trajectories = batch["m_pred_matrices"].to(device)
    ref_trajectories = batch["m_ref_matrices"].to(device)
    masks = batch["masks"].to(device)
    
    return pred_feats, ref_feat, text_feats, pred_trajectories, ref_trajectories, masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set pred_path for model predictions")
    
    # Add pred_path as an argument
    parser.add_argument(
        '--pred_path', 
        type=str, 
        required=True,  # Makes the argument required
        help="Path to the predictions file"
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Access the pred_path from the parsed arguments
    pred_path = args.pred_path
    
    set_random_seed(42)

    device = "cuda:0"
    
    batch = np.load(pred_path, allow_pickle=True).item()
    
    pred_feats, ref_feat, text_feats, pred_trajectories, ref_trajectories, masks = get_feats(batch)

    num_cams = masks.shape[1]
    num_classes = 27 * 7 
    
    metrics, metric_callback = compute_metrics(num_cams, num_classes, device, pred_trajectories, ref_trajectories, pred_feats, ref_feat, text_feats, masks)

    metrics_floats = {'val/' + key: round(value.item(), 4) for key, value in metrics.items()}

    metrics_df = pd.DataFrame([metrics_floats])
    base_dir = os.path.dirname(pred_path)
    metrics_df.to_csv(os.path.join(base_dir, f"{pred_path.split('/')[-1][:-4]}.csv"), index=False)
    json_path = os.path.join(base_dir, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics_floats, f)
    print(json_path)

    caption_metrics = metric_callback.caption_metrics['test']
    score_metrics = metric_callback.clatr_score['test']
    rows = []
    for i in range(len(batch['ref_filenames'])):
        rows.append({
            'filename': batch['ref_filenames'][i],
            'captions/precision': caption_metrics.precision_list[i],
            'captions/recall': caption_metrics.recall_list[i],
            'captions/f1': caption_metrics.fscore_list[i],
            'captions/score': score_metrics.score_list[i],
            'pred_segments': caption_metrics.pred_segment_list[i],
            'target_segments': caption_metrics.target_segment_list[i],
        })
    import csv
    output_csv = os.path.join(base_dir, f"{pred_path.split('/')[-1][:-4]}_pcf.csv")
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "captions/precision", "captions/recall", "captions/f1", 'captions/score', 'pred_segments', 'target_segments']
        )
        writer.writeheader()
        writer.writerows(rows)
    print(output_csv)
