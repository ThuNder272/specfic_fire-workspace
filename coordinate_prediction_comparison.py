#!/usr/bin/env python3
"""
坐标预测模型对比脚本

对一个或多个坐标预测模型 checkpoint 在同一数据集上的表现进行统一评估，
输出 MSE / RMSE / MAE / R² / PSNR / 推理时间，并保存 JSON 与图表。
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_MODEL_DIR = "coordinate_prediction_models"
DEFAULT_OUTPUT_DIR = "coordinate_prediction_comparison_results"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def compute_regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, Any]:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.shape != predictions.shape:
        raise ValueError(
            f"预测形状 {predictions.shape} 与目标形状 {targets.shape} 不一致"
        )
    if targets.ndim != 2:
        raise ValueError(f"指标计算要求二维数组，收到形状 {targets.shape}")
    if targets.shape[0] == 0:
        raise ValueError("空数据集无法计算指标")

    errors = predictions - targets
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2

    mse = float(np.mean(squared_errors))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_errors))

    ss_res = float(np.sum(squared_errors))
    centered_targets = targets - np.mean(targets, axis=0, keepdims=True)
    ss_tot = float(np.sum(centered_targets ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    psnr = float("inf") if mse <= 0.0 else float(20.0 * np.log10(1.0 / np.sqrt(mse)))

    coord_labels = ("x", "y", "w", "h")
    per_coordinate: Dict[str, Dict[str, float]] = {}
    for idx in range(targets.shape[1]):
        label = coord_labels[idx] if idx < len(coord_labels) else f"dim_{idx}"
        coord_sq = squared_errors[:, idx]
        coord_abs = abs_errors[:, idx]
        coord_mse = float(np.mean(coord_sq))
        coord_rmse = float(np.sqrt(coord_mse))
        coord_mae = float(np.mean(coord_abs))
        per_coordinate[label] = {
            "mse": coord_mse,
            "rmse": coord_rmse,
            "mae": coord_mae,
        }

    sample_mae = np.mean(abs_errors, axis=1)
    sample_l2 = np.linalg.norm(errors, axis=1)

    return {
        "num_samples": int(targets.shape[0]),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "psnr": psnr,
        "per_coordinate": per_coordinate,
        "sample_mae": sample_mae.tolist(),
        "sample_l2": sample_l2.tolist(),
    }


def build_baseline_predictions(dataset: Any) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if not hasattr(dataset, "sequences") or not hasattr(dataset, "input_sequence_length"):
        raise ValueError("dataset 缺少 sequences 或 input_sequence_length，无法构造 baseline")

    targets: List[np.ndarray] = []
    last_input_predictions: List[np.ndarray] = []
    linear_predictions: List[np.ndarray] = []

    for sequence in dataset.sequences:
        coords = np.asarray(sequence["coordinates"], dtype=np.float32)
        if len(coords) < dataset.input_sequence_length:
            continue
        input_coords = coords[: dataset.input_sequence_length]
        target_index = int(sequence.get("target_index", len(coords) - 1))
        target_index = min(max(target_index, 0), len(coords) - 1)
        target = np.asarray(coords[target_index], dtype=np.float32)

        last_input = np.asarray(input_coords[-1], dtype=np.float32)
        effective_lead = max(0, target_index - (dataset.input_sequence_length - 1))

        linear_prediction = last_input.copy()
        if len(input_coords) >= 2:
            velocity = np.asarray(input_coords[-1] - input_coords[-2], dtype=np.float32)
            linear_prediction = last_input + velocity * effective_lead

        targets.append(target)
        last_input_predictions.append(np.clip(last_input, 0.0, 1.0))
        linear_predictions.append(np.clip(linear_prediction, 0.0, 1.0))

    if not targets:
        raise ValueError("dataset 中没有可评估的序列")

    return np.stack(targets, axis=0), {
        "Baseline-LastInput": np.stack(last_input_predictions, axis=0),
        "Baseline-LinearExtrapolation": np.stack(linear_predictions, axis=0),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _find_latest_recording(recordings_dir: str) -> Optional[str]:
    base_dir = os.path.abspath(recordings_dir)
    patterns = ("*.avi", "*.mp4", "*.mov", "*.mkv")
    candidates: List[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(os.path.join(base_dir, pattern)))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _resolve_video_path(args: argparse.Namespace) -> str:
    if args.video_path:
        video_path = os.path.abspath(args.video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"评估视频不存在: {video_path}")
        return video_path

    if args.use_latest_recording:
        latest = _find_latest_recording(args.recordings_dir)
        if latest is None:
            raise FileNotFoundError(
                f"在录制目录中未找到视频: {os.path.abspath(args.recordings_dir)}"
            )
        print(f"🎞️ 使用最新录制视频: {latest}")
        return latest

    fallback = os.path.abspath("test2.avi")
    if os.path.exists(fallback):
        print(f"⚠️ 未指定评估视频，回退到默认样例: {fallback}")
        return fallback

    latest = _find_latest_recording(args.recordings_dir)
    if latest is not None:
        print(f"⚠️ 未指定评估视频，自动使用最新录制视频: {latest}")
        return latest

    raise FileNotFoundError(
        "未找到评估视频，请通过 --video-path 指定，或使用 --use-latest-recording"
    )


def _resolve_model_specs(
    model_paths: Optional[Sequence[str]],
    model_names: Optional[Sequence[str]],
) -> List[Tuple[str, str]]:
    resolved_paths: List[str]
    if model_paths:
        resolved_paths = [os.path.abspath(path) for path in model_paths]
    else:
        default_dir = os.path.abspath(DEFAULT_MODEL_DIR)
        resolved_paths = sorted(glob.glob(os.path.join(default_dir, "*.pth")))
        if not resolved_paths:
            raise FileNotFoundError(
                "未指定 --model-path，且 coordinate_prediction_models/ 中没有可用的 .pth 模型"
            )

    for path in resolved_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")

    if model_names and len(model_names) != len(resolved_paths):
        raise ValueError("--model-name 的数量必须与 --model-path 一致")

    specs: List[Tuple[str, str]] = []
    for idx, path in enumerate(resolved_paths):
        name = (
            model_names[idx]
            if model_names and idx < len(model_names)
            else Path(path).stem
        )
        specs.append((name, path))
    return specs


class CoordinatePredictionComparison:
    def __init__(self, device: Optional[str] = None):
        self.requested_device = device
        self.device = None
        self.loaded_models: List[Dict[str, Any]] = []

    @staticmethod
    def _import_runtime_modules():
        import torch
        from torch.utils.data import DataLoader
        from coordinate_prediction_model import CoordinateDataset, CoordinatePredictionModel

        return torch, DataLoader, CoordinateDataset, CoordinatePredictionModel

    def _resolve_device(self) -> str:
        torch, _, _, _ = self._import_runtime_modules()
        if self.requested_device:
            return str(self.requested_device)
        return "cuda" if torch.cuda.is_available() else "cpu"

    def create_coordinate_dataset(
        self,
        *,
        video_path: str,
        yolo_model_path: str,
        sequence_length: int,
        num_sequences: int,
        confidence_threshold: float,
        context_padding: int,
        input_sequence_length: int,
        default_lead_frames: int,
        min_lead_frames: int,
        max_lead_frames: int,
        bullet_speed_mps: float,
        system_latency_s: float,
        default_fps: float,
        pnp_profile: str,
        max_pnp_error: float,
        auto_target_type: bool,
        target_type: str,
        scale_intrinsics: bool,
    ):
        _, _, CoordinateDataset, _ = self._import_runtime_modules()
        return CoordinateDataset(
            video_path=video_path,
            yolo_model_path=yolo_model_path,
            sequence_length=sequence_length,
            num_sequences=num_sequences,
            confidence_threshold=confidence_threshold,
            context_padding=context_padding,
            input_sequence_length=input_sequence_length,
            default_lead_frames=default_lead_frames,
            min_lead_frames=min_lead_frames,
            max_lead_frames=max_lead_frames,
            bullet_speed_mps=bullet_speed_mps,
            system_latency_s=system_latency_s,
            default_fps=default_fps,
            pnp_profile=pnp_profile,
            max_pnp_error=max_pnp_error,
            auto_target_type=auto_target_type,
            target_type=target_type,
            scale_intrinsics=scale_intrinsics,
        )

    def load_models(
        self,
        model_specs: Sequence[Tuple[str, str]],
    ) -> List[Dict[str, Any]]:
        torch, _, _, CoordinatePredictionModel = self._import_runtime_modules()
        self.device = self._resolve_device()
        self.loaded_models = []

        for model_name, model_path in model_specs:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                model_config = dict(checkpoint.get("model_config", {}))
            else:
                state_dict = checkpoint
                model_config = {}

            coordinate_dim = int(model_config.get("coordinate_dim", 4))
            if coordinate_dim != 4:
                raise ValueError(
                    f"当前对比脚本仅支持 4 维 bbox 输出模型，{model_name} 的 coordinate_dim={coordinate_dim}"
                )

            model = CoordinatePredictionModel(
                input_size=int(model_config.get("input_size", 64 * 64)),
                hidden_size=int(model_config.get("hidden_size", 128)),
                num_lstm_layers=int(model_config.get("num_lstm_layers", 2)),
                dropout=float(model_config.get("dropout", 0.1)),
                coordinate_dim=coordinate_dim,
            )
            load_msg = model.load_state_dict(state_dict, strict=False)
            model = model.to(self.device)
            model.eval()

            self.loaded_models.append(
                {
                    "name": model_name,
                    "path": model_path,
                    "model": model,
                    "config": model_config,
                    "missing_keys": list(getattr(load_msg, "missing_keys", [])),
                    "unexpected_keys": list(getattr(load_msg, "unexpected_keys", [])),
                }
            )
        return self.loaded_models

    def evaluate_model(
        self,
        loaded_model: Dict[str, Any],
        dataset: Any,
        batch_size: int,
    ) -> Dict[str, Any]:
        torch, DataLoader, _, _ = self._import_runtime_modules()
        model = loaded_model["model"]
        eval_batch = max(1, min(int(batch_size), len(dataset)))
        loader = DataLoader(dataset, batch_size=eval_batch, shuffle=False, num_workers=0)

        predictions: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        total_inference_s = 0.0
        total_samples = 0
        use_cuda_timing = str(self.device).startswith("cuda") and torch.cuda.is_available()

        with torch.no_grad():
            for input_frames, input_coords, target_coords in loader:
                input_frames = input_frames.to(self.device)
                input_coords = input_coords.to(self.device)
                if use_cuda_timing:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                outputs = model(input_frames, input_coords=input_coords)
                if use_cuda_timing:
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                predicted_coordinates = outputs["predicted_coordinates"].detach().cpu().numpy()
                batch_targets = target_coords.detach().cpu().numpy()

                predictions.append(predicted_coordinates)
                targets.append(batch_targets)
                total_inference_s += elapsed
                total_samples += predicted_coordinates.shape[0]

        all_predictions = np.concatenate(predictions, axis=0)
        all_targets = np.concatenate(targets, axis=0)
        metrics = compute_regression_metrics(all_targets, all_predictions)
        metrics["inference_ms"] = float(total_inference_s * 1000.0 / max(1, total_samples))

        return {
            "name": loaded_model["name"],
            "kind": "model",
            "path": loaded_model["path"],
            "metrics": metrics,
            "config": loaded_model.get("config", {}),
            "missing_keys": loaded_model.get("missing_keys", []),
            "unexpected_keys": loaded_model.get("unexpected_keys", []),
        }

    def evaluate_baselines(self, dataset: Any) -> List[Dict[str, Any]]:
        targets, baseline_predictions = build_baseline_predictions(dataset)
        results: List[Dict[str, Any]] = []
        for baseline_name, predictions in baseline_predictions.items():
            metrics = compute_regression_metrics(targets, predictions)
            metrics["inference_ms"] = 0.0
            results.append(
                {
                    "name": baseline_name,
                    "kind": "baseline",
                    "path": None,
                    "metrics": metrics,
                    "config": {},
                    "missing_keys": [],
                    "unexpected_keys": [],
                }
            )
        return results

    def evaluate_coordinate_prediction(
        self,
        dataset: Any,
        *,
        batch_size: int,
        include_baselines: bool = True,
    ) -> List[Dict[str, Any]]:
        if len(dataset) == 0:
            raise ValueError("数据集为空，无法进行模型对比")

        results: List[Dict[str, Any]] = []
        if include_baselines:
            results.extend(self.evaluate_baselines(dataset))
        for loaded_model in self.loaded_models:
            results.append(self.evaluate_model(loaded_model, dataset, batch_size=batch_size))
        results.sort(key=lambda item: item["metrics"]["mse"])
        return results


def save_results(
    output_dir: str,
    *,
    dataset_info: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    detailed_payload = {
        "dataset": dataset_info,
        "results": [json_safe(result) for result in results],
    }
    summary_payload = [
        {
            "rank": idx + 1,
            "name": result["name"],
            "kind": result["kind"],
            "path": result.get("path"),
            "mse": json_safe(result["metrics"].get("mse")),
            "rmse": json_safe(result["metrics"].get("rmse")),
            "mae": json_safe(result["metrics"].get("mae")),
            "r2": json_safe(result["metrics"].get("r2")),
            "psnr": json_safe(result["metrics"].get("psnr")),
            "inference_ms": json_safe(result["metrics"].get("inference_ms")),
        }
        for idx, result in enumerate(results)
    ]

    detailed_path = os.path.join(output_dir, "detailed_results.json")
    summary_path = os.path.join(output_dir, "summary_results.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_payload, f, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    return detailed_path, summary_path


def plot_results(output_dir: str, results: Sequence[Dict[str, Any]]) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"⚠️ 跳过图表生成: matplotlib 不可用 ({exc})")
        return []

    os.makedirs(output_dir, exist_ok=True)
    names = [result["name"] for result in results]
    mse_values = [result["metrics"]["mse"] for result in results]
    rmse_values = [result["metrics"]["rmse"] for result in results]
    mae_values = [result["metrics"]["mae"] for result in results]
    r2_values = [0.0 if result["metrics"]["r2"] is None else result["metrics"]["r2"] for result in results]
    infer_values = [result["metrics"]["inference_ms"] for result in results]

    metric_plot_path = os.path.join(output_dir, "coordinate_prediction_metrics_comparison.png")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metric_specs = [
        ("MSE", mse_values),
        ("RMSE", rmse_values),
        ("MAE", mae_values),
        ("R²", r2_values),
        ("Inference ms", infer_values),
    ]
    for axis, (title, values) in zip(axes.flat, metric_specs):
        axis.bar(names, values)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)
        axis.grid(True, alpha=0.3)
    axes[1, 2].axis("off")
    plt.tight_layout()
    plt.savefig(metric_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    distribution_plot_path = os.path.join(
        output_dir, "coordinate_prediction_distribution_comparison.png"
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].boxplot(
        [result["metrics"]["sample_mae"] for result in results],
        labels=names,
        showfliers=False,
    )
    axes[0].set_title("Sample MAE Distribution")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot(
        [result["metrics"]["sample_l2"] for result in results],
        labels=names,
        showfliers=False,
    )
    axes[1].set_title("Sample L2 Error Distribution")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(distribution_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return [metric_plot_path, distribution_plot_path]


def print_summary(results: Sequence[Dict[str, Any]]) -> None:
    print("\n📊 模型对比结果")
    print("=" * 80)
    for idx, result in enumerate(results, start=1):
        metrics = result["metrics"]
        print(
            f"{idx:>2}. {result['name']:<32} "
            f"MSE={metrics['mse']:.6f}  "
            f"RMSE={metrics['rmse']:.6f}  "
            f"MAE={metrics['mae']:.6f}  "
            f"R²={metrics['r2']:.4f}  "
            f"PSNR={metrics['psnr']:.2f}  "
            f"Infer={metrics['inference_ms']:.3f}ms"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比一个或多个坐标预测模型在同一数据集上的表现")
    parser.add_argument("--video-path", default=None, help="评估视频路径")
    parser.add_argument(
        "--use-latest-recording",
        action="store_true",
        default=False,
        help="自动使用 recordings/ 目录下最新的录制视频",
    )
    parser.add_argument("--recordings-dir", default="recordings", help="录制视频目录")
    parser.add_argument(
        "--yolo-model",
        default="best.pt",
        help="检测模型路径，支持普通YOLO权重以及RM四点 .pt/.engine",
    )
    parser.add_argument(
        "--model-path",
        action="append",
        default=None,
        help="要评估的模型 checkpoint 路径，可重复传入多个",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=None,
        help="模型显示名称，数量需与 --model-path 一致",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="评估结果输出目录")
    parser.add_argument("--device", default=None, help="运行设备，默认自动选择 cuda/cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="评估批次大小")
    parser.add_argument("--input-seq", type=int, default=5, help="输入序列长度")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=0,
        help="总序列长度，0 表示自动使用 input-seq + max-lead-frames",
    )
    parser.add_argument("--min-lead-frames", type=int, default=1, help="最小预测步长")
    parser.add_argument("--max-lead-frames", type=int, default=15, help="最大预测步长")
    parser.add_argument("--default-lead-frames", type=int, default=15, help="PnP失败时默认预测步长")
    parser.add_argument("--num-sequences", type=int, default=500, help="评估序列数量")
    parser.add_argument("--confidence-threshold", type=float, default=0.3, help="检测置信度阈值")
    parser.add_argument("--context-padding", type=int, default=15, help="ROI上下文扩展像素")
    parser.add_argument("--bullet-speed", type=float, default=12.0, help="评估时使用的弹速(m/s)")
    parser.add_argument("--system-latency-ms", type=float, default=0.0, help="评估时使用的系统时延(ms)")
    parser.add_argument("--default-fps", type=float, default=60.0, help="视频FPS缺失时使用的默认值")
    parser.add_argument("--pnp-profile", default="mer_139_210u3c", help="PnP相机内参配置名")
    parser.add_argument("--max-pnp-error", type=float, default=5.0, help="PnP误差阈值")
    parser.add_argument("--target-type", default="armor_small", help="默认目标类型")
    parser.add_argument(
        "--auto-target-type",
        action="store_true",
        default=True,
        help="自动根据检测类别选择目标类型(默认开启)",
    )
    parser.add_argument(
        "--no-auto-target-type",
        action="store_false",
        dest="auto_target_type",
        help="关闭自动目标类型",
    )
    parser.add_argument(
        "--scale-intrinsics",
        action="store_true",
        default=True,
        help="按视频分辨率缩放内参(默认开启)",
    )
    parser.add_argument(
        "--no-scale-intrinsics",
        action="store_false",
        dest="scale_intrinsics",
        help="关闭内参缩放",
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        default=True,
        help="包含 LastInput / LinearExtrapolation baseline (默认开启)",
    )
    parser.add_argument(
        "--no-baselines",
        action="store_false",
        dest="include_baselines",
        help="不包含 baseline",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(int(args.seed))

    video_path = _resolve_video_path(args)
    yolo_model_path = os.path.abspath(args.yolo_model)
    if not os.path.exists(yolo_model_path):
        raise FileNotFoundError(f"检测模型不存在: {yolo_model_path}")

    input_sequence_length = max(1, int(args.input_seq))
    min_lead_frames = max(0, int(args.min_lead_frames))
    max_lead_frames = max(min_lead_frames, int(args.max_lead_frames))
    default_lead_frames = int(args.default_lead_frames)
    if args.sequence_length and int(args.sequence_length) > 0:
        sequence_length = max(int(args.sequence_length), input_sequence_length + max_lead_frames)
    else:
        sequence_length = input_sequence_length + max_lead_frames

    comparison = CoordinatePredictionComparison(device=args.device)
    model_specs = _resolve_model_specs(args.model_path, args.model_name)
    comparison.load_models(model_specs)

    dataset = comparison.create_coordinate_dataset(
        video_path=video_path,
        yolo_model_path=yolo_model_path,
        sequence_length=sequence_length,
        num_sequences=max(1, int(args.num_sequences)),
        confidence_threshold=float(args.confidence_threshold),
        context_padding=max(0, int(args.context_padding)),
        input_sequence_length=input_sequence_length,
        default_lead_frames=default_lead_frames,
        min_lead_frames=min_lead_frames,
        max_lead_frames=max_lead_frames,
        bullet_speed_mps=max(1e-3, float(args.bullet_speed)),
        system_latency_s=max(0.0, float(args.system_latency_ms) / 1000.0),
        default_fps=max(1.0, float(args.default_fps)),
        pnp_profile=args.pnp_profile,
        max_pnp_error=float(args.max_pnp_error),
        auto_target_type=bool(args.auto_target_type),
        target_type=args.target_type,
        scale_intrinsics=bool(args.scale_intrinsics),
    )

    results = comparison.evaluate_coordinate_prediction(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        include_baselines=bool(args.include_baselines),
    )

    dataset_info = {
        "video_path": video_path,
        "yolo_model_path": yolo_model_path,
        "device": comparison.device,
        "num_sequences": len(dataset),
        "requested_num_sequences": int(args.num_sequences),
        "input_sequence_length": input_sequence_length,
        "sequence_length": sequence_length,
        "min_lead_frames": min_lead_frames,
        "max_lead_frames": max_lead_frames,
        "default_lead_frames": default_lead_frames,
        "bullet_speed_mps": float(args.bullet_speed),
        "system_latency_ms": float(args.system_latency_ms),
        "default_fps": float(args.default_fps),
        "pnp_profile": args.pnp_profile,
        "max_pnp_error": float(args.max_pnp_error),
        "auto_target_type": bool(args.auto_target_type),
        "target_type": args.target_type,
        "scale_intrinsics": bool(args.scale_intrinsics),
        "seed": int(args.seed),
        "backend_kind": getattr(dataset, "_backend_kind", None),
    }

    detailed_path, summary_path = save_results(
        os.path.abspath(args.output_dir),
        dataset_info=dataset_info,
        results=results,
    )
    plot_paths = plot_results(os.path.abspath(args.output_dir), results)
    print_summary(results)
    print(f"\n📁 详细结果: {detailed_path}")
    print(f"📁 摘要结果: {summary_path}")
    for plot_path in plot_paths:
        print(f"🖼️ 图表: {plot_path}")


if __name__ == "__main__":
    main()
