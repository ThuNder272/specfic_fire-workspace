from __future__ import annotations

import contextlib
import importlib
import math
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision

try:
    import tensorrt as trt
except Exception:
    trt = None


@dataclass
class DetectionTimings:
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0


@dataclass
class DetectionPostStats:
    raw_candidates: int = 0
    obj_candidates: int = 0
    class_candidates: int = 0
    kept_candidates: int = 0


def looks_like_legacy_rm4pt_weight(model_path: str) -> bool:
    if not os.path.isfile(model_path) or not model_path.lower().endswith(".pt"):
        return False
    try:
        with zipfile.ZipFile(model_path) as zf:
            if "archive/data.pkl" not in zf.namelist():
                return False
            data = zf.read("archive/data.pkl")
    except Exception:
        return False
    markers = (b"models.yolo", b"landmark", b"yolov5s.yaml")
    return all(marker in data for marker in markers)


def looks_like_legacy_rm4pt_engine(model_path: str) -> bool:
    if trt is None or not os.path.isfile(model_path) or not model_path.lower().endswith(".engine"):
        return False
    try:
        logger = trt.Logger(trt.Logger.ERROR)
        with open(model_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None or engine.num_io_tensors != 2:
            return False
        input_shapes = []
        output_shapes = []
        for idx in range(engine.num_io_tensors):
            name = engine.get_tensor_name(idx)
            shape = tuple(int(dim) for dim in engine.get_tensor_shape(name))
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_shapes.append(shape)
            elif mode == trt.TensorIOMode.OUTPUT:
                output_shapes.append(shape)
        if len(input_shapes) != 1 or len(output_shapes) != 1:
            return False
        input_shape = input_shapes[0]
        output_shape = output_shapes[0]
        return (
            len(input_shape) == 4
            and input_shape[1] == 3
            and len(output_shape) == 3
            and output_shape[-1] == 49
        )
    except Exception:
        return False


@contextlib.contextmanager
def _legacy_module_aliases():
    alias_map = {
        "models": "camera_adaptation.rm4pt_runtime.compat.models",
        "models.common": "camera_adaptation.rm4pt_runtime.compat.models.common",
        "models.yolo": "camera_adaptation.rm4pt_runtime.compat.models.yolo",
    }
    previous = {name: sys.modules.get(name) for name in alias_map}
    try:
        for alias, target in alias_map.items():
            sys.modules[alias] = importlib.import_module(target)
        yield
    finally:
        for alias, module in previous.items():
            if module is None:
                sys.modules.pop(alias, None)
            else:
                sys.modules[alias] = module


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _load_legacy_names_from_weight(model_path: str) -> List[str]:
    if not looks_like_legacy_rm4pt_weight(model_path):
        return []
    with _legacy_module_aliases():
        checkpoint = torch.load(model_path, map_location="cpu")
    model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    raw_names = getattr(model, "names", None)
    return list(raw_names) if isinstance(raw_names, (list, tuple)) else []


def _load_sidecar_legacy_names(model_path: str) -> List[str]:
    sidecar_path = os.path.splitext(model_path)[0] + ".pt"
    if not os.path.exists(sidecar_path):
        return []
    try:
        return _load_legacy_names_from_weight(sidecar_path)
    except Exception:
        return []


def _clip_boxes(boxes: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    boxes[:, 0].clamp_(0, shape[1])
    boxes[:, 1].clamp_(0, shape[0])
    boxes[:, 2].clamp_(0, shape[1])
    boxes[:, 3].clamp_(0, shape[0])
    return boxes


def _clip_landmarks(coords: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    coords[:, [0, 2, 4, 6]].clamp_(0, shape[1])
    coords[:, [1, 3, 5, 7]].clamp_(0, shape[0])
    return coords


def _letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scale_up: bool = True,
    stride: int = 32,
):
    shape = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:
        ratio = min(ratio, 1.0)

    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2.0
    dh /= 2.0

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    if isinstance(ratio, tuple):
        ratio_pad = (ratio, (dw, dh))
    else:
        ratio_pad = ((ratio, ratio), (dw, dh))
    return image, ratio_pad


def _scale_coords(
    img1_shape: Sequence[int],
    coords: torch.Tensor,
    img0_shape: Sequence[int],
    ratio_pad=None,
) -> torch.Tensor:
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )
    else:
        ratio, pad = ratio_pad
        gain = ratio[0] if isinstance(ratio, tuple) else ratio
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    return _clip_boxes(coords, img0_shape)


def _scale_coords_landmarks(
    img1_shape: Sequence[int],
    coords: torch.Tensor,
    img0_shape: Sequence[int],
    ratio_pad=None,
) -> torch.Tensor:
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )
    else:
        ratio, pad = ratio_pad
        gain = ratio[0] if isinstance(ratio, tuple) else ratio
    coords[:, [0, 2, 4, 6]] -= pad[0]
    coords[:, [1, 3, 5, 7]] -= pad[1]
    coords[:, :8] /= gain
    return _clip_landmarks(coords, img0_shape)


def _xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _compute_gain_pad(
    network_shape: Sequence[int],
    frame_shape: Sequence[int],
    ratio_pad=None,
) -> Tuple[float, Tuple[float, float]]:
    if ratio_pad is None:
        gain = min(network_shape[0] / frame_shape[0], network_shape[1] / frame_shape[1])
        pad = (
            (network_shape[1] - frame_shape[1] * gain) / 2,
            (network_shape[0] - frame_shape[0] * gain) / 2,
        )
        return float(gain), (float(pad[0]), float(pad[1]))
    ratio, pad = ratio_pad
    gain = ratio[0] if isinstance(ratio, tuple) else ratio
    return float(gain), (float(pad[0]), float(pad[1]))


def _scale_boxes_np(
    boxes: np.ndarray,
    network_shape: Sequence[int],
    frame_shape: Sequence[int],
    ratio_pad=None,
) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    gain, pad = _compute_gain_pad(network_shape, frame_shape, ratio_pad=ratio_pad)
    scaled = boxes.astype(np.float32, copy=True)
    scaled[:, [0, 2]] -= pad[0]
    scaled[:, [1, 3]] -= pad[1]
    scaled[:, :4] /= gain
    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, frame_shape[1])
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, frame_shape[0])
    return scaled


def _scale_landmarks_np(
    coords: np.ndarray,
    network_shape: Sequence[int],
    frame_shape: Sequence[int],
    ratio_pad=None,
) -> np.ndarray:
    if coords.size == 0:
        return coords
    gain, pad = _compute_gain_pad(network_shape, frame_shape, ratio_pad=ratio_pad)
    scaled = coords.astype(np.float32, copy=True)
    scaled[:, [0, 2, 4, 6]] -= pad[0]
    scaled[:, [1, 3, 5, 7]] -= pad[1]
    scaled[:, :8] /= gain
    scaled[:, [0, 2, 4, 6]] = np.clip(scaled[:, [0, 2, 4, 6]], 0, frame_shape[1])
    scaled[:, [1, 3, 5, 7]] = np.clip(scaled[:, [1, 3, 5, 7]], 0, frame_shape[0])
    return scaled


def _nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thres: float,
    max_det: int,
) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0 and len(keep) < max_det:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[current] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_thres]

    return np.asarray(keep, dtype=np.int64)


def non_max_suppression_face(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[Sequence[int]] = None,
    agnostic: bool = False,
    max_det: int = 300,
    return_stats: bool = False,
):
    nc = prediction.shape[2] - 13
    xc = prediction[..., 4] > conf_thres
    max_wh = 4096
    multi_label = nc > 1
    output = [torch.zeros((0, 17), device=prediction.device)] * prediction.shape[0]
    stats = [DetectionPostStats(raw_candidates=int(prediction.shape[1])) for _ in range(prediction.shape[0])]

    for xi, x in enumerate(prediction):
        stats[xi].raw_candidates = int(x.shape[0])
        x = x[xc[xi]]
        stats[xi].obj_candidates = int(x.shape[0])
        if not x.shape[0]:
            continue
        x[:, 13:] *= x[:, 4:5]
        box = _xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 13:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 13, None], x[i, 5:13], j[:, None].float()), 1)
        else:
            conf, j = x[:, 13:].max(1, keepdim=True)
            x = torch.cat((box, conf, x[:, 5:13], j.float()), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            class_tensor = torch.tensor(classes, device=x.device)
            x = x[(x[:, 13:14] == class_tensor).any(1)]
        n = x.shape[0]
        stats[xi].class_candidates = int(n)
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)]
        c = x[:, 13:14] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        keep = torchvision.ops.nms(boxes, scores, iou_thres)
        if keep.shape[0] > max_det:
            keep = keep[:max_det]
        stats[xi].kept_candidates = int(keep.shape[0])
        output[xi] = x[keep]
    if return_stats:
        return output, stats
    return output


def _postprocess_rm4pt_predictions_cpu(
    prediction: torch.Tensor,
    frame_shape: Sequence[int],
    network_shape: Sequence[int],
    ratio_pad,
    conf_thres: float,
    max_det: int,
    classes: Optional[Sequence[int]],
    names: Sequence[str],
) -> Tuple[List[Dict], DetectionPostStats]:
    pred = prediction[0].detach().float().cpu().numpy()
    nc = pred.shape[1] - 13
    stats = DetectionPostStats(raw_candidates=int(pred.shape[0]))
    pred = pred[pred[:, 4] > conf_thres]
    stats.obj_candidates = int(pred.shape[0])
    if pred.size == 0:
        return [], stats

    class_conf = pred[:, 13:] * pred[:, 4:5]
    row_idx, cls_idx = np.nonzero(class_conf > conf_thres)
    if classes is not None and row_idx.size:
        class_mask = np.isin(cls_idx, np.asarray(classes, dtype=np.int64))
        row_idx = row_idx[class_mask]
        cls_idx = cls_idx[class_mask]
    stats.class_candidates = int(row_idx.size)
    if row_idx.size == 0:
        return [], stats

    boxes = pred[row_idx, :4].astype(np.float32, copy=True)
    boxes_xyxy = np.empty_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    conf = class_conf[row_idx, cls_idx].astype(np.float32, copy=False)
    landmarks = pred[row_idx, 5:13].astype(np.float32, copy=True)
    cls_idx = cls_idx.astype(np.int32, copy=False)

    order = conf.argsort()[::-1]
    boxes_xyxy = boxes_xyxy[order]
    conf = conf[order]
    landmarks = landmarks[order]
    cls_idx = cls_idx[order]

    max_wh = 4096.0
    class_offsets = cls_idx.astype(np.float32)[:, None] * max_wh
    keep = _nms_numpy(boxes_xyxy + class_offsets, conf, iou_thres=0.45, max_det=max_det)
    stats.kept_candidates = int(keep.shape[0])
    if keep.size == 0:
        return [], stats

    boxes_xyxy = _scale_boxes_np(boxes_xyxy[keep], network_shape, frame_shape, ratio_pad=ratio_pad)
    landmarks = _scale_landmarks_np(landmarks[keep], network_shape, frame_shape, ratio_pad=ratio_pad)
    boxes_xyxy = np.rint(boxes_xyxy)
    landmarks = np.rint(landmarks)
    conf = conf[keep]
    cls_idx = cls_idx[keep]

    height, width = frame_shape[:2]
    results: List[Dict] = []
    for box, score, pts_flat, cls_id in zip(boxes_xyxy, conf, landmarks, cls_idx):
        corners = pts_flat.reshape(4, 2)
        if not np.all(np.isfinite(corners)):
            continue
        corners[:, 0] = np.clip(corners[:, 0], 0, max(0, width - 1))
        corners[:, 1] = np.clip(corners[:, 1], 0, max(0, height - 1))
        ordered = _order_quad_points(corners)
        class_name = str(names[cls_id]) if names and 0 <= cls_id < len(names) else str(cls_id)
        x1 = int(np.clip(box[0], 0, max(0, width - 1)))
        y1 = int(np.clip(box[1], 0, max(0, height - 1)))
        x2 = int(np.clip(box[2], 0, max(0, width - 1)))
        y2 = int(np.clip(box[3], 0, max(0, height - 1)))
        if x2 <= x1:
            x2 = min(width - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(height - 1, y1 + 1)
        results.append(
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float(score),
                "class": int(cls_id),
                "class_name": class_name,
                "corners": ordered.astype(np.float32),
                "backend": "rm4pt",
            }
        )
    return results, stats


def _postprocess_rm4pt_predictions(
    prediction: torch.Tensor,
    frame_shape: Sequence[int],
    network_shape: Sequence[int],
    ratio_pad,
    conf_thres: float,
    max_det: int,
    classes: Optional[Sequence[int]],
    names: Sequence[str],
) -> Tuple[List[Dict], DetectionPostStats]:
    detections, stats = non_max_suppression_face(
        prediction,
        conf_thres=conf_thres,
        iou_thres=0.45,
        classes=classes,
        max_det=max_det,
        return_stats=True,
    )
    det = detections[0] if detections else torch.zeros((0, 17), device=prediction.device)
    results: List[Dict] = []
    if not len(det):
        return results, stats[0]

    det = det.clone()
    det[:, :4] = _scale_coords(network_shape, det[:, :4], frame_shape, ratio_pad=ratio_pad).round()
    det[:, 5:13] = _scale_coords_landmarks(network_shape, det[:, 5:13], frame_shape, ratio_pad=ratio_pad).round()

    height, width = frame_shape[:2]
    for row in det:
        xyxy = row[:4].detach().cpu().numpy().astype(np.int32)
        conf = float(row[4].item())
        corners = row[5:13].detach().cpu().numpy().astype(np.float32).reshape(4, 2)
        if not np.all(np.isfinite(corners)):
            continue
        corners[:, 0] = np.clip(corners[:, 0], 0, max(0, width - 1))
        corners[:, 1] = np.clip(corners[:, 1], 0, max(0, height - 1))
        ordered = _order_quad_points(corners)
        cls_idx = int(row[13].item())
        class_name = str(names[cls_idx]) if names and 0 <= cls_idx < len(names) else str(cls_idx)
        x1 = int(np.clip(xyxy[0], 0, max(0, width - 1)))
        y1 = int(np.clip(xyxy[1], 0, max(0, height - 1)))
        x2 = int(np.clip(xyxy[2], 0, max(0, width - 1)))
        y2 = int(np.clip(xyxy[3], 0, max(0, height - 1)))
        if x2 <= x1:
            x2 = min(width - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(height - 1, y1 + 1)
        results.append(
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class": cls_idx,
                "class_name": class_name,
                "corners": ordered.astype(np.float32),
                "backend": "rm4pt",
            }
        )
    return results, stats[0]


def _torch_dtype_from_trt(dtype):
    if trt is None:
        raise RuntimeError("TensorRT 不可用")
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
    }
    if hasattr(trt.DataType, "UINT8"):
        mapping[trt.DataType.UINT8] = torch.uint8
    if dtype not in mapping:
        raise TypeError(f"不支持的TensorRT数据类型: {dtype}")
    return mapping[dtype]


class Legacy4PointDetector:
    def __init__(self, model_path: str, device: Optional[str] = None, image_size: int = 640):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"四点模型文件不存在: {model_path}")
        self.model_path = model_path
        self.image_size = int(image_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.last_timings = DetectionTimings()
        self.last_post_stats = DetectionPostStats()
        with _legacy_module_aliases():
            checkpoint = torch.load(model_path, map_location=self.device)
        model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        if not hasattr(model, "fuse"):
            raise TypeError("legacy 四点模型不包含可用的模型对象")
        model = model.float().fuse().eval()
        self.model = model.to(self.device)
        self.stride = int(getattr(self.model, "stride", torch.tensor([32])).max().item())
        raw_names = getattr(self.model, "names", None)
        self.names = list(raw_names) if isinstance(raw_names, (list, tuple)) else []

    def describe_backend(self) -> str:
        return (
            f"RM4PointLegacy, device: {self.device}, "
            f"pt=True, classes={len(self.names) if self.names else 'NA'}, corners=True"
        )

    def detect(
        self,
        frame_bgr: np.ndarray,
        conf_thres: float,
        max_det: int,
        classes: Optional[Sequence[int]] = None,
    ) -> List[Dict]:
        pre_start = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image, ratio_pad = _letterbox(
            frame_rgb,
            new_shape=(self.image_size, self.image_size),
            auto=False,
            stride=self.stride,
        )
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        tensor = tensor.to(self.device).float().unsqueeze(0) / 255.0
        self.last_timings.preprocess_ms = (time.perf_counter() - pre_start) * 1000.0

        infer_start = time.perf_counter()
        with torch.no_grad():
            prediction = self.model(tensor)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.last_timings.inference_ms = (time.perf_counter() - infer_start) * 1000.0

        post_start = time.perf_counter()
        results, post_stats = _postprocess_rm4pt_predictions(
            prediction=prediction,
            frame_shape=frame_rgb.shape,
            network_shape=tensor.shape[2:],
            ratio_pad=ratio_pad,
            conf_thres=conf_thres,
            max_det=max_det,
            classes=classes,
            names=self.names,
        )
        self.last_post_stats = post_stats
        self.last_timings.postprocess_ms = (time.perf_counter() - post_start) * 1000.0
        return results


class Legacy4PointTensorRTDetector:
    def __init__(self, model_path: str, image_size: int = 640):
        if trt is None:
            raise ImportError("当前环境缺少 tensorrt Python 包")
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT 四点后端需要可用的 CUDA 设备")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"四点 TensorRT 引擎不存在: {model_path}")

        self.model_path = model_path
        self.image_size = int(image_size)
        self.device = torch.device("cuda")
        self.last_timings = DetectionTimings()
        self.last_post_stats = DetectionPostStats()
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"TensorRT 引擎反序列化失败: {model_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT 执行上下文创建失败")

        self.input_name = ""
        self.output_name = ""
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name
        if not self.input_name or not self.output_name:
            raise RuntimeError("TensorRT 引擎缺少输入或输出张量")

        self.input_shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(self.input_name))
        if any(dim < 0 for dim in self.input_shape):
            self.context.set_input_shape(self.input_name, (1, 3, self.image_size, self.image_size))
            self.input_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(self.input_name))
        if self.input_shape != (1, 3, self.image_size, self.image_size):
            raise ValueError(f"不支持的四点引擎输入形状: {self.input_shape}")

        self.output_shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(self.output_name))
        if any(dim < 0 for dim in self.output_shape):
            self.output_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(self.output_name))
        if len(self.output_shape) != 3 or self.output_shape[-1] != 49:
            raise ValueError(f"不支持的四点引擎输出形状: {self.output_shape}")

        self.input_dtype = _torch_dtype_from_trt(self.engine.get_tensor_dtype(self.input_name))
        self.input_tensor = torch.empty(self.input_shape, device=self.device, dtype=self.input_dtype)
        self.output_dtype = _torch_dtype_from_trt(self.engine.get_tensor_dtype(self.output_name))
        self.output_tensor = torch.empty(self.output_shape, device=self.device, dtype=self.output_dtype)
        self._network_buffer = np.empty(
            (self.image_size, self.image_size, int(self.input_shape[1])),
            dtype=np.uint8,
        )
        self._preprocess_plan_cache: Dict[Tuple[int, int], Tuple[int, int, int, int, Tuple[Tuple[float, float], Tuple[float, float]]]] = {}
        try:
            self._input_cpu_tensor = torch.empty(self.input_shape, dtype=torch.uint8, pin_memory=True)
            self._input_staging = self._input_cpu_tensor.numpy()
            self._input_non_blocking = True
        except Exception:
            self._input_staging = np.empty(self.input_shape, dtype=np.uint8)
            self._input_cpu_tensor = torch.from_numpy(self._input_staging)
            self._input_non_blocking = False
        self.context.set_tensor_address(self.input_name, int(self.input_tensor.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(self.output_tensor.data_ptr()))

        self.names = _load_sidecar_legacy_names(model_path)
        self.num_classes = len(self.names) if self.names else int(self.output_shape[-1] - 13)

    def describe_backend(self) -> str:
        return (
            f"RM4PointTensorRT, device: {self.device}, "
            f"engine=True, pt=False, classes={self.num_classes}, corners=True"
        )

    def _get_preprocess_plan(
        self,
        frame_shape: Sequence[int],
    ) -> Tuple[int, int, int, int, Tuple[Tuple[float, float], Tuple[float, float]]]:
        key = (int(frame_shape[0]), int(frame_shape[1]))
        plan = self._preprocess_plan_cache.get(key)
        if plan is not None:
            return plan

        src_h, src_w = key
        ratio = min(self.image_size / src_h, self.image_size / src_w)
        new_w = int(round(src_w * ratio))
        new_h = int(round(src_h * ratio))
        dw = (self.image_size - new_w) / 2.0
        dh = (self.image_size - new_h) / 2.0
        left = int(round(dw - 0.1))
        top = int(round(dh - 0.1))
        ratio_pad = ((ratio, ratio), (dw, dh))
        plan = (new_w, new_h, top, left, ratio_pad)
        self._preprocess_plan_cache[key] = plan
        return plan

    def detect(
        self,
        frame_bgr: np.ndarray,
        conf_thres: float,
        max_det: int,
        classes: Optional[Sequence[int]] = None,
    ) -> List[Dict]:
        pre_start = time.perf_counter()
        new_w, new_h, top, left, ratio_pad = self._get_preprocess_plan(frame_bgr.shape[:2])
        self._network_buffer.fill(114)
        dst = self._network_buffer[top : top + new_h, left : left + new_w]
        if frame_bgr.shape[1] == new_w and frame_bgr.shape[0] == new_h:
            np.copyto(dst, frame_bgr)
        else:
            cv2.resize(frame_bgr, (new_w, new_h), dst=dst, interpolation=cv2.INTER_LINEAR)
        np.copyto(self._input_staging[0], self._network_buffer[:, :, ::-1].transpose(2, 0, 1))
        self.input_tensor.copy_(self._input_cpu_tensor, non_blocking=self._input_non_blocking)
        self.input_tensor.mul_(1.0 / 255.0)
        self.last_timings.preprocess_ms = (time.perf_counter() - pre_start) * 1000.0

        infer_start = time.perf_counter()
        stream = torch.cuda.current_stream(self.device)
        ok = self.context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT 四点推理执行失败")
        stream.synchronize()
        prediction = self.output_tensor
        self.last_timings.inference_ms = (time.perf_counter() - infer_start) * 1000.0

        post_start = time.perf_counter()
        results, post_stats = _postprocess_rm4pt_predictions_cpu(
            prediction=prediction,
            frame_shape=frame_bgr.shape,
            network_shape=self.input_shape[2:],
            ratio_pad=ratio_pad,
            conf_thres=conf_thres,
            max_det=max_det,
            classes=classes,
            names=self.names,
        )
        self.last_post_stats = post_stats
        self.last_timings.postprocess_ms = (time.perf_counter() - post_start) * 1000.0
        return results


__all__ = [
    "DetectionTimings",
    "DetectionPostStats",
    "Legacy4PointDetector",
    "Legacy4PointTensorRTDetector",
    "looks_like_legacy_rm4pt_weight",
    "looks_like_legacy_rm4pt_engine",
    "non_max_suppression_face",
]
