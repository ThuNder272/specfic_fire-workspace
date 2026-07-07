#!/usr/bin/env python
# coding=utf-8

"""
工业相机数据处理器 - 专为工业相机实时流设计
集成YOLOv10目标检测，支持实时工业相机采集
"""

import cv2
import numpy as np
import torch
import os
from typing import List, Optional, Dict, Tuple, Sequence
from ultralytics import YOLO

from camera_adaptation.rm4pt_runtime import (
    Legacy4PointDetector,
    Legacy4PointTensorRTDetector,
    looks_like_legacy_rm4pt_engine,
    looks_like_legacy_rm4pt_weight,
)


class IndustrialCameraProcessor:
    """工业相机数据处理器 - 专为实时工业相机流设计"""

    @staticmethod
    def _normalize_target_color(target_color: Optional[str]) -> Optional[str]:
        if target_color is None:
            return None
        value = str(target_color).strip().lower()
        if not value or value == "any":
            return None
        if value not in {"red", "blue"}:
            raise ValueError("target_color must be 'red' or 'blue'")
        return value

    @staticmethod
    def _class_name_to_color(class_name) -> Optional[str]:
        if class_name is None:
            return None
        label = str(class_name).strip().lower()
        if not label:
            return None
        compact = "".join(ch for ch in label if ch.isalnum())
        if not compact:
            return None
        if compact.startswith("red") or compact == "r":
            return "red"
        if compact.startswith("blue") or compact == "b":
            return "blue"
        if compact[0] == "r":
            return "red"
        if compact[0] == "b":
            return "blue"
        return None

    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        target_size: Tuple[int, int] = (64, 64),
        confidence_threshold: float = 0.3,
        context_padding: int = 10,
        yolo_max_det: int = 1,
        yolo_log_speed: bool = False,
        yolo_verbose: bool = False,
        yolo_imgsz: Optional[Tuple[int, int]] = None,
        detector_backend: str = "auto",
    ):
        """
        初始化工业相机数据处理器
        
        Args:
            yolo_model_path: YOLOv10模型路径
            target_size: 目标尺寸（用于ROI调整）
            confidence_threshold: 检测置信度阈值
            context_padding: 上下文填充像素数
        """
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        self.context_padding = context_padding
        self.yolo_max_det = max(1, int(yolo_max_det))
        self.yolo_log_speed = bool(yolo_log_speed)
        self.yolo_verbose = bool(yolo_verbose)
        self.yolo_imgsz = yolo_imgsz
        self.detector_backend = str(detector_backend or "auto").strip().lower()
        self._yolo_backend_logged = False
        self.use_detection = False
        self._backend_kind = "none"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._last_runtime_stats = {
            "backend": self._backend_kind,
            "preprocess_ms": 0.0,
            "inference_ms": 0.0,
            "postprocess_ms": 0.0,
            "output_count": 0,
            "raw_candidates": 0,
            "obj_candidates": 0,
            "class_candidates": 0,
            "kept_candidates": 0,
        }

        if yolo_model_path and os.path.exists(yolo_model_path):
            self.yolo_model = self._load_detection_model(yolo_model_path)
            if self.yolo_model is not None:
                self.use_detection = True
                print("✓ 工业相机已启用目标检测")
            else:
                self.use_detection = False
                print("⚠️ 工业相机未启用目标检测，将处理整个帧")
        else:
            self.yolo_model = None
            self.use_detection = False
            print("⚠️ 工业相机未提供检测模型，将处理整个帧")

    def _choose_backend(self, yolo_model_path: str) -> str:
        backend = self.detector_backend
        if backend in {"ultralytics", "rm4pt"}:
            return backend
        if looks_like_legacy_rm4pt_weight(yolo_model_path):
            return "rm4pt"
        if looks_like_legacy_rm4pt_engine(yolo_model_path):
            return "rm4pt"
        return "ultralytics"

    def _load_detection_model(self, yolo_model_path: str):
        """加载检测模型。"""
        backend = self._choose_backend(yolo_model_path)
        print(f"正在为工业相机加载检测模型: {yolo_model_path}")
        if backend == "rm4pt":
            if self.yolo_imgsz not in (None, (640, 640)):
                raise ValueError("rm4pt 后端仅支持 --yolo-imgsz 640")
            try:
                if yolo_model_path.lower().endswith(".engine"):
                    detector = Legacy4PointTensorRTDetector(yolo_model_path, image_size=640)
                else:
                    detector = Legacy4PointDetector(yolo_model_path, device=self.device, image_size=640)
                self._backend_kind = "rm4pt"
                print("✓ 成功为工业相机加载RM四点模型")
                return detector
            except Exception as e:
                print(f"❌ 工业相机RM四点模型加载失败: {e}")
                return self._create_mock_yolo_model()

        try:
            yolo_model = YOLO(yolo_model_path, task="detect")
            self._backend_kind = "ultralytics"
            print("✓ 成功为工业相机加载Ultralytics检测模型")
            return yolo_model
        except Exception as e:
            print(f"❌ 工业相机Ultralytics模型加载失败: {e}")
            if looks_like_legacy_rm4pt_weight(yolo_model_path) or looks_like_legacy_rm4pt_engine(yolo_model_path):
                try:
                    if yolo_model_path.lower().endswith(".engine"):
                        detector = Legacy4PointTensorRTDetector(yolo_model_path, image_size=640)
                    else:
                        detector = Legacy4PointDetector(yolo_model_path, device=self.device, image_size=640)
                    self._backend_kind = "rm4pt"
                    print("✓ 已自动回退到RM四点legacy后端")
                    return detector
                except Exception as legacy_exc:
                    print(f"❌ RM四点legacy后端加载失败: {legacy_exc}")
            print("为工业相机创建模拟检测模型...")
            return self._create_mock_yolo_model()
    
    def _create_mock_yolo_model(self):
        """为工业相机创建模拟检测模型"""
        class MockYOLOModel:
            def __init__(self):
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            def __call__(self, img, *args, **kwargs):
                # 返回模拟的检测结果 - 假设图像中心有一个目标
                h, w = img.shape[:2]
                x1, y1 = int(w * 0.3), int(h * 0.3)
                x2, y2 = int(w * 0.7), int(h * 0.7)
                conf = 0.8
                cls = 0
                
                # 创建模拟结果对象，兼容YOLOv10格式
                class MockResult:
                    def __init__(self):
                        self.boxes = MockBoxes()
                
                class MockBoxes:
                    def __init__(self):
                        import torch
                        self.xyxy = torch.tensor([[x1, y1, x2, y2]], device='cpu')
                        self.conf = torch.tensor([conf], device='cpu')
                        self.cls = torch.tensor([cls], device='cpu')
                
                return [MockResult()]

        return MockYOLOModel()

    def _detect_with_ultralytics(
        self,
        frame: np.ndarray,
        classes: Optional[Sequence[int]] = None,
    ) -> List[Dict]:
        predict_kwargs = {
            "conf": self.confidence_threshold,
            "max_det": self.yolo_max_det,
            "verbose": self.yolo_verbose,
        }
        if classes is not None:
            predict_kwargs["classes"] = list(classes)
        if self.yolo_imgsz is not None:
            predict_kwargs["imgsz"] = self.yolo_imgsz
        results = self.yolo_model(frame, **predict_kwargs)
        if not self._yolo_backend_logged:
            self._log_yolo_backend()
        detections = []
        result0 = results[0] if isinstance(results, list) and results else results
        speed = getattr(result0, "speed", {}) if result0 is not None else {}
        if self.yolo_log_speed and hasattr(result0, "speed"):
            try:
                print(
                    f"YOLO speed: pre={speed.get('preprocess', 0):.1f} "
                    f"infer={speed.get('inference', 0):.1f} "
                    f"post={speed.get('postprocess', 0):.1f} ms"
                )
            except Exception:
                pass

        iterable = results if isinstance(results, list) else [results]
        for result in iterable:
            if not hasattr(result, "boxes") or result.boxes is None:
                continue
            boxes = result.boxes
            if not hasattr(boxes, "xyxy") or len(boxes.xyxy) <= 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy() if hasattr(boxes, "conf") else None
            clss = boxes.cls.detach().cpu().numpy() if hasattr(boxes, "cls") else None
            names = getattr(result, "names", None)

            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = box[:4]
                conf = float(confs[i]) if confs is not None else 1.0
                cls = int(clss[i]) if clss is not None else 0
                class_name = (
                    str(names.get(cls, cls))
                    if isinstance(names, dict)
                    else (str(names[cls]) if isinstance(names, (list, tuple)) and 0 <= cls < len(names) else str(cls))
                )
                if conf >= self.confidence_threshold:
                    detections.append(
                        {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": conf,
                            "class": cls,
                            "class_name": class_name,
                            "corners": None,
                            "backend": "ultralytics",
                        }
                    )
        self._last_runtime_stats = {
            "backend": self._backend_kind,
            "preprocess_ms": float(speed.get("preprocess", 0.0)),
            "inference_ms": float(speed.get("inference", 0.0)),
            "postprocess_ms": float(speed.get("postprocess", 0.0)),
            "output_count": len(detections),
            "raw_candidates": len(detections),
            "obj_candidates": len(detections),
            "class_candidates": len(detections),
            "kept_candidates": len(detections),
        }
        return detections

    def _detect_with_rm4pt(
        self,
        frame: np.ndarray,
        classes: Optional[Sequence[int]] = None,
    ) -> List[Dict]:
        detections = self.yolo_model.detect(
            frame,
            conf_thres=self.confidence_threshold,
            max_det=self.yolo_max_det,
            classes=classes,
        )
        if not self._yolo_backend_logged:
            self._log_yolo_backend()
        if self.yolo_log_speed:
            timing = getattr(self.yolo_model, "last_timings", None)
            if timing is not None:
                print(
                    f"YOLO speed: pre={timing.preprocess_ms:.1f} "
                    f"infer={timing.inference_ms:.1f} "
                    f"post={timing.postprocess_ms:.1f} ms"
                )
        timing = getattr(self.yolo_model, "last_timings", None)
        post_stats = getattr(self.yolo_model, "last_post_stats", None)
        self._last_runtime_stats = {
            "backend": self._backend_kind,
            "preprocess_ms": float(getattr(timing, "preprocess_ms", 0.0)),
            "inference_ms": float(getattr(timing, "inference_ms", 0.0)),
            "postprocess_ms": float(getattr(timing, "postprocess_ms", 0.0)),
            "output_count": len(detections),
            "raw_candidates": int(getattr(post_stats, "raw_candidates", 0)),
            "obj_candidates": int(getattr(post_stats, "obj_candidates", 0)),
            "class_candidates": int(getattr(post_stats, "class_candidates", 0)),
            "kept_candidates": int(getattr(post_stats, "kept_candidates", 0)),
        }
        return detections

    def get_color_class_ids(self, target_color: Optional[str]) -> Optional[Tuple[int, ...]]:
        color = self._normalize_target_color(target_color)
        if color is None or self._backend_kind != "rm4pt" or self.yolo_model is None:
            return None

        names = getattr(self.yolo_model, "names", None)
        if isinstance(names, (list, tuple)) and names:
            class_ids = tuple(
                idx
                for idx, class_name in enumerate(names)
                if self._class_name_to_color(class_name) == color
            )
            if class_ids:
                return class_ids

        num_classes = getattr(self.yolo_model, "num_classes", None)
        if num_classes is None and isinstance(names, (list, tuple)):
            num_classes = len(names)
        try:
            num_classes = int(num_classes)
        except Exception:
            return None

        # Fallback to the standard RM4PT 36-class ordering shipped in this repo:
        # [BG, B1..B5, BO, BBs, BBb, RG, R1..R5, RO, RBs, RBb, ...]
        if num_classes >= 18:
            if color == "blue":
                return tuple(range(0, 9))
            return tuple(range(9, 18))
        return None

    def detect_objects(
        self,
        frame: np.ndarray,
        classes: Optional[Sequence[int]] = None,
    ) -> List[Dict]:
        """工业相机目标检测。"""
        if self.yolo_model is None:
            return []

        try:
            if self._backend_kind == "rm4pt":
                return self._detect_with_rm4pt(frame, classes=classes)
            return self._detect_with_ultralytics(frame, classes=classes)
        except Exception as e:
            print(f"工业相机检测失败: {e}")
            return []

    def _log_yolo_backend(self):
        """首次推理后打印YOLO后端信息"""
        if self._backend_kind == "rm4pt":
            try:
                print(f"✓ YOLO backend: {self.yolo_model.describe_backend()}")
            except Exception:
                print("⚠️ YOLO backend信息获取失败")
            self._yolo_backend_logged = True
            return
        try:
            backend = None
            predictor = getattr(self.yolo_model, "predictor", None)
            if predictor is not None and getattr(predictor, "model", None) is not None:
                backend = predictor.model
            if backend is None:
                model_attr = getattr(self.yolo_model, "model", None)
                if model_attr is not None and not isinstance(model_attr, (str, bytes)):
                    backend = model_attr
            if backend is None:
                backend = getattr(self.yolo_model, "backend", None)
                if isinstance(backend, (str, bytes)):
                    backend = None

            backend_name = backend.__class__.__name__ if backend is not None else "Unknown"
            device = getattr(backend, "device", None) if backend is not None else None
            if device is None:
                device = getattr(self.yolo_model, "device", None)
            device_str = str(device) if device is not None else "Unknown"
            engine = getattr(backend, "engine", None) if backend is not None else None
            pt = getattr(backend, "pt", None) if backend is not None else None
            onnx = getattr(backend, "onnx", None) if backend is not None else None
            triton = getattr(backend, "triton", None) if backend is not None else None
            print(
                "✓ YOLO backend: "
                f"{backend_name}, device: {device_str}, "
                f"engine={engine}, pt={pt}, onnx={onnx}, triton={triton}"
            )
        except Exception:
            print("⚠️ YOLO backend信息获取失败")
        self._yolo_backend_logged = True
    
    def extract_roi(self, frame: np.ndarray, detection: Optional[Dict]) -> np.ndarray:
        """从检测框中提取ROI区域（专为工业相机优化）"""
        try:
            if detection is None:
                # 如果没有检测，返回中心区域
                h, w = frame.shape[:2]
                center_size = min(h, w) // 4
                x1 = max(0, (w - center_size) // 2)
                y1 = max(0, (h - center_size) // 2)
                x2 = min(w, x1 + center_size)
                y2 = min(h, y1 + center_size)
            else:
                x1, y1, x2, y2 = detection['bbox']
            
            # 确保坐标在图像范围内
            h, w = frame.shape[:2]
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            
            # 确保坐标顺序正确
            if x1 >= x2 or y1 >= y2:
                # 如果坐标无效，使用中心区域
                center_size = min(h, w) // 4
                x1 = max(0, (w - center_size) // 2)
                y1 = max(0, (h - center_size) // 2)
                x2 = min(w, x1 + center_size)
                y2 = min(h, y1 + center_size)
            
            # 提取ROI
            roi = frame[y1:y2, x1:x2]
            
            # 调整大小到目标尺寸
            if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
                roi = cv2.resize(roi, self.target_size)
            else:
                # 如果ROI为空或无效，创建空白图像
                roi = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            
            return roi
            
        except Exception as e:
            print(f"工业相机提取ROI失败: {e}")
            # 返回默认的空白ROI
            return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
    
    def get_frame_statistics(self) -> dict:
        """获取工业相机帧统计信息"""
        return {
            'use_detection': self.use_detection,
            'confidence_threshold': self.confidence_threshold,
            'target_size': self.target_size,
            'detector_backend': self._backend_kind,
        }

    def get_last_runtime_stats(self) -> dict:
        return dict(self._last_runtime_stats)
