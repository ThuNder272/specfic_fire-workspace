#!/usr/bin/env python3
"""
坐标预测模型 - 基于LSTM和Kalman Filter的融合模型
预测目标物体的坐标位置 (x, y, w, h)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List, Sequence
import numpy as np
import os

from camera_adaptation.pnp_solver import (
    CameraIntrinsics,
    TargetGeometry,
    choose_target_type_by_class,
    compute_target_distance,
    get_camera_intrinsics,
    scale_intrinsics_to_frame,
    solve_pose,
)
from camera_adaptation.rm4pt_runtime import (
    Legacy4PointDetector,
    Legacy4PointTensorRTDetector,
    looks_like_legacy_rm4pt_engine,
    looks_like_legacy_rm4pt_weight,
)

from lstm_module import LSTMFeatureExtractor
from kalman_filter import AdaptiveKalmanFilter
from fusion_module import FusionModule

class CoordinatePredictionModel(nn.Module):
    """坐标预测模型 - 预测目标物体的边界框坐标"""
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_lstm_layers: int = 2,
         
                 dropout: float = 0.1,
                 coordinate_dim: int = 4):  # x, y, w, h
        super(CoordinatePredictionModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers

        self.dropout = dropout
        self.coordinate_dim = coordinate_dim
        self.fusion_method = "hierarchical_coord_gated"
        
        # LSTM 特征提取器
        self.lstm_extractor = LSTMFeatureExtractor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout
        )
        
        # Kalman Filter 模块
        self.kalman_filter = AdaptiveKalmanFilter(hidden_size=hidden_size)
        
        # 融合模块
        self.fusion_module = FusionModule(hidden_size=hidden_size)

        # 坐标时序分支：编码输入坐标序列 (x, y, w, h)
        self.coord_lstm = nn.LSTM(
            input_size=coordinate_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.coord_fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self.coord_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 坐标预测头
        self.coordinate_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, coordinate_dim),
            nn.Sigmoid()  # 输出范围0-1，需要反归一化到实际坐标
        )
        
        # 置信度预测头
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor = None,
        input_coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入序列，形状为 (batch_size, sequence_length, input_size)
            h_prev: 前一时刻的融合隐藏状态，形状为 (batch_size, hidden_size)
            input_coords: 输入坐标序列，形状为 (batch_size, sequence_length, 4)
            
        Returns:
            包含各种输出的字典
        """
        batch_size = x.size(0)
        
        # 初始化前一时刻隐藏状态
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # 1. LSTM 特征提取
        h_lstm, _ = self.lstm_extractor(x)
        
        # 2. Kalman Filter 处理
        h_kf = self.kalman_filter(h_prev, h_lstm)
        
        # 3. 融合 LSTM 和 Kalman Filter 的输出
        h_fused = self.fusion_module(h_lstm, h_kf)
        
        # 4. 融合坐标时序特征（若提供 input_coords）
        h_final = h_fused
        coord_gate = None
        coord_used = False
        if input_coords is not None:
            coord_seq = input_coords.float()
            if coord_seq.dim() == 2:
                coord_seq = coord_seq.unsqueeze(1)
            if (
                coord_seq.dim() == 3
                and coord_seq.size(0) == batch_size
                and coord_seq.size(-1) == self.coordinate_dim
            ):
                _, (h_coord_last, _) = self.coord_lstm(coord_seq)
                h_coord = h_coord_last[-1]
                coord_gate = self.coord_fusion_gate(torch.cat([h_fused, h_coord], dim=-1))
                h_final = coord_gate * h_fused + (1.0 - coord_gate) * h_coord
                h_final = self.coord_projection(h_final)
                coord_used = True

        # 5. 预测坐标和置信度
        predicted_coordinates = self.coordinate_head(h_final)
        predicted_confidence = self.confidence_head(h_final)
        
        # 返回所有中间结果和最终输出
        outputs = {
            'h_lstm': h_lstm,                    # LSTM 隐藏状态
            'h_kf': h_kf,                        # Kalman Filter 隐藏状态
            'h_fused': h_fused,                  # 融合后的隐藏状态
            'h_final': h_final,                  # 融合坐标后的隐藏状态
            'predicted_coordinates': predicted_coordinates,   # 预测坐标 (x,y,w,h)
            'predicted_confidence': predicted_confidence,     # 预测置信度
            'coord_gate': coord_gate,            # 坐标融合门控 (可选)
            'coord_used': coord_used,            # 是否使用了坐标分支
            'fusion_weights': self._get_fusion_weights()      # 融合权重
        }
        
        return outputs
    
    def _get_fusion_weights(self) -> Dict[str, torch.Tensor]:
        """获取融合权重（如果适用）"""
        if hasattr(self.fusion_module, 'get_weights'):
            return self.fusion_module.get_weights()
        return {}
    
    def predict_coordinates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测坐标的便捷方法
        
        Args:
            x: 输入序列
            
        Returns:
            coordinates: 预测坐标 (batch_size, 4)
            confidence: 预测置信度 (batch_size, 1)
        """
        outputs = self.forward(x)
        return outputs['predicted_coordinates'], outputs['predicted_confidence']
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_name': 'Coordinate-Prediction-Model',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_lstm_layers': self.num_lstm_layers,
            'coordinate_dim': self.coordinate_dim,
            'fusion_method': self.fusion_method,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
        
        return info

class CoordinateDataset:
    """坐标预测数据集"""
    
    def __init__(
        self,
        video_path: str,
        yolo_model_path: str,
        sequence_length: int = 15,
        num_sequences: int = 1000,
        confidence_threshold: float = 0.3,
        context_padding: int = 15,
        input_sequence_length: int = 5,
        default_lead_frames: int = 15,
        min_lead_frames: int = 1,
        max_lead_frames: int = 15,
        bullet_speed_mps: float = 28.0,
        system_latency_s: float = 0.0,
        default_fps: float = 60.0,
        pnp_profile: str = "mer_139_210u3c",
        max_pnp_error: float = 5.0,
        auto_target_type: bool = True,
        target_type: str = TargetGeometry.ARMOR_SMALL,
        scale_intrinsics: bool = True,
    ):
        """
        初始化坐标预测数据集
        
        Args:
            video_path: 视频文件路径
            yolo_model_path: YOLO模型路径
            sequence_length: 序列长度（输入序列 + 预测步长）
            num_sequences: 训练序列数量
            confidence_threshold: YOLO检测置信度阈值
            context_padding: 上下文填充像素
            input_sequence_length: 输入帧数
            default_lead_frames: PnP失败时的默认预测步长
            min_lead_frames: 最小预测步长
            max_lead_frames: 最大预测步长
            bullet_speed_mps: 弹速(m/s)
            system_latency_s: 系统延迟(秒)
            default_fps: 视频FPS读取失败时的默认值
            pnp_profile: 相机内参配置名
            max_pnp_error: PnP重投影误差阈值
            auto_target_type: 是否根据检测类别自动选择目标类型（无类别信息时回退默认类型）
            target_type: 默认目标类型
            scale_intrinsics: 是否根据视频分辨率缩放相机内参
        """
        self.video_path = video_path
        self.yolo_model_path = yolo_model_path
        self.input_sequence_length = max(1, int(input_sequence_length))
        self.min_lead_frames = max(0, int(min_lead_frames))
        self.max_lead_frames = max(self.min_lead_frames, int(max_lead_frames))
        self.default_lead_frames = int(default_lead_frames)
        if self.default_lead_frames < self.min_lead_frames:
            self.default_lead_frames = self.min_lead_frames
        if self.default_lead_frames > self.max_lead_frames:
            self.default_lead_frames = self.max_lead_frames
        self.sequence_length = max(
            int(sequence_length), self.input_sequence_length + self.max_lead_frames
        )
        self.num_sequences = num_sequences
        self.confidence_threshold = confidence_threshold
        self.context_padding = context_padding
        self.frame_shape = None
        self.bullet_speed_mps = max(1e-3, float(bullet_speed_mps))
        self.system_latency_s = max(0.0, float(system_latency_s))
        self.default_fps = max(1.0, float(default_fps))
        self.video_fps = None
        self.pnp_profile = pnp_profile
        self.max_pnp_error = float(max_pnp_error)
        self.auto_target_type = bool(auto_target_type)
        self.target_type = target_type
        self.scale_intrinsics = bool(scale_intrinsics)
        self._intrinsics_base = None
        self._intrinsics_scaled = None
        self._intrinsics_scaled_shape = None
        self._backend_kind = "unknown"
        self.detections: List[Dict[str, Any]] = []
        
        print(f"📹 初始化坐标预测数据集")
        print(f"   - 视频文件: {video_path}")
        print(f"   - YOLO模型: {yolo_model_path}")
        print(f"   - 序列长度: {self.sequence_length}")
        print(f"   - 目标序列数: {num_sequences}")
        print(f"   - 置信度阈值: {confidence_threshold}")
        print(f"   - 输入帧数: {self.input_sequence_length}")
        print(
            f"   - PnP预测步长: {self.min_lead_frames}~{self.max_lead_frames} "
            f"(默认 {self.default_lead_frames})"
        )
        print(f"   - 弹速: {self.bullet_speed_mps} m/s")
        print(f"   - 系统延迟: {self.system_latency_s:.3f} s")
        print(f"   - 默认FPS: {self.default_fps}")
        print(f"   - 相机内参: {self.pnp_profile}")
        print(f"   - PnP误差阈值: {self.max_pnp_error}")
        print(f"   - 自动目标类型: {self.auto_target_type}")
        print(f"   - 内参缩放: {self.scale_intrinsics}")

        try:
            self._intrinsics_base = get_camera_intrinsics(self.pnp_profile)
        except Exception as exc:
            print(f"⚠️ 相机内参配置不可用: {exc}，将使用默认参数")
            self._intrinsics_base = get_camera_intrinsics("default")
        
        # 加载YOLO模型
        self._load_yolo_model()
        
        # 提取帧和检测结果
        print(f"\n🎯 开始提取视频帧和目标坐标...")
        self.frames, self.detections = self._extract_frames_and_detections()
        self.coordinates = [
            np.asarray(det["normalized_bbox"], dtype=np.float32).copy()
            for det in self.detections
            if det.get("normalized_bbox") is not None
        ]
        
        # 检查提取的数据
        if len(self.frames) == 0:
            print("⚠️ 警告：没有提取到任何帧，创建模拟数据用于测试...")
            self._create_mock_data()
        
        if len(self.coordinates) == 0:
            print("⚠️ 警告：没有检测到任何目标，创建模拟坐标数据...")
            self._create_mock_coordinates()
        
        print(f"✓ 数据提取完成")
        print(f"   - 提取帧数: {len(self.frames)}")
        print(f"   - 检测坐标数: {len(self.coordinates)}")
        if self.video_fps is not None:
            print(f"   - 视频FPS: {self.video_fps:.2f}")
        else:
            print(f"   - 视频FPS: 未知(使用默认 {self.default_fps})")
        
        # 生成训练序列
        print(f"\n🔄 生成训练序列...")
        self.sequences = self._generate_sequences()
        
        if len(self.sequences) == 0:
            print("⚠️ 警告：没有生成任何训练序列，创建模拟序列...")
            self._create_mock_sequences()
        
        print(f"✓ 数据集初始化完成")
        print(f"   - 总序列数: {len(self.sequences)}")
    
    def _choose_detection_backend(self) -> str:
        if looks_like_legacy_rm4pt_weight(self.yolo_model_path):
            return "rm4pt"
        if looks_like_legacy_rm4pt_engine(self.yolo_model_path):
            return "rm4pt"
        if self.yolo_model_path.lower().endswith(".engine"):
            sidecar_pt = os.path.splitext(self.yolo_model_path)[0] + ".pt"
            if looks_like_legacy_rm4pt_weight(sidecar_pt):
                return "rm4pt"
        return "standard"

    def _load_yolo_model(self):
        """加载检测模型。RM 四点模型加载失败时直接报错。"""
        print(f"正在加载YOLO模型: {self.yolo_model_path}")
        backend = self._choose_detection_backend()
        self._backend_kind = backend
        print(f"   - 检测后端: {backend}")

        if backend == "rm4pt":
            if self.yolo_model_path.lower().endswith(".engine"):
                self.yolo_model = Legacy4PointTensorRTDetector(
                    self.yolo_model_path,
                    image_size=640,
                )
            else:
                self.yolo_model = Legacy4PointDetector(
                    self.yolo_model_path,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    image_size=640,
                )
            self.yolo_version = "rm4pt"
            print("✓ 成功加载RM四点检测模型")
            return

        # 设置torch hub缓存目录
        torch_home = os.environ.get("TORCH_HOME")
        if torch_home:
            cache_dir = os.path.join(torch_home, "hub")
        else:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub")
        os.makedirs(cache_dir, exist_ok=True)
        torch.hub.set_dir(cache_dir)

        # 方法1: 尝试直接加载本地YOLOv5模型
        try:
            print("尝试直接加载本地YOLOv5模型...")
            self.yolo_model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=self.yolo_model_path,
                force_reload=False,
                trust_repo=True,
                source="local",
            )
            self.yolo_version = "v5"
            self._backend_kind = "yolov5"
            print("✓ 成功加载本地YOLOv5模型")
            return
        except Exception as e:
            print(f"直接加载失败: {e}")

        # 方法2: 尝试从缓存或网络加载YOLOv5模型
        try:
            print("尝试从缓存加载YOLOv5模型...")
            self.yolo_model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=self.yolo_model_path,
                force_reload=False,
                trust_repo=True,
            )
            self.yolo_version = "v5"
            self._backend_kind = "yolov5"
            print("✓ 成功从缓存加载YOLOv5模型")
            return
        except Exception as e:
            print(f"缓存加载失败: {e}")

        # 方法3: 尝试使用ultralytics包加载（支持新的.pt/.engine）
        try:
            print("尝试使用ultralytics包加载...")
            from ultralytics import YOLO

            self.yolo_model = YOLO(self.yolo_model_path, task="detect")
            self.yolo_version = "v8"
            self._backend_kind = "ultralytics"
            print("✓ 成功使用ultralytics加载模型")
            return
        except Exception as e:
            print(f"ultralytics加载失败: {e}")

        raise RuntimeError(f"无法加载检测模型: {self.yolo_model_path}")

    def _resolve_class_name(self, names: Any, class_id: Optional[int]) -> Optional[str]:
        if class_id is None:
            return None
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _build_detection_record(
        self,
        frame_shape: Sequence[int],
        bbox: Sequence[float],
        confidence: float,
        class_id: Optional[int] = None,
        class_name: Optional[str] = None,
        corners: Optional[np.ndarray] = None,
        backend: str = "unknown",
    ) -> Optional[Dict[str, Any]]:
        frame_h, frame_w = frame_shape[:2]
        if frame_h <= 0 or frame_w <= 0 or len(bbox) < 4:
            return None

        x1 = int(np.clip(float(bbox[0]), 0, max(0, frame_w - 1)))
        y1 = int(np.clip(float(bbox[1]), 0, max(0, frame_h - 1)))
        x2 = int(np.clip(float(bbox[2]), 0, max(0, frame_w - 1)))
        y2 = int(np.clip(float(bbox[3]), 0, max(0, frame_h - 1)))
        if x2 <= x1:
            x2 = min(frame_w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(frame_h - 1, y1 + 1)

        width = (x2 - x1) / frame_w
        height = (y2 - y1) / frame_h
        if width <= 0.0 or height <= 0.0:
            return None

        normalized_bbox = np.array(
            [
                ((x1 + x2) / 2.0) / frame_w,
                ((y1 + y2) / 2.0) / frame_h,
                width,
                height,
            ],
            dtype=np.float32,
        )

        normalized_corners = None
        if corners is not None:
            corners_array = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
            if corners_array.shape == (4, 2) and np.all(np.isfinite(corners_array)):
                corners_array = corners_array.copy()
                corners_array[:, 0] = np.clip(corners_array[:, 0], 0, max(0, frame_w - 1))
                corners_array[:, 1] = np.clip(corners_array[:, 1], 0, max(0, frame_h - 1))
                normalized_corners = corners_array

        return {
            "bbox": [x1, y1, x2, y2],
            "normalized_bbox": normalized_bbox,
            "confidence": float(confidence),
            "class": int(class_id) if class_id is not None else None,
            "class_name": class_name,
            "corners": normalized_corners,
            "backend": backend,
        }

    def _extract_detection_from_result_boxes(self, result: Any, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or not hasattr(boxes, "xyxy") or len(boxes.xyxy) <= 0:
            return None

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if hasattr(boxes, "conf") else None
        clss = boxes.cls.detach().cpu().numpy() if hasattr(boxes, "cls") else None
        names = getattr(result, "names", None)
        for idx, box in enumerate(xyxy):
            conf = float(confs[idx]) if confs is not None else 1.0
            if conf < self.confidence_threshold:
                continue
            class_id = int(clss[idx]) if clss is not None else None
            return self._build_detection_record(
                frame.shape,
                bbox=box[:4],
                confidence=conf,
                class_id=class_id,
                class_name=self._resolve_class_name(names, class_id),
                backend=self._backend_kind,
            )
        return None

    def _extract_detection_from_batches(
        self,
        frame: np.ndarray,
        batches: Any,
        names: Any,
    ) -> Optional[Dict[str, Any]]:
        if batches is None:
            return None
        for batch in batches:
            if batch is None or len(batch) == 0:
                continue
            for detection in batch:
                if hasattr(detection, "detach"):
                    det_values = detection.detach().cpu().numpy()
                else:
                    det_values = np.asarray(detection)
                if det_values.shape[0] < 6:
                    continue
                x1, y1, x2, y2, conf, cls = det_values[:6]
                if float(conf) < self.confidence_threshold:
                    continue
                class_id = int(cls)
                return self._build_detection_record(
                    frame.shape,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    class_id=class_id,
                    class_name=self._resolve_class_name(names, class_id),
                    backend=self._backend_kind,
                )
        return None

    def _detect_with_standard_model(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        if self.yolo_version == "v8":
            results = self.yolo_model(frame, verbose=False)
        else:
            results = self.yolo_model(frame)

        iterable = results if isinstance(results, (list, tuple)) else [results]
        for result in iterable:
            detection = self._extract_detection_from_result_boxes(result, frame)
            if detection is not None:
                return detection

            names = getattr(result, "names", None)
            detection = self._extract_detection_from_batches(
                frame,
                getattr(result, "pred", None),
                names,
            )
            if detection is not None:
                return detection

            detection = self._extract_detection_from_batches(
                frame,
                getattr(result, "xyxy", None),
                names,
            )
            if detection is not None:
                return detection
        return None

    def _extract_frames_and_detections(self):
        """提取帧和对应的检测结果。"""
        import cv2

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is not None and fps > 1e-3:
            self.video_fps = float(fps)
        else:
            self.video_fps = None

        frames = []
        detections: List[Dict[str, Any]] = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if self.frame_shape is None:
                self.frame_shape = frame.shape

            detection = self._detect_objects(frame)
            if detection is None:
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (64, 64))
            normalized_frame = resized_frame.astype(np.float32) / 255.0

            frames.append(normalized_frame)
            detections.append(detection)
            frame_count += 1

            if frame_count % 100 == 0:
                print(f"已提取 {frame_count} 帧")

        cap.release()
        print(f"总共提取了 {len(frames)} 帧，检测到 {len(detections)} 个目标坐标")
        return frames, detections

    def _detect_objects(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """检测目标并返回结构化检测结果。"""
        try:
            if self._backend_kind == "rm4pt":
                detections = self.yolo_model.detect(
                    frame,
                    conf_thres=self.confidence_threshold,
                    max_det=1,
                    classes=None,
                )
                for det in detections:
                    if float(det.get("confidence", 0.0)) < self.confidence_threshold:
                        continue
                    return self._build_detection_record(
                        frame.shape,
                        bbox=det.get("bbox", []),
                        confidence=float(det.get("confidence", 1.0)),
                        class_id=det.get("class"),
                        class_name=det.get("class_name"),
                        corners=det.get("corners"),
                        backend=str(det.get("backend", "rm4pt")),
                    )
                return None

            return self._detect_with_standard_model(frame)
        except Exception as e:
            print(f"目标检测失败: {e}")
            return None

    def _get_intrinsics_for_frame(self, frame_shape: tuple) -> Optional[CameraIntrinsics]:
        if self._intrinsics_base is None:
            return None
        if not self.scale_intrinsics:
            return self._intrinsics_base
        frame_h, frame_w = frame_shape[:2]
        if (
            self._intrinsics_scaled is not None
            and self._intrinsics_scaled_shape == (frame_w, frame_h)
        ):
            return self._intrinsics_scaled
        scaled = scale_intrinsics_to_frame(self._intrinsics_base, frame_shape)
        self._intrinsics_scaled = scaled
        self._intrinsics_scaled_shape = (frame_w, frame_h)
        return scaled

    def _bbox_from_normalized(self, coord: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        if self.frame_shape is None:
            return None
        x_center, y_center, w, h = coord
        if w <= 0.0 or h <= 0.0:
            return None
        frame_h, frame_w = self.frame_shape[:2]
        x1 = int(max(0, min(frame_w - 1, (x_center - w / 2.0) * frame_w)))
        x2 = int(max(0, min(frame_w - 1, (x_center + w / 2.0) * frame_w)))
        y1 = int(max(0, min(frame_h - 1, (y_center - h / 2.0) * frame_h)))
        y2 = int(max(0, min(frame_h - 1, (y_center + h / 2.0) * frame_h)))
        if x2 <= x1:
            x2 = min(frame_w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(frame_h - 1, y1 + 1)
        return (x1, y1, x2, y2)

    def _resolve_target_type(
        self,
        target_class: Optional[int] = None,
        target_class_name: Optional[str] = None,
    ) -> str:
        target_type = self.target_type
        if self.auto_target_type:
            target_type = choose_target_type_by_class(
                class_id=target_class,
                class_name=target_class_name,
                default_type=target_type,
            )
        return target_type

    def _estimate_distance_from_image_points(
        self,
        image_points: Optional[np.ndarray],
        target_class: Optional[int] = None,
        target_class_name: Optional[str] = None,
    ) -> Optional[float]:
        if image_points is None or self.frame_shape is None:
            return None
        points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
        if points.shape != (4, 2) or not np.all(np.isfinite(points)):
            return None

        intrinsics = self._get_intrinsics_for_frame(self.frame_shape)
        if intrinsics is None:
            return None

        frame_h, frame_w = self.frame_shape[:2]
        points = points.copy()
        points[:, 0] = np.clip(points[:, 0], 0, max(0, frame_w - 1))
        points[:, 1] = np.clip(points[:, 1], 0, max(0, frame_h - 1))
        pose = solve_pose(
            points,
            target_type=self._resolve_target_type(target_class, target_class_name),
            intrinsics=intrinsics,
        )
        if not pose.success or pose.tvec is None or pose.reprojection_error is None:
            return None
        if pose.reprojection_error > self.max_pnp_error:
            return None
        return compute_target_distance(pose.tvec)

    def _estimate_distance_from_bbox(
        self,
        bbox: Optional[tuple[int, int, int, int]],
        target_class: Optional[int] = None,
        target_class_name: Optional[str] = None,
    ) -> Optional[float]:
        if bbox is None or self.frame_shape is None:
            return None
        x1, y1, x2, y2 = bbox
        bbox_points = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
        )
        return self._estimate_distance_from_image_points(
            bbox_points,
            target_class=target_class,
            target_class_name=target_class_name,
        )

    def _estimate_distance_from_detection(
        self,
        detection: Optional[Dict[str, Any]],
    ) -> Optional[float]:
        if detection is None:
            return None

        target_class = detection.get("class")
        target_class_name = detection.get("class_name")
        corners = detection.get("corners")
        if corners is not None:
            distance = self._estimate_distance_from_image_points(
                corners,
                target_class=target_class,
                target_class_name=target_class_name,
            )
            if distance is not None:
                return distance

        bbox = detection.get("bbox")
        if bbox is None and detection.get("normalized_bbox") is not None:
            bbox = self._bbox_from_normalized(
                np.asarray(detection["normalized_bbox"], dtype=np.float32)
            )
        return self._estimate_distance_from_bbox(
            bbox,
            target_class=target_class,
            target_class_name=target_class_name,
        )

    def _resolve_fps(self) -> float:
        if self.video_fps is None or self.video_fps <= 1e-3:
            return self.default_fps
        return self.video_fps

    def _estimate_lead_frames(self, detection_or_coord: Any) -> int:
        if isinstance(detection_or_coord, dict):
            distance = self._estimate_distance_from_detection(detection_or_coord)
        else:
            coord = np.asarray(detection_or_coord, dtype=np.float32)
            distance = self._estimate_distance_from_bbox(self._bbox_from_normalized(coord))
        if distance is None:
            return self.default_lead_frames
        lead_time = distance / self.bullet_speed_mps + self.system_latency_s
        frames = int(np.ceil(lead_time * self._resolve_fps()))
        if frames < self.min_lead_frames:
            frames = self.min_lead_frames
        if frames > self.max_lead_frames:
            frames = self.max_lead_frames
        return frames
    
    def _generate_sequences(self):
        """生成训练序列"""
        sequences = []
        
        for _ in range(self.num_sequences):
            try:
                # 随机选择起始帧
                if len(self.frames) < self.sequence_length:
                    continue
                
                start_idx = np.random.randint(0, len(self.frames) - self.sequence_length + 1)
                
                # 提取帧序列和坐标序列
                frame_sequence = self.frames[start_idx:start_idx + self.sequence_length]
                coord_sequence = self.coordinates[start_idx:start_idx + self.sequence_length]
                detection_sequence = self.detections[start_idx:start_idx + self.sequence_length]
                if len(coord_sequence) < self.input_sequence_length:
                    continue

                lead_frames = self._estimate_lead_frames(
                    detection_sequence[self.input_sequence_length - 1]
                )
                target_index = self.input_sequence_length - 1 + lead_frames
                if target_index >= len(coord_sequence):
                    target_index = len(coord_sequence) - 1
                    lead_frames = max(
                        0, target_index - (self.input_sequence_length - 1)
                    )

                sequences.append({
                    'frames': frame_sequence,
                    'coordinates': coord_sequence,
                    'detections': detection_sequence,
                    'lead_frames': lead_frames,
                    'target_index': target_index,
                })
            except Exception as e:
                continue
        
        print(f"成功生成 {len(sequences)} 个训练序列")
        if sequences:
            lead_values = [seq.get('lead_frames', self.default_lead_frames) for seq in sequences]
            print(
                "预测步长统计: "
                f"min={min(lead_values)}, "
                f"mean={np.mean(lead_values):.2f}, "
                f"max={max(lead_values)}"
            )
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # 输入：前N帧的帧数据和坐标
        input_frames_np = np.asarray(
            sequence['frames'][:self.input_sequence_length], dtype=np.float32
        )  # (N, 64, 64)
        input_coords_np = np.asarray(
            sequence['coordinates'][:self.input_sequence_length], dtype=np.float32
        )  # (N, 4)
        input_frames = torch.from_numpy(input_frames_np)
        input_coords = torch.from_numpy(input_coords_np)

        # 目标：基于PnP估计步长的目标帧坐标
        target_index = sequence.get('target_index', len(sequence['coordinates']) - 1)
        target_index = min(max(0, int(target_index)), len(sequence['coordinates']) - 1)
        target_coords = torch.from_numpy(
            np.asarray(sequence['coordinates'][target_index], dtype=np.float32)
        )  # (4,)
        
        # 展平帧数据
        input_frames = input_frames.view(input_frames.size(0), -1)  # (N, 4096)
        
        return input_frames, input_coords, target_coords

    def _create_mock_data(self):
        """创建模拟的帧数据用于测试"""
        print("创建模拟帧数据...")
        if self.frame_shape is None:
            self.frame_shape = (64, 64, 3)
        if self.video_fps is None:
            self.video_fps = self.default_fps
        self.frames = []
        for i in range(100):  # 创建100帧模拟数据
            # 创建64x64的模拟灰度图像
            mock_frame = np.random.rand(64, 64).astype(np.float32)
            self.frames.append(mock_frame)
        self.detections = []
        print(f"✓ 创建了 {len(self.frames)} 帧模拟数据")
    
    def _create_mock_coordinates(self):
        """创建模拟的坐标数据用于测试"""
        print("创建模拟坐标数据...")
        self.coordinates = []
        self.detections = []
        for i in range(len(self.frames)):
            # 创建随机的相对坐标 (x_center, y_center, width, height)
            x_center = np.random.uniform(0.2, 0.8)
            y_center = np.random.uniform(0.2, 0.8)
            width = np.random.uniform(0.1, 0.4)
            height = np.random.uniform(0.1, 0.4)
            coord = np.array([x_center, y_center, width, height], dtype=np.float32)
            self.coordinates.append(coord)
            bbox = self._bbox_from_normalized(coord)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            detection = self._build_detection_record(
                self.frame_shape,
                bbox=(x1, y1, x2, y2),
                confidence=1.0,
                backend="mock",
            )
            if detection is not None:
                detection["normalized_bbox"] = coord.copy()
                self.detections.append(detection)
        print(f"✓ 创建了 {len(self.coordinates)} 个模拟坐标")
    
    def _create_mock_sequences(self):
        """创建模拟的训练序列用于测试"""
        print("创建模拟训练序列...")
        self.sequences = []
        for i in range(min(self.num_sequences, 100)):  # 最多创建100个序列
            try:
                # 随机选择起始帧
                if len(self.frames) < self.sequence_length:
                    continue
                
                start_idx = np.random.randint(0, len(self.frames) - self.sequence_length + 1)
                
                # 提取帧序列和坐标序列
                frame_sequence = self.frames[start_idx:start_idx + self.sequence_length]
                coord_sequence = self.coordinates[start_idx:start_idx + self.sequence_length]
                detection_sequence = self.detections[start_idx:start_idx + self.sequence_length]

                if len(coord_sequence) < self.input_sequence_length:
                    continue

                lead_frames = self._estimate_lead_frames(
                    detection_sequence[self.input_sequence_length - 1]
                )
                target_index = self.input_sequence_length - 1 + lead_frames
                if target_index >= len(coord_sequence):
                    target_index = len(coord_sequence) - 1
                    lead_frames = max(
                        0, target_index - (self.input_sequence_length - 1)
                    )

                self.sequences.append({
                    'frames': frame_sequence,
                    'coordinates': coord_sequence,
                    'detections': detection_sequence,
                    'lead_frames': lead_frames,
                    'target_index': target_index,
                })
            except Exception as e:
                continue
        
        print(f"✓ 创建了 {len(self.sequences)} 个模拟训练序列")

def test_coordinate_prediction_model():
    """测试坐标预测模型"""
    print("测试坐标预测模型...")
    
    batch_size = 4
    sequence_length = 5  # 修改为5帧输入
    input_size = 64 * 64  # 帧数据
    hidden_size = 128
    coordinate_dim = 4
    
    model = CoordinatePredictionModel(
        input_size=input_size,
        hidden_size=hidden_size,
        coordinate_dim=coordinate_dim
    )
    
    print(f"模型信息:")
    model_info = model.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # 创建测试输入
    input_frames = torch.randn(batch_size, sequence_length, input_size)
    
    print(f"\n输入形状: {input_frames.shape}")
    
    # 前向传播
    outputs = model(input_frames)
    
    print(f"\n输出形状:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print("\n坐标预测模型测试完成！")
    
    return model

if __name__ == "__main__":
    test_coordinate_prediction_model()
