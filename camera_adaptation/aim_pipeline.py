#!/usr/bin/env python
# coding=utf-8

"""
Aim pipeline: detect armor, predict motion, compute yaw/pitch (PnP), send UART frames.
Focus: lock target with confidence-gated firing on prediction output.
"""

import math
import os
import time
import threading
import queue
import glob
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import serial
import torch

from camera_adaptation.industrial_camera_processor import IndustrialCameraProcessor
from camera_adaptation.ballistics import solve_pitch_for_target
from camera_adaptation.pnp_solver import (
    solve_angles_from_bbox,
    get_camera_intrinsics,
    choose_target_type_by_detection,
    scale_intrinsics_to_frame,
    detect_armor_corners,
    TargetGeometry,
)
from camera_adaptation.uart_sender import build_armor_packet
from coordinate_prediction_model import CoordinatePredictionModel

try:
    from camerapytest import CameraAPI  # type: ignore
except Exception:
    CameraAPI = None  # type: ignore


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
#DEFAULT_YOLO_MODEL = "best.pt"
DEFAULT_YOLO_MODEL = "best.engine"
#DEFAULT_YOLO_MODEL = "/home/nvidia/paper_arm/best.engine"



DEFAULT_COORD_MODEL = os.path.join(
    "coordinate_prediction_models",
    "Coordinate-Prediction-Model_best.pth",
)

SERIAL_PORT_DEFAULT = "/dev/ttyUSB0"
SERIAL_BAUD_DEFAULT = 115200
SERIAL_TIMEOUT_DEFAULT = 0

DEFAULT_DAHENG_CONFIG = os.path.join(
    BASE_DIR,
    "MER-139-210U3C(KE0210010002).txt",
)
if not os.path.exists(DEFAULT_DAHENG_CONFIG):
    DEFAULT_DAHENG_CONFIG = None


@dataclass
class _PerfBucket:
    percentiles: Tuple[float, ...]
    samples: dict = field(default_factory=dict)
    frame_count: int = 0

    def add(self, metrics: dict):
        self.frame_count += 1
        for name, value in metrics.items():
            if value is None:
                continue
            try:
                number = float(value)
            except Exception:
                continue
            if not np.isfinite(number):
                continue
            self.samples.setdefault(name, []).append(number)

    def clear(self):
        self.samples.clear()
        self.frame_count = 0

    def summarize(self, name: str) -> Optional[dict]:
        values = self.samples.get(name)
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float32)
        stats = {
            "avg": float(arr.mean()),
            "max": float(arr.max()),
        }
        for percentile in self.percentiles:
            label = (
                f"p{int(percentile)}"
                if float(percentile).is_integer()
                else f"p{str(percentile).replace('.', '_')}"
            )
            stats[label] = float(np.percentile(arr, percentile))
        return stats


@dataclass
class MotionObservation:
    ts: float
    yaw_deg: float
    pitch_deg: float
    quad_area_px: Optional[float]
    quad_aspect: Optional[float]
    distance_m: Optional[float]
    normal_yaw_deg: Optional[float]


class AimPipeline:
    """调度：标框检测 -> 预测 -> PnP角度 -> 串口发送"""

    def __init__(
        self,
        yolo_model_path: str,
        coord_model_path: str,
        uart_port: str,
        uart_baud: int,
        send_rate: float,
        confidence_threshold: float,
        camera_id: int,
        use_daheng: bool,
        daheng_config: Optional[str],
        daheng_sn: Optional[str],
        swap_rb: bool,
        pnp_profile: str,
        target_type: str,
        target_color: Optional[str],
        max_pnp_error: float,
        angle_scale: float,
        input_sequence_length: int,
        sequence_length: int,
        context_padding: int,
        max_yaw_rate: float,
        max_pitch_rate: float,
        lost_threshold: int,
        show_window: bool,
        window_name: str,
        display_max_fps: float,
        invert_yaw: bool,
        invert_pitch: bool,
        show_tx: bool,
        gun_offset_y: float,  # 枪口相对于相机光心的y轴偏移（毫米）
        scale_intrinsics: bool,
        auto_target_type: bool,
        use_corners: bool,
        bbox_shrink: float,
        target_class_ids: Optional[Sequence[int]] = None,
        exclude_class_ids: Optional[Sequence[int]] = None,
        yolo_max_det: int = 1,
        yolo_log_speed: bool = False,
        yolo_verbose: bool = False,
        yolo_imgsz: Optional[Tuple[int, int]] = None,
        detector_backend: str = "auto",
        perf_log: bool = False,
        perf_log_interval_s: float = 1.0,
        perf_log_percentiles: Tuple[float, ...] = (50.0, 90.0),
        perf_profile_hint: bool = True,
        cv_threads: int = 0,
        torch_threads: int = 0,
        profile_pred: bool = False,
        bullet_speed_mps: float = 22.0,
        system_latency_s: float = 0.115,
        max_comp_distance_m: float = 6.0,
        model_bullet_speed_mps: float = 28.0,
        model_latency_s: float = 0.0,
        enable_prediction: bool = True,
        pred_async: bool = False,
        pred_max_lag: int = 1,
        lag_comp_enable: bool = True,
        lag_comp_max_s: float = 0.120,
        ballistic_enable: bool = False,
        ballistic_drag_k: float = 0.02,
        ballistic_pitch_min_deg: float = -100.0,
        ballistic_pitch_max_deg: float = 100.0,
        ballistic_dt_ms: float = 1.0,
        use_ballistic_time: bool = False,
        enable_ec_feedback: bool = True,
        ec_feedback_invert_yaw: bool = False,
        ec_feedback_invert_pitch: bool = False,
        ec_t0_s: float = 0.060,
        ec_additional_predict_time_s: float = 0.0,
        spin_aware: bool = True,
        spin_enter_threshold: float = 0.66,
        spin_exit_threshold: float = 0.45,
        spin_yaw_reverse_bias_deg: float = 5.0,
        spin_yaw_dir_lock_min_conf: float = 0.75,
        spin_yaw_dir_min_rate_dps: float = 10.0,
        spin_yaw_dir_lock_threshold: float = 4.0,
        spin_yaw_dir_switch_min_conf: float = 0.75,
        spin_yaw_dir_switch_min_rate_dps: float = 15.0,
        spin_yaw_dir_switch_threshold: float = 4.0,
        disable_image_time_comp_with_feedback: bool = True,
        fire_confidence_threshold: float = 0.6,
        fire_force_interval_s: float = 1.0,
        record_video: bool = False,
        record_path: Optional[str] = None,
        record_fps: float = 30.0,
        record_fourcc: str = "XVID",
        rate_fast_alpha: float = 0.25,
        rate_slow_alpha: float = 0.08,
        max_ec_lead_deg: float = 20.0,
    ):
        self.cv_threads = max(0, int(cv_threads))
        self.torch_threads = max(0, int(torch_threads))
        self.display_max_fps = max(1.0, float(display_max_fps))
        self.perf_profile_hint = bool(perf_profile_hint)
        self._apply_runtime_thread_hints()
        self.detector = IndustrialCameraProcessor(
            yolo_model_path=yolo_model_path,
            confidence_threshold=confidence_threshold,
            yolo_max_det=yolo_max_det,
            yolo_log_speed=yolo_log_speed,
            yolo_verbose=yolo_verbose,
            yolo_imgsz=yolo_imgsz,
            detector_backend=detector_backend,
        )
        self.camera_id = camera_id
        self.use_daheng = use_daheng
        self.daheng_config = daheng_config or None
        if self.daheng_config and not os.path.exists(self.daheng_config):
            print(f"⚠️ 大恒配置文件不存在，将忽略: {self.daheng_config}")
            self.daheng_config = None
        self.daheng_sn = daheng_sn
        self.swap_rb = swap_rb

        self.uart_port = uart_port
        self.uart_baud = uart_baud
        self.send_rate = max(1e-3, float(send_rate))
        self.predict_fps = max(1e-3, self.send_rate)
        self.angle_scale = float(angle_scale)
        self.show_window = bool(show_window)
        self.window_name = window_name
        self.invert_yaw = bool(invert_yaw)
        self.invert_pitch = bool(invert_pitch)
        self.show_tx = show_tx
        self.target_color = self._normalize_target_color(target_color)
        self.target_class_ids = self._normalize_class_ids(target_class_ids, "target_class_ids")
        self.exclude_class_ids = self._normalize_class_ids(exclude_class_ids, "exclude_class_ids")
        self._excluded_class_ids = set(self.exclude_class_ids or ())
        self._target_color_class_ids = None
        if self.target_class_ids is not None:
            allowed_class_ids = self.target_class_ids
        elif self.target_color is not None:
            resolved_color_ids = self.detector.get_color_class_ids(self.target_color)
            if not resolved_color_ids:
                raise ValueError(
                    f"无法为 target_color={self.target_color!r} 解析 rm4pt 类别ID，请检查 sidecar .pt 类别名"
                )
            allowed_class_ids = tuple(int(class_id) for class_id in resolved_color_ids)
        else:
            allowed_class_ids = None
        if allowed_class_ids is not None:
            filtered_class_ids = tuple(
                class_id for class_id in allowed_class_ids if class_id not in self._excluded_class_ids
            )
            if not filtered_class_ids:
                raise ValueError("exclude_class_ids 过滤后没有剩余可用类别")
            self._target_color_class_ids = filtered_class_ids
        else:
            self._target_color_class_ids = None
        self._detector_class_ids = self._target_color_class_ids
        self.perf_log = bool(perf_log)
        self.perf_log_interval_s = max(0.2, float(perf_log_interval_s))
        raw_percentiles = tuple(float(p) for p in perf_log_percentiles) if perf_log_percentiles else (50.0, 90.0)
        self.perf_log_percentiles = tuple(
            sorted({min(100.0, max(0.0, float(p))) for p in raw_percentiles})
        )
        self._perf_buckets = {
            "all": _PerfBucket(self.perf_log_percentiles),
            "target": _PerfBucket(self.perf_log_percentiles),
            "idle": _PerfBucket(self.perf_log_percentiles),
        }
        self._worker_perf_lock = threading.Lock()
        self._worker_perf_buckets = {
            "display": _PerfBucket(self.perf_log_percentiles),
            "send": _PerfBucket(self.perf_log_percentiles),
        }
        self._perf_last_log_time = time.time()
        self.profile_pred = bool(profile_pred)
        self.enable_prediction = bool(enable_prediction)
        self.pred_async = bool(pred_async) and self.enable_prediction
        self.pred_max_lag = max(0, int(pred_max_lag))
        self.lag_comp_enable = bool(lag_comp_enable)
        self.lag_comp_max_s = max(0.0, float(lag_comp_max_s))
        self.gun_offset_y = float(gun_offset_y) / 1000.0  # 转换为米
        self.bullet_speed_mps = max(1e-3, float(bullet_speed_mps))
        self.system_latency_s = max(0.0, float(system_latency_s))
        self.max_comp_distance_m = max(0.0, float(max_comp_distance_m))
        self.model_bullet_speed_mps = max(0.0, float(model_bullet_speed_mps))
        self.model_latency_s = max(0.0, float(model_latency_s))
        self.ballistic_enable = bool(ballistic_enable)
        self.ballistic_drag_k = max(0.0, float(ballistic_drag_k))
        pitch_min = float(ballistic_pitch_min_deg)
        pitch_max = float(ballistic_pitch_max_deg)
        if pitch_min > pitch_max:
            pitch_min, pitch_max = pitch_max, pitch_min
        self.ballistic_pitch_min_rad = math.radians(pitch_min)
        self.ballistic_pitch_max_rad = math.radians(pitch_max)
        self.ballistic_dt_s = max(1e-4, float(ballistic_dt_ms) / 1000.0)
        self.ballistic_g = 9.81
        self.use_ballistic_time = bool(use_ballistic_time)

        self.enable_ec_feedback = bool(enable_ec_feedback)
        self.ec_feedback_invert_yaw = bool(ec_feedback_invert_yaw)
        self.ec_feedback_invert_pitch = bool(ec_feedback_invert_pitch)
        self.ec_t0_s = max(0.0, float(ec_t0_s))
        self.ec_additional_predict_time_s = float(ec_additional_predict_time_s)
        self.max_ec_lead_deg = max(1.0, float(max_ec_lead_deg))
        self.spin_aware = bool(spin_aware)
        self.spin_enter_threshold = max(0.0, min(1.0, float(spin_enter_threshold)))
        self.spin_exit_threshold = max(0.0, min(1.0, float(spin_exit_threshold)))
        if self.spin_exit_threshold > self.spin_enter_threshold:
            self.spin_exit_threshold = self.spin_enter_threshold
        self.spin_yaw_reverse_bias_deg = max(0.0, float(spin_yaw_reverse_bias_deg))
        self.spin_yaw_dir_lock_min_conf = max(0.0, min(1.0, float(spin_yaw_dir_lock_min_conf)))
        self.spin_yaw_dir_min_rate_dps = max(0.0, float(spin_yaw_dir_min_rate_dps))
        self.spin_yaw_dir_lock_threshold = max(1e-3, float(spin_yaw_dir_lock_threshold))
        self.spin_yaw_dir_switch_min_conf = max(0.0, min(1.0, float(spin_yaw_dir_switch_min_conf)))
        self.spin_yaw_dir_switch_min_rate_dps = max(
            self.spin_yaw_dir_min_rate_dps,
            float(spin_yaw_dir_switch_min_rate_dps),
        )
        self.spin_yaw_dir_switch_threshold = max(
            self.spin_yaw_dir_lock_threshold,
            float(spin_yaw_dir_switch_threshold),
        )
        self.disable_image_time_comp_with_feedback = bool(disable_image_time_comp_with_feedback)
        self.fire_confidence_threshold = max(0.0, min(1.0, float(fire_confidence_threshold)))
        self.fire_force_interval_s = max(0.0, float(fire_force_interval_s))
        self._last_fire_ts = time.time()
        self._last_fire_reason = "INIT"

        self._ec_thread = None
        self._ec_stop_event = threading.Event()
        self._ec_lock = threading.Lock()
        self._ec_latest = None  # (yaw_deg, pitch_deg, yaw_rate_dps, pitch_rate_dps, mode, shoot, ts)
        self._ec_last_ts = None
        self._ec_last_yaw_wrapped = None
        self._ec_last_pitch_wrapped = None
        self._ec_last_yaw_unwrapped = None
        self._stop_requested = threading.Event()
        self._display_thread = None
        self._display_stop_event = threading.Event()
        self._display_lock = threading.Lock()
        self._display_snapshot = None
        self._display_version = 0
        self._display_dropped_pending = 0
        self._send_thread = None
        self._send_stop_event = threading.Event()
        self._send_lock = threading.Lock()
        self._send_snapshot = None
        self._send_version = 0
        self._send_last_warn_time = 0.0
        self._tx_lock = threading.Lock()
        self._tx_total_count = 0
        self._tx_total_bytes = 0

        self.camera = None
        self.cap = None
        self.serial = None
        self.record_video = bool(record_video)
        self.record_path = record_path
        self.record_fps = max(1.0, float(record_fps))
        self.record_fourcc = (str(record_fourcc or "XVID").strip().upper() or "XVID")[:4]
        if len(self.record_fourcc) < 4:
            self.record_fourcc = self.record_fourcc.ljust(4, "X")
        self._record_writer = None
        self._record_path_resolved = self._prepare_record_path() if self.record_video else None
        self._record_frame_size = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.coordinate_model = (
            self._load_coordinate_model(coord_model_path) if self.enable_prediction else None
        )
        self._frame_id = 0
        self.input_sequence_length = max(1, int(input_sequence_length))
        self.sequence_length = max(int(sequence_length), self.input_sequence_length)
        self.context_padding = max(0, int(context_padding))
        self.max_yaw_rate = max(0.0, float(max_yaw_rate))
        self.max_pitch_rate = max(0.0, float(max_pitch_rate))
        self.lost_threshold = max(0, int(lost_threshold))
        self._roi_size = (64, 64)
        self.frame_buffer = np.zeros(
            (self.sequence_length, self._roi_size[1], self._roi_size[0]), dtype=np.float32
        )
        self.coord_buffer = np.zeros((self.sequence_length, 4), dtype=np.float32)
        self.roi_bbox_buffer = np.zeros((self.sequence_length, 4), dtype=np.int32)
        self._buffer_idx = 0
        self._buffer_len = 0
        self._pred_input_cpu = None
        self._pred_input_gpu = None
        if (
            self.enable_prediction
            and self.device.startswith("cuda")
            and torch.cuda.is_available()
        ):
            pred_shape = (1, self.input_sequence_length, self._roi_size[0] * self._roi_size[1])
            try:
                self._pred_input_cpu = torch.empty(
                    pred_shape, dtype=torch.float32, pin_memory=True
                )
                self._pred_input_gpu = torch.empty(
                    pred_shape, dtype=torch.float32, device=self.device
                )
            except Exception:
                self._pred_input_cpu = None
                self._pred_input_gpu = None
        self._pred_lock = threading.Lock()
        self._pred_queue = queue.Queue(maxsize=1) if self.pred_async else None
        self._pred_result = None
        self._pred_result_lock = threading.Lock()
        self._pred_stop_event = threading.Event()
        self._pred_thread = None
        self._last_pred_lag = None
        self._frame_period_alpha = 0.2
        self._frame_period_ema_s = 1.0 / self.predict_fps
        self._last_frame_ts = None

        try:
            self.camera_intrinsics = get_camera_intrinsics(pnp_profile)
        except Exception as exc:
            print(f"⚠️ 相机内参配置不可用: {exc}，将使用默认参数")
            self.camera_intrinsics = get_camera_intrinsics("default")
        self._intrinsics_base = self.camera_intrinsics
        self._intrinsics_scaled = None
        self._intrinsics_scaled_shape = None
        self.scale_intrinsics = bool(scale_intrinsics)

        self.target_type = target_type
        self.max_pnp_error = max_pnp_error

        empty_packet = build_armor_packet(
            yaw_deg=0,
            pitch_deg=0,
            armor_x=0,
            armor_y=0,
            armor_cmd=0x00,
            fire_cmd=0x00,
            angle_scale=self.angle_scale,
        )
        self._empty_packet = empty_packet
        self._latest_detection_packet = empty_packet
        self._latest_prediction_packet = empty_packet
        self._send_snapshot = {
            "packet": bytes(empty_packet),
            "info": None,
            "flight_time_s": None,
            "comp_distance_m": None,
            "lead_frames": 0,
            "comp_time_s": 0.0,
            "lag_frames": 0,
            "lag_time_s": 0.0,
            "fire_reason": "INIT",
        }
        self._latest_detection_bbox = None
        self._latest_prediction_bbox = None
        self._latest_detection_info = None
        self._latest_prediction_info = None
        self._latest_detection_conf = None
        self._latest_detection_corners = None
        self._latest_prediction_corners = None
        self._angle_sign_threshold = 0.1
        self._fps_count = 0
        self._fps_last_time = time.time()
        self._fps_value = 0.0
        self._last_read_ms = 0.0
        self._last_yolo_ms = 0.0
        self._last_select_ms = 0.0
        self._last_detector_pre_ms = 0.0
        self._last_detector_infer_ms = 0.0
        self._last_detector_post_ms = 0.0
        self._last_detector_output_count = 0
        self._last_detector_raw_candidates = 0
        self._last_detector_obj_candidates = 0
        self._last_detector_class_candidates = 0
        self._last_detector_kept_candidates = 0
        self._last_pnp_ms = 0.0
        self._last_detection_bbox = None
        self._last_det_angles = None
        self._last_pred_angles = None
        self._last_det_angle_time = None
        self._last_pred_angle_time = None
        self._last_buffer_len = 0
        self._pred_ready = False
        self._last_pred_center = None
        self._last_pred_size = None
        self._last_pred_time = None
        self._pred_velocity = (0.0, 0.0)
        self._last_rel_angles_for_rate = None
        self._last_rel_angle_time_for_rate = None
        self._target_yaw_rate_dps = 0.0
        self._target_pitch_rate_dps = 0.0
        self._target_rate_alpha = float(rate_fast_alpha)
        self._target_rate_fast_alpha = float(rate_fast_alpha)
        self._target_rate_slow_alpha = float(rate_slow_alpha)
        self._target_yaw_rate_fast_dps = 0.0
        self._target_pitch_rate_fast_dps = 0.0
        self._target_yaw_rate_slow_dps = 0.0
        self._target_pitch_rate_slow_dps = 0.0
        self._target_yaw_rate_effective_dps = 0.0
        self._target_pitch_rate_effective_dps = 0.0
        self._motion_history = deque(maxlen=32)
        self._spin_confidence = 0.0
        self._spin_active = False
        self._spin_yaw_direction_score = 0.0
        self._spin_yaw_direction_locked = 0
        self._spin_yaw_fake_rate_dps = 0.0
        self._last_spin_yaw_bias_deg = 0.0
        self._last_valid_motion_rvec_ts = None
        self._last_image_time_comp_scale = 1.0
        self._lost_count = 0
        self._last_distance_m = None
        self._last_raw_distance_m = None
        self._last_comp_distance_m = None
        self._last_comp_time_s = 0.0
        self._last_extra_time_s = 0.0
        self._last_lead_frames = 0
        self._last_lag_time_s = 0.0
        self._last_lag_frames = 0
        self._last_flight_time_s = None
        self._last_pnp_tvec = None
        self._last_pnp_rvec = None
        self._last_det_ballistic_delta = None
        self._last_ballistic_time_s = None
        self.auto_target_type = bool(auto_target_type)
        self.use_corners = bool(use_corners)
        self._last_det_target_type = None
        self._last_pred_target_type = None
        self._last_det_comp = None
        self.bbox_shrink = float(bbox_shrink)
        if self.bbox_shrink < 0.1:
            self.bbox_shrink = 0.1
        if self.bbox_shrink > 1.0:
            self.bbox_shrink = 1.0
        self._last_update_ms = 0.0
        self._last_det_angle_ms = 0.0
        self._last_det_ballistic_ms = 0.0
        self._last_det_ec_ms = 0.0
        self._last_det_rate_limit_ms = 0.0
        self._last_det_packet_ms = 0.0
        self._last_pred_ms = 0.0
        self._last_pred_pre_ms = 0.0
        self._last_pred_h2d_ms = 0.0
        self._last_pred_fwd_ms = 0.0
        self._last_pred_d2h_ms = 0.0
        self._last_pred_resolve_ms = 0.0
        self._last_pred_ec_ms = 0.0
        self._last_pred_rate_limit_ms = 0.0
        self._last_pred_packet_ms = 0.0
        self._last_fire_decide_ms = 0.0
        self._last_det_corner_ms = 0.0
        self._last_pred_corner_ms = 0.0
        self._last_draw_ms = 0.0
        self._last_present_ms = 0.0
        self._last_waitkey_ms = 0.0
        self._last_imshow_ms = 0.0
        self._last_serial_ms = 0.0
        self._last_record_ms = 0.0
        self._last_loop_ms = 0.0

    def _apply_runtime_thread_hints(self):
        if self.cv_threads > 0:
            try:
                cv2.setNumThreads(self.cv_threads)
            except Exception as exc:
                print(f"⚠️ 设置 OpenCV 线程数失败: {exc}")
        if self.torch_threads > 0:
            try:
                torch.set_num_threads(self.torch_threads)
            except Exception as exc:
                print(f"⚠️ 设置 PyTorch 线程数失败: {exc}")
            try:
                torch.set_num_interop_threads(self.torch_threads)
            except Exception:
                pass

    @staticmethod
    def _read_text_file(path: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return None

    def _prepare_record_path(self) -> str:
        if self.record_path:
            path = os.path.abspath(self.record_path)
        else:
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            path = os.path.join(BASE_DIR, "recordings", f"match_{ts}.avi")
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        return path

    def _ensure_record_writer(self, frame_shape: Tuple[int, int, int]) -> bool:
        if not self.record_video:
            return False
        frame_h, frame_w = frame_shape[:2]
        frame_size = (int(frame_w), int(frame_h))
        if (
            self._record_writer is not None
            and self._record_frame_size == frame_size
            and self._record_path_resolved is not None
        ):
            return True

        self._close_record_writer()
        record_path = self._record_path_resolved or self._prepare_record_path()
        fourcc = cv2.VideoWriter_fourcc(*self.record_fourcc)
        writer = cv2.VideoWriter(record_path, fourcc, self.record_fps, frame_size)
        if not writer.isOpened():
            print(f"❌ 无法打开录制文件: {record_path}")
            return False

        self._record_writer = writer
        self._record_path_resolved = record_path
        self._record_frame_size = frame_size
        print(
            f"🎥 比赛视频录制已开启: {record_path} "
            f"({frame_size[0]}x{frame_size[1]} @ {self.record_fps:.1f} FPS, {self.record_fourcc})"
        )
        return True

    def _write_record_frame(self, frame: np.ndarray):
        if not self.record_video:
            self._last_record_ms = 0.0
            return
        start = time.perf_counter()
        if not self._ensure_record_writer(frame.shape):
            self._last_record_ms = (time.perf_counter() - start) * 1000.0
            return
        try:
            self._record_writer.write(frame)
        except Exception as exc:
            print(f"❌ 录制视频写入失败: {exc}")
            self._close_record_writer()
        self._last_record_ms = (time.perf_counter() - start) * 1000.0

    def _close_record_writer(self):
        if self._record_writer is None:
            return
        try:
            self._record_writer.release()
        except Exception:
            pass
        self._record_writer = None
        self._record_frame_size = None

    def _log_perf_profile_hint(self):
        if not self.perf_profile_hint or shutil.which("jetson_clocks") is None:
            return

        governor_files = sorted(glob.glob("/sys/devices/system/cpu/cpufreq/policy*/scaling_governor"))
        governors = [self._read_text_file(path) for path in governor_files]
        governors = [gov for gov in governors if gov]
        hint_parts = []
        if governors and any(gov != "performance" for gov in governors):
            hint_parts.append(f"CPU governor={','.join(sorted(set(governors)))}")

        nvpmodel_mode = None
        if shutil.which("nvpmodel") is not None:
            try:
                proc = subprocess.run(
                    ["nvpmodel", "-q"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=1.0,
                )
                output = (proc.stdout or "") + (proc.stderr or "")
                for line in output.splitlines():
                    if "NV Power Mode:" in line:
                        nvpmodel_mode = line.split("NV Power Mode:", 1)[1].strip()
                        break
            except Exception:
                nvpmodel_mode = None
        if nvpmodel_mode and "MAXN" not in nvpmodel_mode.upper():
            hint_parts.append(f"nvpmodel={nvpmodel_mode}")

        if hint_parts:
            joined = " | ".join(hint_parts)
            print(
                "⚠️ Jetson 性能模式提示: "
                f"{joined}，推荐运行 `sudo nvpmodel -m 0 && sudo jetson_clocks`"
            )

    def _record_worker_perf(self, bucket_name: str, metrics: dict):
        if not self.perf_log:
            return
        with self._worker_perf_lock:
            bucket = self._worker_perf_buckets.get(bucket_name)
            if bucket is not None:
                bucket.add(metrics)

    @staticmethod
    def _copy_points(points):
        if points is None:
            return None
        arr = np.asarray(points, dtype=np.float32)
        return arr.copy()

    def _build_display_snapshot(self, frame: np.ndarray) -> dict:
        return {
            "frame": frame,
            "detection_bbox": tuple(self._latest_detection_bbox) if self._latest_detection_bbox is not None else None,
            "detection_conf": self._latest_detection_conf,
            "prediction_bbox": tuple(self._latest_prediction_bbox) if self._latest_prediction_bbox is not None else None,
            "detection_corners": self._copy_points(self._latest_detection_corners),
            "prediction_corners": self._copy_points(self._latest_prediction_corners),
            "fps_value": self._fps_value,
            "yolo_ms": self._last_yolo_ms,
            "pnp_ms": self._last_pnp_ms,
            "enable_prediction": self.enable_prediction,
            "pred_async": self.pred_async,
            "pred_lag": self._last_pred_lag,
            "buffer_len": self._last_buffer_len,
            "input_sequence_length": self.input_sequence_length,
            "pred_ready": self._pred_ready,
            "comp_distance_m": self._last_comp_distance_m,
            "raw_distance_m": self._last_raw_distance_m,
            "lead_frames": self._last_lead_frames,
            "comp_time_s": self._last_comp_time_s,
            "extra_time_s": self._last_extra_time_s,
            "lag_time_s": self._last_lag_time_s,
            "lag_frames": self._last_lag_frames,
            "enable_ec_feedback": self.enable_ec_feedback,
            "ec_feedback": self._get_ec_feedback(),
            "target_yaw_rate_dps": self._target_yaw_rate_effective_dps,
            "target_pitch_rate_dps": self._target_pitch_rate_effective_dps,
            "target_yaw_rate_fast_dps": self._target_yaw_rate_fast_dps,
            "target_pitch_rate_fast_dps": self._target_pitch_rate_fast_dps,
            "target_yaw_rate_slow_dps": self._target_yaw_rate_slow_dps,
            "target_pitch_rate_slow_dps": self._target_pitch_rate_slow_dps,
            "target_yaw_rate_effective_dps": self._target_yaw_rate_effective_dps,
            "target_pitch_rate_effective_dps": self._target_pitch_rate_effective_dps,
            "spin_active": self._spin_active,
            "spin_confidence": self._spin_confidence,
            "spin_image_time_comp_scale": self._last_image_time_comp_scale,
            "spin_yaw_direction_locked": self._spin_yaw_direction_locked,
            "spin_yaw_direction_score": self._spin_yaw_direction_score,
            "spin_yaw_fake_rate_dps": self._spin_yaw_fake_rate_dps,
            "last_spin_yaw_bias_deg": self._last_spin_yaw_bias_deg,
            "ec_t0_s": self.ec_t0_s,
            "detector_pre_ms": self._last_detector_pre_ms,
            "detector_infer_ms": self._last_detector_infer_ms,
            "detector_post_ms": self._last_detector_post_ms,
            "update_ms": self._last_update_ms,
            "det_angle_ms": self._last_det_angle_ms,
            "pred_ms": self._last_pred_ms,
            "det_corner_ms": self._last_det_corner_ms,
            "pred_corner_ms": self._last_pred_corner_ms,
            "loop_ms": self._last_loop_ms,
            "draw_ms": self._last_draw_ms,
            "present_ms": self._last_present_ms,
            "waitkey_ms": self._last_waitkey_ms,
            "serial_ms": self._last_serial_ms,
            "record_video": self.record_video,
            "record_ms": self._last_record_ms,
            "profile_pred": self.profile_pred,
            "pred_pre_ms": self._last_pred_pre_ms,
            "pred_h2d_ms": self._last_pred_h2d_ms,
            "pred_fwd_ms": self._last_pred_fwd_ms,
            "pred_d2h_ms": self._last_pred_d2h_ms,
        }

    def _render_overlay_snapshot(self, snapshot: dict) -> Tuple[np.ndarray, float]:
        draw_start = time.perf_counter()
        frame = snapshot["frame"]
        drawn = frame.copy()
        detection_bbox = snapshot["detection_bbox"]
        if detection_bbox is not None:
            x1, y1, x2, y2 = detection_bbox
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
            detection_conf = snapshot["detection_conf"]
            if detection_conf is not None:
                conf_text = f"conf: {float(detection_conf):.2f}"
                text_y = max(0, y1 - 10)
                cv2.putText(
                    drawn,
                    conf_text,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
        prediction_bbox = snapshot["prediction_bbox"]
        if prediction_bbox is not None:
            x1, y1, x2, y2 = prediction_bbox
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 0, 255), 2)
        detection_corners = snapshot["detection_corners"]
        if detection_corners is not None:
            pts = detection_corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(drawn, [pts], True, (0, 255, 255), 2)
            for x, y in pts.reshape(-1, 2):
                cv2.circle(drawn, (int(x), int(y)), 3, (0, 255, 255), -1)
        prediction_corners = snapshot["prediction_corners"]
        if prediction_corners is not None:
            pts = prediction_corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(drawn, [pts], True, (255, 0, 255), 2)
            for x, y in pts.reshape(-1, 2):
                cv2.circle(drawn, (int(x), int(y)), 3, (255, 0, 255), -1)

        cv2.putText(drawn, f"FPS: {snapshot['fps_value']:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(drawn, f"YOLO: {snapshot['yolo_ms']:.1f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(drawn, f"PnP: {snapshot['pnp_ms']:.1f} ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if not snapshot["enable_prediction"]:
            pred_status = "OFF"
        elif snapshot["pred_async"]:
            lag = snapshot["pred_lag"]
            pred_status = f"ASYNC lag={'NA' if lag is None else lag}"
        else:
            pred_status = (
                f"{snapshot['buffer_len']}/{snapshot['input_sequence_length']} "
                f"{'READY' if snapshot['pred_ready'] else 'WAIT'}"
            )
        cv2.putText(drawn, f"Pred: {pred_status}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        comp_distance_m = snapshot["comp_distance_m"]
        raw_distance_m = snapshot["raw_distance_m"]
        if comp_distance_m is not None:
            dist_text = f"Dist: {comp_distance_m:.2f} m"
            if raw_distance_m is not None and raw_distance_m > comp_distance_m + 1e-3:
                dist_text += " (cap)"
        else:
            dist_text = "Dist: NA"
        cv2.putText(drawn, dist_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        lead_text = f"Lead: {snapshot['lead_frames']} t={snapshot['comp_time_s']:.3f}s"
        if snapshot["extra_time_s"] > 1e-3:
            lead_text += f" extra={snapshot['extra_time_s']:.3f}s"
        if snapshot["lag_time_s"] > 1e-3:
            lead_text += f" lag={snapshot['lag_frames']}f/{snapshot['lag_time_s'] * 1000.0:.1f}ms"
        cv2.putText(drawn, lead_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if snapshot["enable_ec_feedback"]:
            ec = snapshot["ec_feedback"]
            if ec is not None and abs(time.time() - float(ec[-1])) <= 0.2:
                _, _, gimbal_yaw_rate, gimbal_pitch_rate, _, _, _ = ec
                ec_text = (
                    f"EC: gYawV={float(gimbal_yaw_rate):.1f} "
                    f"tYawV={snapshot['target_yaw_rate_effective_dps']:.1f} "
                    f"gPitV={float(gimbal_pitch_rate):.1f} "
                    f"tPitV={snapshot['target_pitch_rate_effective_dps']:.1f} "
                    f"t0={snapshot['ec_t0_s'] * 1000.0:.0f}ms"
                )
                cv2.putText(drawn, ec_text, (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        spin_text = (
            f"SPIN:{'ON' if snapshot['spin_active'] else 'OFF'} "
            f"conf={snapshot['spin_confidence']:.2f} imgScale={snapshot['spin_image_time_comp_scale']:.2f} "
            f"dir={snapshot['spin_yaw_direction_locked']:+d}/{snapshot['spin_yaw_direction_score']:+.1f} "
            f"bias={snapshot['last_spin_yaw_bias_deg']:+.1f} "
            f"yawV={snapshot['target_yaw_rate_fast_dps']:.1f}/"
            f"{snapshot['target_yaw_rate_slow_dps']:.1f}/"
            f"{snapshot['target_yaw_rate_effective_dps']:.1f} "
            f"fake={snapshot['spin_yaw_fake_rate_dps']:.1f}"
        )
        cv2.putText(drawn, spin_text, (10, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        spin_pitch_text = (
            f"SPIN PitV={snapshot['target_pitch_rate_fast_dps']:.1f}/"
            f"{snapshot['target_pitch_rate_slow_dps']:.1f}/"
            f"{snapshot['target_pitch_rate_effective_dps']:.1f}"
        )
        cv2.putText(drawn, spin_pitch_text, (10, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        timing_text = (
            f"Pre:{snapshot['detector_pre_ms']:.1f} "
            f"Infer:{snapshot['detector_infer_ms']:.1f} "
            f"Post:{snapshot['detector_post_ms']:.1f}"
        )
        cv2.putText(drawn, timing_text, (10, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        timing_text = (
            f"Upd:{snapshot['update_ms']:.1f} "
            f"Det:{snapshot['det_angle_ms']:.1f} "
            f"Pred:{snapshot['pred_ms']:.1f} "
            f"C:{snapshot['det_corner_ms'] + snapshot['pred_corner_ms']:.1f}"
        )
        cv2.putText(drawn, timing_text, (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        io_text = (
            f"Loop:{snapshot['loop_ms']:.1f} "
            f"Draw:{snapshot['draw_ms']:.1f} "
            f"Disp:{snapshot['present_ms']:.1f} "
            f"Wait:{snapshot['waitkey_ms']:.1f} "
            f"Serial:{snapshot['serial_ms']:.1f}"
        )
        cv2.putText(drawn, io_text, (10, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        record_text = "Rec: OFF"
        if snapshot["record_video"]:
            record_text = f"Rec: ON {snapshot['record_ms']:.1f} ms"
        cv2.putText(drawn, record_text, (10, 338), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        if snapshot["profile_pred"]:
            pred_text = (
                f"Ppre:{snapshot['pred_pre_ms']:.1f} "
                f"H2D:{snapshot['pred_h2d_ms']:.1f} "
                f"FWD:{snapshot['pred_fwd_ms']:.1f} "
                f"D2H:{snapshot['pred_d2h_ms']:.1f}"
            )
            cv2.putText(drawn, pred_text, (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        return drawn, (time.perf_counter() - draw_start) * 1000.0

    def _publish_display_snapshot(self, frame: np.ndarray):
        snapshot = self._build_display_snapshot(frame)
        with self._display_lock:
            if self._display_snapshot is not None:
                self._display_dropped_pending += 1
            self._display_snapshot = snapshot
            self._display_version += 1

    def _publish_send_snapshot(self):
        snapshot = {
            "packet": bytes(self._latest_prediction_packet),
            "info": self._latest_prediction_info,
            "flight_time_s": self._last_flight_time_s,
            "comp_distance_m": self._last_comp_distance_m,
            "lead_frames": self._last_lead_frames,
            "comp_time_s": self._last_comp_time_s,
            "lag_frames": self._last_lag_frames,
            "lag_time_s": self._last_lag_time_s,
            "fire_reason": self._last_fire_reason,
            "spin_active": self._spin_active,
            "spin_confidence": self._spin_confidence,
            "spin_image_time_comp_scale": self._last_image_time_comp_scale,
            "spin_yaw_direction_locked": self._spin_yaw_direction_locked,
            "spin_yaw_direction_score": self._spin_yaw_direction_score,
            "spin_yaw_fake_rate_dps": self._spin_yaw_fake_rate_dps,
            "last_spin_yaw_bias_deg": self._last_spin_yaw_bias_deg,
            "target_yaw_rate_fast_dps": self._target_yaw_rate_fast_dps,
            "target_yaw_rate_slow_dps": self._target_yaw_rate_slow_dps,
            "target_yaw_rate_effective_dps": self._target_yaw_rate_effective_dps,
        }
        with self._send_lock:
            self._send_snapshot = snapshot
            self._send_version += 1

    def _start_display_worker(self):
        if not self.show_window or self._display_thread is not None:
            return
        self._display_stop_event.clear()
        self._display_thread = threading.Thread(
            target=self._display_worker_loop,
            name="display-worker",
            daemon=True,
        )
        self._display_thread.start()

    def _stop_display_worker(self):
        if self._display_thread is None:
            return
        self._display_stop_event.set()
        self._stop_requested.set()
        self._display_thread.join(timeout=1.0)
        self._display_thread = None

    def _start_send_worker(self):
        if self._send_thread is not None:
            return
        self._send_stop_event.clear()
        self._send_thread = threading.Thread(
            target=self._send_worker_loop,
            name="send-worker",
            daemon=True,
        )
        self._send_thread.start()

    def _stop_send_worker(self):
        if self._send_thread is None:
            return
        self._send_stop_event.set()
        self._send_thread.join(timeout=1.0)
        self._send_thread = None

    def _display_worker_loop(self):
        frame_interval = 1.0 / self.display_max_fps
        next_display_ts = 0.0
        last_display_ts = None
        last_seen_version = -1
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            while not self._display_stop_event.is_set() and not self._stop_requested.is_set():
                now = time.perf_counter()
                if now < next_display_ts:
                    time.sleep(min(0.001, next_display_ts - now))
                    continue

                with self._display_lock:
                    version = self._display_version
                    if self._display_snapshot is None or version == last_seen_version:
                        snapshot = None
                        dropped = 0
                    else:
                        snapshot = self._display_snapshot
                        self._display_snapshot = None
                        dropped = self._display_dropped_pending
                        self._display_dropped_pending = 0
                        last_seen_version = version
                if snapshot is None:
                    time.sleep(0.001)
                    continue

                view, draw_ms = self._render_overlay_snapshot(snapshot)
                present_start = time.perf_counter()
                cv2.imshow(self.window_name, view)
                present_ms = (time.perf_counter() - present_start) * 1000.0
                wait_start = time.perf_counter()
                key = cv2.waitKey(1)
                wait_ms = (time.perf_counter() - wait_start) * 1000.0
                total_ms = draw_ms + present_ms + wait_ms
                self._last_draw_ms = draw_ms
                self._last_present_ms = present_ms
                self._last_waitkey_ms = wait_ms
                self._last_imshow_ms = present_ms + wait_ms

                if last_display_ts is None:
                    display_fps = 0.0
                else:
                    dt = max(1e-6, time.perf_counter() - last_display_ts)
                    display_fps = 1.0 / dt
                last_display_ts = time.perf_counter()
                self._record_worker_perf(
                    "display",
                    {
                        "draw_ms": draw_ms,
                        "present_ms": present_ms,
                        "wait_ms": wait_ms,
                        "display_total_ms": total_ms,
                        "display_fps": display_fps if display_fps > 0.0 else None,
                        "display_drop_count": float(dropped),
                    },
                )
                next_display_ts = time.perf_counter() + frame_interval
                if key & 0xFF == ord("q"):
                    self._stop_requested.set()
                    break
        finally:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass

    def _send_worker_loop(self):
        send_interval = 1.0 / self.send_rate
        next_send_ts = time.perf_counter()
        last_sent_version = 0
        last_send_ts = None
        while not self._send_stop_event.is_set() and not self._stop_requested.is_set():
            before_sleep = time.perf_counter()
            sleep_ms = 0.0
            if before_sleep < next_send_ts:
                time.sleep(max(0.0, next_send_ts - before_sleep))
                sleep_ms = (time.perf_counter() - before_sleep) * 1000.0

            with self._send_lock:
                snapshot = self._send_snapshot
                version = self._send_version

            dropped_updates = max(0, version - last_sent_version - 1)
            last_sent_version = version
            send_ms = 0.0
            now = time.time()
            serial_open = self.serial is not None and getattr(self.serial, "is_open", False)
            if snapshot is not None and serial_open:
                try:
                    send_start = time.perf_counter()
                    sent = self.serial.write(snapshot["packet"])
                    send_ms = (time.perf_counter() - send_start) * 1000.0
                    self._last_serial_ms = send_ms
                    with self._tx_lock:
                        self._tx_total_count += 1
                        self._tx_total_bytes += int(sent)
                    if self.show_tx:
                        pred_hex = " ".join(f"{b:02X}" for b in snapshot["packet"])
                        info = snapshot["info"]
                        if info is not None:
                            pred_yaw, pred_pitch, pred_x, pred_y, pred_src = info
                            if snapshot["flight_time_s"] is not None:
                                t1_text = f"{snapshot['flight_time_s']:.3f}s"
                            elif snapshot["comp_distance_m"] is not None:
                                t1_text = f"{snapshot['comp_distance_m'] / self.bullet_speed_mps:.3f}s"
                            else:
                                t1_text = "NA"
                            t2_text = f"{self.system_latency_s:.3f}s"
                            fire_cmd = int(snapshot["packet"][2]) if len(snapshot["packet"]) >= 3 else 0
                            print(
                                f"TX[PRED]: src={pred_src} yaw={pred_yaw:.2f} pitch={pred_pitch:.2f} "
                                f"x={pred_x:.1f} y={pred_y:.1f} t1={t1_text} t2={t2_text} "
                                f"lead={snapshot['lead_frames']} t={snapshot['comp_time_s']:.3f}s "
                                f"lag={snapshot['lag_frames']}/{snapshot['lag_time_s'] * 1000.0:.1f}ms "
                                f"fire={fire_cmd} reason={snapshot['fire_reason']} "
                                f"spin={'ON' if snapshot['spin_active'] else 'OFF'}/{snapshot['spin_confidence']:.2f} "
                                f"imgScale={snapshot['spin_image_time_comp_scale']:.2f} "
                                f"dir={snapshot['spin_yaw_direction_locked']:+d}/{snapshot['spin_yaw_direction_score']:+.1f} "
                                f"fakeYawV={snapshot['spin_yaw_fake_rate_dps']:.1f} "
                                f"yawBias={snapshot['last_spin_yaw_bias_deg']:+.1f} "
                                f"ecYawV={snapshot['target_yaw_rate_fast_dps']:.1f}/"
                                f"{snapshot['target_yaw_rate_slow_dps']:.1f}/"
                                f"{snapshot['target_yaw_rate_effective_dps']:.1f} "
                                f"| {pred_hex}"
                            )
                        else:
                            print(f"TX[PRED]: {pred_hex}")
                except Exception as exc:
                    print(f"❌ 串口发送失败: {exc}")
            elif now - self._send_last_warn_time >= 5.0:
                print("⚠️ 串口未打开，跳过发送")
                self._send_last_warn_time = now

            loop_now = time.perf_counter()
            send_fps = 0.0
            if last_send_ts is not None:
                send_fps = 1.0 / max(1e-6, loop_now - last_send_ts)
            last_send_ts = loop_now
            self._record_worker_perf(
                "send",
                {
                    "send_ms": send_ms if serial_open else None,
                    "send_sleep_ms": sleep_ms,
                    "send_fps": send_fps if send_fps > 0.0 else None,
                    "send_drop_count": float(dropped_updates),
                },
            )
            next_send_ts = loop_now + send_interval

    def _load_coordinate_model(self, model_path: str) -> CoordinatePredictionModel:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"坐标预测模型文件不存在: {model_path}")
        model = CoordinatePredictionModel(
            input_size=64 * 64,
            hidden_size=128,
            num_lstm_layers=2,
            dropout=0.1,
            coordinate_dim=4,
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        load_msg = model.load_state_dict(state_dict, strict=False)
        if (
            getattr(load_msg, "missing_keys", None)
            or getattr(load_msg, "unexpected_keys", None)
        ):
            miss = len(getattr(load_msg, "missing_keys", []))
            unexp = len(getattr(load_msg, "unexpected_keys", []))
            print(
                f"⚠️ 坐标模型权重部分不匹配: missing={miss}, unexpected={unexp} "
                "(已按strict=False加载)"
            )
        model.to(self.device)
        model.eval()
        print(f"✅ 坐标预测模型加载成功: {model_path}")
        return model

    def _open_camera(self) -> bool:
        if self.use_daheng:
            if CameraAPI is None:
                print("❌ camerapytest.CameraAPI 未能导入，无法使用大恒摄像头")
                return False
            try:
                self.camera = CameraAPI(
                    config_path=self.daheng_config,
                    device_sn=self.daheng_sn,
                    allow_opencv_fallback=False,
                )
                if not self.camera.open():
                    print("❌ 无法打开大恒摄像头")
                    return False
                self.camera.start()
                time.sleep(0.3)
                return True
            except Exception as exc:
                print(f"❌ 初始化大恒摄像头失败: {exc}")
                return False

        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"❌ 无法打开摄像头 {self.camera_id}")
            return False
        return True

    def _read_frame(self):
        if self.use_daheng and self.camera is not None:
            frame = self.camera.read()
            return frame
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def _open_serial(self) -> bool:
        try:
            self.serial = serial.Serial(
                self.uart_port,
                self.uart_baud,
                timeout=SERIAL_TIMEOUT_DEFAULT,
            )
        except Exception as exc:
            print(f"❌ 串口打开失败: {exc}")
            return False
        if not self.serial.is_open:
            print("❌ 串口未打开")
            return False
        print(f"✅ 串口打开成功: {self.uart_port} @ {self.uart_baud}")
        return True

    def _close_resources(self):
        if self.cap is not None:
            self.cap.release()
        if self.camera is not None:
            self.camera.stop()
        self._close_record_writer()
        if self.serial is not None:
            try:
                self.serial.close()
                print("✅ 串口已关闭")
            except Exception:
                pass

    @staticmethod
    def _uint_to_float(x_uint: int, x_min: float, x_max: float, bits: int) -> float:
        if bits <= 0:
            return float(x_min)
        span = float(x_max) - float(x_min)
        max_int = float((1 << int(bits)) - 1)
        if max_int <= 0.0:
            return float(x_min)
        return float(x_uint) * span / max_int + float(x_min)

    def _start_ec_feedback_worker(self):
        if not self.enable_ec_feedback:
            return
        if self._ec_thread is not None and self._ec_thread.is_alive():
            return
        self._ec_stop_event.clear()
        self._ec_thread = threading.Thread(target=self._ec_feedback_worker, daemon=True)
        self._ec_thread.start()

    def _stop_ec_feedback_worker(self):
        if self._ec_thread is None:
            return
        self._ec_stop_event.set()
        try:
            self._ec_thread.join(timeout=0.5)
        except Exception:
            pass
        self._ec_thread = None

    def _get_ec_feedback(self):
        with self._ec_lock:
            return self._ec_latest

    def _ec_feedback_worker(self):
        buf = bytearray()
        while not self._ec_stop_event.is_set():
            ser = self.serial
            if ser is None or not getattr(ser, "is_open", False):
                time.sleep(0.05)
                continue

            try:
                data = ser.read(64)
            except Exception:
                time.sleep(0.05)
                continue

            if not data:
                time.sleep(0.001)
                continue

            buf.extend(data)
            while len(buf) >= 8:
                sof_idx = buf.find(b"\xAA")
                if sof_idx < 0:
                    buf.clear()
                    break
                if sof_idx > 0:
                    del buf[:sof_idx]
                if len(buf) < 8:
                    break
                if buf[7] != 0xBB:
                    del buf[0:1]
                    continue

                frame = bytes(buf[:8])
                del buf[:8]

                mode = int(frame[1])
                yaw_u = int(frame[2] | (frame[3] << 8))
                pitch_u = int(frame[4] | (frame[5] << 8))
                shoot = int(frame[6])

                yaw_deg = self._uint_to_float(yaw_u, -180.0, 180.0, 16)
                pitch_deg = self._uint_to_float(pitch_u, -180.0, 180.0, 16)
                if self.ec_feedback_invert_yaw:
                    yaw_deg = -yaw_deg
                if self.ec_feedback_invert_pitch:
                    pitch_deg = -pitch_deg

                now = time.time()
                yaw_rate_dps = 0.0
                pitch_rate_dps = 0.0
                with self._ec_lock:
                    if self._ec_last_ts is not None:
                        dt = max(1e-3, now - self._ec_last_ts)
                        if self._ec_last_yaw_wrapped is not None:
                            dyaw = yaw_deg - self._ec_last_yaw_wrapped
                            if dyaw > 180.0:
                                dyaw -= 360.0
                            elif dyaw < -180.0:
                                dyaw += 360.0
                            yaw_rate_dps = dyaw / dt
                            if self._ec_last_yaw_unwrapped is None:
                                self._ec_last_yaw_unwrapped = yaw_deg
                            else:
                                self._ec_last_yaw_unwrapped += dyaw
                        if self._ec_last_pitch_wrapped is not None:
                            dpitch = pitch_deg - self._ec_last_pitch_wrapped
                            if dpitch > 180.0:
                                dpitch -= 360.0
                            elif dpitch < -180.0:
                                dpitch += 360.0
                            pitch_rate_dps = dpitch / dt

                    self._ec_last_ts = now
                    self._ec_last_yaw_wrapped = yaw_deg
                    self._ec_last_pitch_wrapped = pitch_deg
                    if self._ec_last_yaw_unwrapped is None:
                        self._ec_last_yaw_unwrapped = yaw_deg

                    self._ec_latest = (
                        float(yaw_deg),
                        float(pitch_deg),
                        float(yaw_rate_dps),
                        float(pitch_rate_dps),
                        mode,
                        shoot,
                        float(now),
                    )

    def _get_intrinsics_for_frame(
        self, frame_shape: Tuple[int, int, int]
    ):
        if not self.scale_intrinsics:
            return self._intrinsics_base
        frame_h, frame_w = frame_shape[:2]
        if self._intrinsics_scaled is not None and self._intrinsics_scaled_shape == (frame_w, frame_h):
            return self._intrinsics_scaled
        scaled = scale_intrinsics_to_frame(self._intrinsics_base, frame_shape)
        self._intrinsics_scaled = scaled
        self._intrinsics_scaled_shape = (frame_w, frame_h)
        return scaled

    def _pixel_to_angle(
        self, x_center_px: float, y_center_px: float, frame_shape: Tuple[int, int, int]
    ) -> Tuple[float, float]:
        height, width = frame_shape[:2]
        intrinsics = self._get_intrinsics_for_frame(frame_shape)
        fx = float(intrinsics.matrix[0, 0])
        fy = float(intrinsics.matrix[1, 1])
        cx = float(intrinsics.matrix[0, 2])
        cy = float(intrinsics.matrix[1, 2])
        if not (0 < cx < width):
            cx = width / 2.0
        if not (0 < cy < height):
            cy = height / 2.0
        dx = x_center_px - cx
        dy = y_center_px - cy
        yaw_deg = float(np.degrees(np.arctan2(dx, fx)))
        pitch_deg = float(np.degrees(np.arctan2(-dy, fy)))
        return yaw_deg, pitch_deg

    def _get_roi_bbox(
        self, frame_shape: Tuple[int, int, int], bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        pad = self.context_padding
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad)
        y2 = min(height, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return (0, 0, width, height)
        return (x1, y1, x2, y2)

    def _preprocess_frame(self, frame: np.ndarray, roi_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = roi_bbox
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
        else:
            roi = frame
        if roi.ndim == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(roi, self._roi_size)
        return resized.astype(np.float32) / 255.0

    def _convert_roi_prediction_to_frame(
        self,
        predicted_coords: np.ndarray,
        roi_bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = roi_bbox
        roi_w = max(1, x2 - x1)
        roi_h = max(1, y2 - y1)
        x_center, y_center, w, h = predicted_coords
        x_center_global = (x1 + x_center * roi_w) / width
        y_center_global = (y1 + y_center * roi_h) / height
        w_global = (w * roi_w) / width
        h_global = (h * roi_h) / height
        return np.array(
            [x_center_global, y_center_global, w_global, h_global], dtype=np.float32
        )

    def _reset_motion_state(self):
        self._motion_history.clear()
        self._spin_confidence = 0.0
        self._spin_active = False
        self._spin_yaw_direction_score = 0.0
        self._spin_yaw_direction_locked = 0
        self._spin_yaw_fake_rate_dps = 0.0
        self._last_spin_yaw_bias_deg = 0.0
        self._last_valid_motion_rvec_ts = None
        self._last_rel_angles_for_rate = None
        self._last_rel_angle_time_for_rate = None
        self._target_yaw_rate_dps = 0.0
        self._target_pitch_rate_dps = 0.0
        self._target_yaw_rate_fast_dps = 0.0
        self._target_pitch_rate_fast_dps = 0.0
        self._target_yaw_rate_slow_dps = 0.0
        self._target_pitch_rate_slow_dps = 0.0
        self._target_yaw_rate_effective_dps = 0.0
        self._target_pitch_rate_effective_dps = 0.0
        self._last_distance_m = None
        self._last_pnp_tvec = None
        self._last_pnp_rvec = None
        self._last_image_time_comp_scale = 1.0

    @staticmethod
    def _angle_delta_deg(current: float, previous: float) -> float:
        delta = float(current) - float(previous)
        while delta > 180.0:
            delta -= 360.0
        while delta < -180.0:
            delta += 360.0
        return delta

    @staticmethod
    def _apply_ema(last_value: float, new_value: float, alpha: float) -> float:
        return (1.0 - alpha) * float(last_value) + alpha * float(new_value)

    def _compute_normal_yaw_deg(self, rvec: Optional[np.ndarray]) -> Optional[float]:
        if rvec is None:
            return None
        try:
            rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float32))
        except Exception:
            return None
        normal = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        yaw_deg = float(np.degrees(np.arctan2(float(normal[0]), float(normal[2]))))
        if self.invert_yaw:
            yaw_deg = -yaw_deg
        return yaw_deg

    @staticmethod
    def _compute_quad_area_px(corners: Optional[np.ndarray]) -> Optional[float]:
        if corners is None:
            return None
        pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
            return None
        x = pts[:, 0]
        y = pts[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @staticmethod
    def _compute_quad_aspect(corners: Optional[np.ndarray]) -> Optional[float]:
        if corners is None:
            return None
        pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
            return None
        top = float(np.linalg.norm(pts[1] - pts[0]))
        right = float(np.linalg.norm(pts[2] - pts[1]))
        bottom = float(np.linalg.norm(pts[2] - pts[3]))
        left = float(np.linalg.norm(pts[3] - pts[0]))
        width = 0.5 * (top + bottom)
        height = 0.5 * (left + right)
        if height <= 1e-6:
            return None
        return float(width / height)

    def _append_motion_observation(self, observation: MotionObservation):
        self._motion_history.append(observation)
        cutoff = float(observation.ts) - 0.18
        while self._motion_history and self._motion_history[0].ts < cutoff:
            self._motion_history.popleft()
        if observation.normal_yaw_deg is not None:
            self._last_valid_motion_rvec_ts = float(observation.ts)

    def _refresh_spin_state(self, now_s: float):
        if not self.spin_aware:
            self._spin_confidence = 0.0
            self._spin_active = False
            return

        history = list(self._motion_history)
        valid_normal = [obs for obs in history if obs.normal_yaw_deg is not None]
        valid_quad = [obs for obs in history if obs.quad_aspect is not None and obs.quad_area_px is not None]
        if len(history) < 4 or len(valid_normal) < 3 or len(valid_quad) < 3:
            if (
                self._last_valid_motion_rvec_ts is None
                or (float(now_s) - float(self._last_valid_motion_rvec_ts)) > 0.2
            ):
                self._spin_confidence *= 0.5
            self._spin_confidence = float(np.clip(self._spin_confidence, 0.0, 1.0))
            if self._spin_active and self._spin_confidence < self.spin_exit_threshold:
                self._spin_active = False
            return

        normal_rates = []
        quad_aspect_rates = []
        prev_quad_obs = None
        prev_normal_obs = None
        for obs in history:
            if obs.quad_aspect is not None:
                if prev_quad_obs is not None and prev_quad_obs.quad_aspect is not None:
                    dt = max(1e-3, float(obs.ts) - float(prev_quad_obs.ts))
                    quad_aspect_rates.append(
                        abs(float(obs.quad_aspect) - float(prev_quad_obs.quad_aspect)) / dt
                    )
                prev_quad_obs = obs
            if obs.normal_yaw_deg is None:
                continue
            if prev_normal_obs is not None and prev_normal_obs.normal_yaw_deg is not None:
                dt = max(1e-3, float(obs.ts) - float(prev_normal_obs.ts))
                normal_rates.append(
                    abs(
                        self._angle_delta_deg(
                            float(obs.normal_yaw_deg),
                            float(prev_normal_obs.normal_yaw_deg),
                        )
                    ) / dt
                )
            prev_normal_obs = obs

        mean_abs_normal_yaw_rate = float(np.mean(normal_rates)) if normal_rates else 0.0
        mean_abs_quad_aspect_rate = float(np.mean(quad_aspect_rates)) if quad_aspect_rates else 0.0

        distance_values = [
            float(obs.distance_m)
            for obs in history
            if obs.distance_m is not None and float(obs.distance_m) > 1e-6
        ]
        distance_cv = None
        if len(distance_values) >= 3:
            mean_distance = float(np.mean(distance_values))
            if mean_distance > 1e-6:
                distance_cv = float(np.std(distance_values)) / mean_distance

        area_values = [
            float(obs.quad_area_px)
            for obs in history
            if obs.quad_area_px is not None and float(obs.quad_area_px) > 1e-6
        ]
        area_cv = None
        if len(area_values) >= 3:
            mean_area = float(np.mean(area_values))
            if mean_area > 1e-6:
                area_cv = float(np.std(area_values)) / mean_area

        rate_score = float(np.clip((mean_abs_normal_yaw_rate - 90.0) / 90.0, 0.0, 1.0))
        aspect_score = float(np.clip((mean_abs_quad_aspect_rate - 1.2) / 1.2, 0.0, 1.0))
        stable_score = 0.0
        if rate_score > 0.0 or aspect_score > 0.0:
            if distance_cv is not None and distance_cv <= 0.06:
                stable_score += 0.5
            if area_cv is not None and area_cv <= 0.12:
                stable_score += 0.5

        self._spin_confidence = float(
            np.clip(0.45 * rate_score + 0.25 * aspect_score + 0.30 * stable_score, 0.0, 1.0)
        )
        if not self._spin_active and self._spin_confidence >= self.spin_enter_threshold:
            self._spin_active = True
        elif self._spin_active and self._spin_confidence < self.spin_exit_threshold:
            self._spin_active = False

    def _update_effective_target_rates(self, target_yaw_rate: float, target_pitch_rate: float):
        if abs(float(target_yaw_rate)) > 720.0:
            target_yaw_rate = self._target_yaw_rate_effective_dps
        if abs(float(target_pitch_rate)) > 720.0:
            target_pitch_rate = self._target_pitch_rate_effective_dps

        self._target_yaw_rate_fast_dps = self._apply_ema(
            self._target_yaw_rate_fast_dps,
            target_yaw_rate,
            self._target_rate_fast_alpha,
        )
        self._target_pitch_rate_fast_dps = self._apply_ema(
            self._target_pitch_rate_fast_dps,
            target_pitch_rate,
            self._target_rate_fast_alpha,
        )
        self._target_yaw_rate_slow_dps = self._apply_ema(
            self._target_yaw_rate_slow_dps,
            target_yaw_rate,
            self._target_rate_slow_alpha,
        )
        self._target_pitch_rate_slow_dps = self._apply_ema(
            self._target_pitch_rate_slow_dps,
            target_pitch_rate,
            self._target_rate_slow_alpha,
        )

        spin_weight = float(np.clip(self._spin_confidence, 0.0, 1.0)) if self.spin_aware else 0.0
        self._target_yaw_rate_effective_dps = self._target_yaw_rate_slow_dps + (
            1.0 - spin_weight
        ) * (self._target_yaw_rate_fast_dps - self._target_yaw_rate_slow_dps)
        self._target_pitch_rate_effective_dps = self._target_pitch_rate_slow_dps + (
            1.0 - spin_weight
        ) * (self._target_pitch_rate_fast_dps - self._target_pitch_rate_slow_dps)
        self._target_yaw_rate_dps = self._target_yaw_rate_effective_dps
        self._target_pitch_rate_dps = self._target_pitch_rate_effective_dps
        self._update_spin_yaw_direction_lock()

    def _update_spin_yaw_direction_lock(self):
        self._spin_yaw_fake_rate_dps = float(
            self._target_yaw_rate_fast_dps - self._target_yaw_rate_slow_dps
        )
        self._spin_yaw_direction_score *= 0.97
        if not self.spin_aware or not self._spin_active:
            return
        if self._spin_confidence < self.spin_yaw_dir_lock_min_conf:
            return

        fake_rate = float(self._spin_yaw_fake_rate_dps)
        if abs(fake_rate) < self.spin_yaw_dir_min_rate_dps:
            return

        vote = 1.0 if fake_rate > 0.0 else -1.0
        weight = min(
            abs(fake_rate) / max(self.spin_yaw_dir_min_rate_dps * 2.0, 1e-3),
            1.0,
        )
        self._spin_yaw_direction_score = float(
            np.clip(
                self._spin_yaw_direction_score + vote * weight,
                -self.spin_yaw_dir_switch_threshold,
                self.spin_yaw_dir_switch_threshold,
            )
        )

        if self._spin_yaw_direction_locked == 0:
            if abs(self._spin_yaw_direction_score) >= self.spin_yaw_dir_lock_threshold:
                self._spin_yaw_direction_locked = 1 if self._spin_yaw_direction_score > 0.0 else -1
            return

        if self._spin_confidence < self.spin_yaw_dir_switch_min_conf:
            return
        if abs(fake_rate) < self.spin_yaw_dir_switch_min_rate_dps:
            return

        if (
            self._spin_yaw_direction_locked > 0
            and self._spin_yaw_direction_score <= -self.spin_yaw_dir_switch_threshold
        ):
            self._spin_yaw_direction_locked = -1
        elif (
            self._spin_yaw_direction_locked < 0
            and self._spin_yaw_direction_score >= self.spin_yaw_dir_switch_threshold
        ):
            self._spin_yaw_direction_locked = 1 if self._spin_yaw_direction_score > 0.0 else -1

    def _apply_spin_yaw_bias(
        self, yaw_deg: float, pitch_deg: float
    ) -> Tuple[float, float, Optional[str]]:
        self._last_spin_yaw_bias_deg = 0.0
        if not self.spin_aware or not self._spin_active:
            return yaw_deg, pitch_deg, None
        if self._spin_yaw_direction_locked == 0 or self.spin_yaw_reverse_bias_deg <= 1e-6:
            return yaw_deg, pitch_deg, None

        bias = -float(self._spin_yaw_direction_locked) * float(self.spin_yaw_reverse_bias_deg)
        self._last_spin_yaw_bias_deg = bias
        return yaw_deg + bias, pitch_deg, f"+SPIN({bias:+.1f})"

    def _update_motion_state(
        self,
        rel_yaw_deg: float,
        rel_pitch_deg: float,
        bbox: Optional[Tuple[int, int, int, int]],
        corners: Optional[np.ndarray],
        now_s: float,
    ):
        quad_area_px = self._compute_quad_area_px(corners)
        quad_aspect = self._compute_quad_aspect(corners)
        distance_m = self._last_distance_m if self._last_pnp_tvec is not None else None
        normal_yaw_deg = None
        if quad_area_px is not None and quad_aspect is not None:
            normal_yaw_deg = self._compute_normal_yaw_deg(self._last_pnp_rvec)
        observation = MotionObservation(
            ts=float(now_s),
            yaw_deg=float(rel_yaw_deg),
            pitch_deg=float(rel_pitch_deg),
            quad_area_px=quad_area_px,
            quad_aspect=quad_aspect,
            distance_m=float(distance_m) if distance_m is not None else None,
            normal_yaw_deg=normal_yaw_deg,
        )
        self._append_motion_observation(observation)
        self._refresh_spin_state(now_s)

        if self._last_rel_angles_for_rate is None or self._last_rel_angle_time_for_rate is None:
            self._last_rel_angles_for_rate = (float(rel_yaw_deg), float(rel_pitch_deg))
            self._last_rel_angle_time_for_rate = float(now_s)
            return

        dt = max(1e-3, float(now_s) - float(self._last_rel_angle_time_for_rate))
        last_yaw, last_pitch = self._last_rel_angles_for_rate
        rel_yaw_dot = (float(rel_yaw_deg) - float(last_yaw)) / dt
        rel_pitch_dot = (float(rel_pitch_deg) - float(last_pitch)) / dt
        self._last_rel_angles_for_rate = (float(rel_yaw_deg), float(rel_pitch_deg))
        self._last_rel_angle_time_for_rate = float(now_s)

        target_yaw_rate = rel_yaw_dot
        target_pitch_rate = rel_pitch_dot
        if self.enable_ec_feedback:
            ec = self._get_ec_feedback()
            if ec is not None:
                _, _, gimbal_yaw_rate, gimbal_pitch_rate, _, _, ts = ec
                if abs(float(now_s) - float(ts)) <= 0.2:
                    target_yaw_rate = rel_yaw_dot + float(gimbal_yaw_rate)
                    target_pitch_rate = rel_pitch_dot + float(gimbal_pitch_rate)

        self._update_effective_target_rates(target_yaw_rate, target_pitch_rate)

    def _reset_prediction_buffers(self):
        self._buffer_idx = 0
        self._buffer_len = 0
        self.coord_buffer.fill(0.0)
        self._last_buffer_len = 0
        self._pred_ready = False
        self._last_pred_center = None
        self._last_pred_size = None
        self._last_pred_time = None
        self._pred_velocity = (0.0, 0.0)
        self._last_pred_lag = None
        self._reset_motion_state()

    def _update_frame_period_estimate(self, now_s: float):
        if self._last_frame_ts is None:
            self._last_frame_ts = now_s
            return
        dt = now_s - self._last_frame_ts
        self._last_frame_ts = now_s
        if dt <= 1e-4 or dt > 0.5:
            return
        alpha = self._frame_period_alpha
        self._frame_period_ema_s = (
            (1.0 - alpha) * self._frame_period_ema_s + alpha * dt
        )

    def _update_detector_runtime_stats(self):
        stats = self.detector.get_last_runtime_stats()
        self._last_detector_pre_ms = float(stats.get("preprocess_ms", 0.0))
        self._last_detector_infer_ms = float(stats.get("inference_ms", 0.0))
        self._last_detector_post_ms = float(stats.get("postprocess_ms", 0.0))
        self._last_detector_output_count = int(stats.get("output_count", 0))
        self._last_detector_raw_candidates = int(stats.get("raw_candidates", 0))
        self._last_detector_obj_candidates = int(stats.get("obj_candidates", 0))
        self._last_detector_class_candidates = int(stats.get("class_candidates", 0))
        self._last_detector_kept_candidates = int(stats.get("kept_candidates", 0))

    def _collect_perf_metrics(self) -> dict:
        metrics = {
            "read_ms": self._last_read_ms,
            "det_total_ms": self._last_yolo_ms,
            "det_pre_ms": self._last_detector_pre_ms,
            "det_infer_ms": self._last_detector_infer_ms,
            "det_post_ms": self._last_detector_post_ms,
            "det_select_ms": self._last_select_ms,
            "update_ms": self._last_update_ms,
            "det_angle_ms": self._last_det_angle_ms,
            "det_ballistic_ms": self._last_det_ballistic_ms,
            "det_ec_ms": self._last_det_ec_ms,
            "det_rate_ms": self._last_det_rate_limit_ms,
            "det_packet_ms": self._last_det_packet_ms,
            "pred_ms": self._last_pred_ms if self.enable_prediction else None,
            "pred_pre_ms": self._last_pred_pre_ms if self.enable_prediction else None,
            "pred_h2d_ms": self._last_pred_h2d_ms if self.enable_prediction else None,
            "pred_fwd_ms": self._last_pred_fwd_ms if self.enable_prediction else None,
            "pred_d2h_ms": self._last_pred_d2h_ms if self.enable_prediction else None,
            "pred_angle_ms": self._last_pred_resolve_ms if self.enable_prediction else None,
            "pred_ec_ms": self._last_pred_ec_ms if self.enable_prediction else None,
            "pred_rate_ms": self._last_pred_rate_limit_ms if self.enable_prediction else None,
            "pred_packet_ms": self._last_pred_packet_ms if self.enable_prediction else None,
            "fire_ms": self._last_fire_decide_ms if self.enable_prediction else None,
            "record_ms": self._last_record_ms if self.record_video else None,
            "pnp_ms": self._last_pnp_ms,
            "det_corner_ms": self._last_det_corner_ms,
            "pred_corner_ms": self._last_pred_corner_ms if self.enable_prediction else None,
            "loop_ms": self._last_loop_ms,
            "det_raw_candidates": self._last_detector_raw_candidates,
            "det_obj_candidates": self._last_detector_obj_candidates,
            "det_class_candidates": self._last_detector_class_candidates,
            "det_keep_candidates": self._last_detector_kept_candidates,
            "det_output_count": self._last_detector_output_count,
        }
        return metrics

    def _format_perf_metric(self, bucket: _PerfBucket, name: str, label: str) -> Optional[str]:
        summary = bucket.summarize(name)
        if summary is None:
            return None
        parts = [f"{label} avg={summary['avg']:.2f}"]
        for percentile in self.perf_log_percentiles:
            p_label = (
                f"p{int(percentile)}"
                if float(percentile).is_integer()
                else f"p{str(percentile).replace('.', '_')}"
            )
            value = summary.get(p_label)
            if value is not None:
                parts.append(f"{p_label}={value:.2f}")
        parts.append(f"max={summary['max']:.2f}")
        return " ".join(parts)

    def _record_perf_sample(self, has_target: bool):
        if not self.perf_log:
            return
        metrics = self._collect_perf_metrics()
        self._perf_buckets["all"].add(metrics)
        self._perf_buckets["target" if has_target else "idle"].add(metrics)

    def _maybe_log_perf(self, now: float):
        if not self.perf_log or (now - self._perf_last_log_time) < self.perf_log_interval_s:
            return
        self._perf_last_log_time = now
        metric_groups = [
            (
                "core",
                (
                    ("read_ms", "Read"),
                    ("det_total_ms", "Det"),
                    ("det_pre_ms", "Pre"),
                    ("det_infer_ms", "Infer"),
                    ("det_post_ms", "Post"),
                    ("loop_ms", "Loop"),
                ),
            ),
                (
                    "target",
                    (
                        ("update_ms", "Upd"),
                        ("pnp_ms", "PnP"),
                        ("det_angle_ms", "DetAng"),
                        ("pred_ms", "Pred"),
                    ),
                ),
                (
                    "detail",
                (
                    ("det_ballistic_ms", "Ball"),
                    ("det_rate_ms", "DetRate"),
                    ("pred_angle_ms", "PredAng"),
                    ("pred_rate_ms", "PredRate"),
                    ("fire_ms", "Fire"),
                    ("record_ms", "Record"),
                    ("det_select_ms", "Select"),
                ),
            ),
            (
                "counts",
                (
                    ("det_raw_candidates", "Raw"),
                    ("det_obj_candidates", "Obj"),
                    ("det_class_candidates", "Cls"),
                    ("det_keep_candidates", "Keep"),
                    ("det_output_count", "Out"),
                ),
            ),
        ]
        for bucket_name, bucket in self._perf_buckets.items():
            if bucket.frame_count <= 0:
                continue
            print(f"PERF[{bucket_name}] n={bucket.frame_count}")
            for group_name, group_metrics in metric_groups:
                parts = [
                    self._format_perf_metric(bucket, metric_name, label)
                    for metric_name, label in group_metrics
                ]
                parts = [part for part in parts if part is not None]
                if parts:
                    print(f"  {group_name}: " + " | ".join(parts))
            bucket.clear()
        worker_metric_groups = {
            "display": [
                (
                    "core",
                    (
                        ("draw_ms", "Draw"),
                        ("present_ms", "Present"),
                        ("wait_ms", "Wait"),
                        ("display_total_ms", "Display"),
                    ),
                ),
                (
                    "rate",
                    (
                        ("display_fps", "FPS"),
                        ("display_drop_count", "Drop"),
                    ),
                ),
            ],
            "send": [
                (
                    "core",
                    (
                        ("send_ms", "Send"),
                        ("send_sleep_ms", "Sleep"),
                    ),
                ),
                (
                    "rate",
                    (
                        ("send_fps", "FPS"),
                        ("send_drop_count", "Drop"),
                    ),
                ),
            ],
        }
        with self._worker_perf_lock:
            for bucket_name, metric_groups in worker_metric_groups.items():
                bucket = self._worker_perf_buckets[bucket_name]
                if bucket.frame_count <= 0:
                    continue
                print(f"PERF[{bucket_name}] n={bucket.frame_count}")
                for group_name, group_metrics in metric_groups:
                    parts = [
                        self._format_perf_metric(bucket, metric_name, metric_label)
                        for metric_name, metric_label in group_metrics
                    ]
                    parts = [part for part in parts if part is not None]
                    if parts:
                        print(f"  {group_name}: " + " | ".join(parts))
                bucket.clear()

    def _estimate_lag_compensation(self) -> Tuple[float, int]:
        if not self.pred_async or not self.lag_comp_enable:
            return 0.0, 0
        lag = self._last_pred_lag
        if lag is None or lag <= 0:
            return 0.0, 0
        frame_period = self._frame_period_ema_s
        if frame_period <= 1e-4:
            frame_period = 1.0 / self.predict_fps
        lag_frames = int(lag)
        lag_time_s = max(0.0, lag_frames * frame_period)
        if self.lag_comp_max_s > 0.0:
            lag_time_s = min(lag_time_s, self.lag_comp_max_s)
        return lag_time_s, lag_frames

    def _start_pred_worker(self):
        if not self.pred_async or self._pred_thread is not None:
            return
        self._pred_stop_event.clear()
        self._pred_thread = threading.Thread(
            target=self._pred_worker_loop, name="pred-worker", daemon=True
        )
        self._pred_thread.start()

    def _stop_pred_worker(self):
        if self._pred_thread is None:
            return
        self._pred_stop_event.set()
        try:
            if self._pred_queue is not None:
                self._pred_queue.put_nowait(None)
        except Exception:
            pass
        self._pred_thread.join(timeout=1.0)
        self._pred_thread = None

    def _pred_worker_loop(self):
        while not self._pred_stop_event.is_set():
            try:
                job = self._pred_queue.get(timeout=0.05) if self._pred_queue else None
            except queue.Empty:
                continue
            if job is None:
                continue
            frame_id, frame, bbox = job
            start = time.perf_counter()
            with self._pred_lock:
                predicted_coords = self._predict_coordinates(frame, bbox)
            elapsed = (time.perf_counter() - start) * 1000.0
            with self._pred_result_lock:
                self._last_pred_ms = elapsed
                self._pred_result = (frame_id, predicted_coords, frame.shape)

    def _submit_pred_job(self, frame_id: int, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        if not self.pred_async or self._pred_queue is None:
            return
        job = (frame_id, frame, bbox)
        try:
            self._pred_queue.put_nowait(job)
        except queue.Full:
            try:
                _ = self._pred_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._pred_queue.put_nowait(job)
            except queue.Full:
                pass

    def _get_latest_pred_result(self):
        if not self.pred_async:
            return None
        with self._pred_result_lock:
            return self._pred_result

    def _predict_coordinates(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        self._last_pred_pre_ms = 0.0
        self._last_pred_h2d_ms = 0.0
        self._last_pred_fwd_ms = 0.0
        self._last_pred_d2h_ms = 0.0
        if not self.enable_prediction or self.coordinate_model is None:
            self._last_pred_pre_ms = 0.0
            return None

        pre_start = time.perf_counter()
        frame_h, frame_w = frame.shape[:2]
        bx1, by1, bx2, by2 = bbox
        coord_now = np.array(
            [
                (bx1 + bx2) * 0.5 / max(1, frame_w),
                (by1 + by2) * 0.5 / max(1, frame_h),
                (bx2 - bx1) / max(1, frame_w),
                (by2 - by1) / max(1, frame_h),
            ],
            dtype=np.float32,
        )
        coord_now = np.clip(coord_now, 0.0, 1.0)
        roi_bbox = self._get_roi_bbox(frame.shape, bbox)
        processed = self._preprocess_frame(frame, roi_bbox)
        self.frame_buffer[self._buffer_idx] = processed
        self.coord_buffer[self._buffer_idx] = coord_now
        self.roi_bbox_buffer[self._buffer_idx] = roi_bbox
        self._buffer_idx = (self._buffer_idx + 1) % self.sequence_length
        if self._buffer_len < self.sequence_length:
            self._buffer_len += 1

        self._last_buffer_len = self._buffer_len
        self._pred_ready = self._buffer_len >= self.input_sequence_length

        if self._buffer_len < self.input_sequence_length:
            self._last_pred_pre_ms = (time.perf_counter() - pre_start) * 1000.0
            return None

        start_idx = (self._buffer_idx - self.input_sequence_length) % self.sequence_length
        if start_idx < self._buffer_idx:
            recent_frames = self.frame_buffer[start_idx:self._buffer_idx]
            recent_coords = self.coord_buffer[start_idx:self._buffer_idx]
        else:
            recent_frames = np.concatenate(
                (self.frame_buffer[start_idx:], self.frame_buffer[: self._buffer_idx]),
                axis=0,
            )
            recent_coords = np.concatenate(
                (self.coord_buffer[start_idx:], self.coord_buffer[: self._buffer_idx]),
                axis=0,
            )
        if not recent_frames.flags["C_CONTIGUOUS"]:
            recent_frames = np.ascontiguousarray(recent_frames)
        if not recent_coords.flags["C_CONTIGUOUS"]:
            recent_coords = np.ascontiguousarray(recent_coords)
        input_sequence_np = recent_frames.reshape(1, self.input_sequence_length, -1)
        input_coords_np = recent_coords.reshape(1, self.input_sequence_length, 4)
        input_sequence_cpu = torch.from_numpy(input_sequence_np)
        input_coords_cpu = torch.from_numpy(input_coords_np)
        self._last_pred_pre_ms = (time.perf_counter() - pre_start) * 1000.0

        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        profile_timing = self.profile_pred and use_cuda

        if profile_timing:
            torch.cuda.synchronize()
        h2d_start = time.perf_counter()
        if self._pred_input_gpu is not None:
            if self._pred_input_cpu is not None:
                self._pred_input_cpu.copy_(input_sequence_cpu, non_blocking=True)
                self._pred_input_gpu.copy_(self._pred_input_cpu, non_blocking=True)
            else:
                self._pred_input_gpu.copy_(input_sequence_cpu, non_blocking=True)
            input_sequence = self._pred_input_gpu
        else:
            input_sequence = input_sequence_cpu.to(self.device, non_blocking=use_cuda)
        input_coords = input_coords_cpu.to(self.device, non_blocking=use_cuda)
        if profile_timing:
            torch.cuda.synchronize()
            self._last_pred_h2d_ms = (time.perf_counter() - h2d_start) * 1000.0
        else:
            self._last_pred_h2d_ms = 0.0

        if profile_timing:
            torch.cuda.synchronize()
        fwd_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.coordinate_model(input_sequence, input_coords=input_coords)
        if profile_timing:
            torch.cuda.synchronize()
            self._last_pred_fwd_ms = (time.perf_counter() - fwd_start) * 1000.0
        else:
            self._last_pred_fwd_ms = 0.0

        if profile_timing:
            torch.cuda.synchronize()
        d2h_start = time.perf_counter()
        predicted_coordinates = outputs["predicted_coordinates"].cpu().numpy()[0]
        if profile_timing:
            torch.cuda.synchronize()
            self._last_pred_d2h_ms = (time.perf_counter() - d2h_start) * 1000.0
        else:
            self._last_pred_d2h_ms = 0.0

        last_idx = (self._buffer_idx - 1) % self.sequence_length
        roi_bbox_for_pred = self.roi_bbox_buffer[last_idx]
        predicted_global = self._convert_roi_prediction_to_frame(
            predicted_coordinates, roi_bbox_for_pred, frame.shape
        )
        return np.clip(predicted_global, 0.0, 1.0)

    def _update_prediction_state(self, pred_bbox: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = pred_bbox
        center = (float(x1 + (x2 - x1) / 2.0), float(y1 + (y2 - y1) / 2.0))
        size = (float(max(1, x2 - x1)), float(max(1, y2 - y1)))
        now = time.time()
        if self._last_pred_center is not None and self._last_pred_time is not None:
            dt = max(1e-3, now - self._last_pred_time)
            vx = (center[0] - self._last_pred_center[0]) / dt
            vy = (center[1] - self._last_pred_center[1]) / dt
            self._pred_velocity = (vx, vy)
        self._last_pred_center = center
        self._last_pred_size = size
        self._last_pred_time = now

    def _update_target_rate_estimate(self, rel_yaw_deg: float, rel_pitch_deg: float, now_s: float):
        self._update_motion_state(rel_yaw_deg, rel_pitch_deg, None, None, now_s)

    def _apply_ec_angle_lead(self, yaw_deg: float, pitch_deg: float) -> Tuple[float, float, Optional[str]]:
        if not self.enable_ec_feedback:
            return yaw_deg, pitch_deg, None
        ec = self._get_ec_feedback()
        if ec is None:
            return yaw_deg, pitch_deg, None
        now = time.time()
        ts = float(ec[-1])
        if abs(now - ts) > 0.2:
            return yaw_deg, pitch_deg, None

        total_time, _, _ = self._compute_time_compensation()
        lead_time = float(total_time) + float(self.ec_additional_predict_time_s) + float(self.ec_t0_s)
        if lead_time <= 1e-3:
            return yaw_deg, pitch_deg, None

        yaw_ff = self._target_yaw_rate_dps * lead_time
        pitch_ff = self._target_pitch_rate_dps * lead_time


        # 防止极端情况下前馈过大
        max_ff = 20.0
        if yaw_ff > max_ff:
            yaw_ff = max_ff
        elif yaw_ff < -max_ff:
            yaw_ff = -max_ff
        if pitch_ff > max_ff:
            pitch_ff = max_ff
        elif pitch_ff < -max_ff:
            pitch_ff = -max_ff

        return yaw_deg + yaw_ff, pitch_deg + pitch_ff, f"+EC({lead_time:.3f}s)"

    def _compute_time_compensation(self) -> Tuple[float, float, int]:
        distance = self._last_distance_m
        if distance is None or distance <= 0.0:
            self._last_raw_distance_m = distance
            self._last_comp_distance_m = None
            self._last_comp_time_s = 0.0
            self._last_extra_time_s = 0.0
            self._last_lead_frames = 0
            self._last_lag_time_s = 0.0
            self._last_lag_frames = 0
            self._last_flight_time_s = None
            return 0.0, 0.0, 0

        raw_distance = float(distance)
        comp_distance = raw_distance
        if self.max_comp_distance_m > 0.0:
            comp_distance = min(comp_distance, self.max_comp_distance_m)

        flight_time = comp_distance / self.bullet_speed_mps
        if (
            self.use_ballistic_time
            and self._last_ballistic_time_s is not None
            and (
                self.max_comp_distance_m <= 0.0
                or raw_distance <= self.max_comp_distance_m + 1e-6
            )
        ):
            flight_time = self._last_ballistic_time_s
        lag_time_s, lag_frames = self._estimate_lag_compensation()
        total_time = flight_time + self.system_latency_s + lag_time_s
        model_time = 0.0
        if self.model_bullet_speed_mps > 0.0:
            model_time = comp_distance / self.model_bullet_speed_mps + self.model_latency_s
        extra_time = max(0.0, total_time - model_time)
        lead_frames = int(round(self.predict_fps * total_time))
        if lead_frames < 0:
            lead_frames = 0

        self._last_raw_distance_m = raw_distance
        self._last_comp_distance_m = comp_distance
        self._last_comp_time_s = total_time
        self._last_extra_time_s = extra_time
        self._last_lead_frames = lead_frames
        self._last_lag_time_s = lag_time_s
        self._last_lag_frames = lag_frames
        self._last_flight_time_s = flight_time
        return total_time, extra_time, lead_frames

    def _shift_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        dx: float,
        dy: float,
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        w = float(max(1, x2 - x1))
        h = float(max(1, y2 - y1))
        cx = float(x1) + w / 2.0 + dx
        cy = float(y1) + h / 2.0 + dy
        cx = max(0.0, min(width - 1.0, cx))
        cy = max(0.0, min(height - 1.0, cy))
        nx1 = int(round(cx - w / 2.0))
        ny1 = int(round(cy - h / 2.0))
        nx2 = int(round(cx + w / 2.0))
        ny2 = int(round(cy + h / 2.0))
        nx1 = max(0, min(width - 1, nx1))
        ny1 = max(0, min(height - 1, ny1))
        nx2 = max(0, min(width - 1, nx2))
        ny2 = max(0, min(height - 1, ny2))
        if nx2 <= nx1:
            nx2 = min(width - 1, nx1 + 1)
        if ny2 <= ny1:
            ny2 = min(height - 1, ny1 + 1)
        return nx1, ny1, nx2, ny2

    def _apply_time_compensation(
        self,
        pred_bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[Tuple[int, int, int, int], bool]:
        if self.disable_image_time_comp_with_feedback and self.enable_ec_feedback:
            ec = self._get_ec_feedback()
            if ec is not None:
                now = time.time()
                ts = float(ec[-1])
                if abs(now - ts) <= 0.2:
                    self._last_image_time_comp_scale = 0.0
                    return pred_bbox, False

        self._last_image_time_comp_scale = 0.0 if (self.spin_aware and self._spin_active) else 1.0
        _, extra_time, _ = self._compute_time_compensation()
        if extra_time <= 1e-3:
            return pred_bbox, False
        vx, vy = self._pred_velocity
        if abs(vx) < 1e-3 and abs(vy) < 1e-3:
            return pred_bbox, False
        dx = vx * extra_time * self._last_image_time_comp_scale
        dy = vy * extra_time * self._last_image_time_comp_scale
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            return pred_bbox, False
        return self._shift_bbox(pred_bbox, dx, dy, frame_shape), True

    def _apply_rate_limit(
        self,
        yaw_deg: float,
        pitch_deg: float,
        last_angles: Optional[Tuple[float, float]],
        last_time: Optional[float],
    ) -> Tuple[float, float, Tuple[float, float], float]:
        now = time.time()
        if last_angles is None or last_time is None:
            return yaw_deg, pitch_deg, (yaw_deg, pitch_deg), now

        dt = max(1e-3, now - last_time)
        limited_yaw = yaw_deg
        limited_pitch = pitch_deg

        if self.max_yaw_rate > 0.0:
            max_delta = self.max_yaw_rate * dt
            delta = yaw_deg - last_angles[0]
            if abs(delta) > max_delta:
                limited_yaw = last_angles[0] + np.sign(delta) * max_delta
        if self.max_pitch_rate > 0.0:
            max_delta = self.max_pitch_rate * dt
            delta = pitch_deg - last_angles[1]
            if abs(delta) > max_delta:
                limited_pitch = last_angles[1] + np.sign(delta) * max_delta

        return limited_yaw, limited_pitch, (limited_yaw, limited_pitch), now

    def _solve_ballistic_pitch(
        self, tvec: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        if not self.ballistic_enable:
            return None
        if tvec is None:
            return None
        if self.bullet_speed_mps <= 0.0 or self.ballistic_dt_s <= 0.0:
            return None
        tvec_flat = tvec.flatten()
        if tvec_flat.size < 3:
            return None
        dx = float(tvec_flat[0])
        dy = float(tvec_flat[1])
        dz = float(tvec_flat[2])
        horiz = float(np.hypot(dx, dz))
        if horiz <= 1e-6:
            return None
        height = -(dy - self.gun_offset_y)
        solution = solve_pitch_for_target(
            range_m=horiz,
            height_m=height,
            muzzle_speed_mps=self.bullet_speed_mps,
            drag_k=self.ballistic_drag_k,
            pitch_min_rad=self.ballistic_pitch_min_rad,
            pitch_max_rad=self.ballistic_pitch_max_rad,
            dt=self.ballistic_dt_s,
            g=self.ballistic_g,
        )
        if solution is None:
            return None
        pitch_rad, flight_time_s = solution
        pitch_deg = math.degrees(float(pitch_rad))
        if self.invert_pitch:
            pitch_deg = -pitch_deg
        return pitch_deg, float(flight_time_s)

    def _apply_ballistic_compensation(
        self, pitch_deg: float, source: str
    ) -> Tuple[float, str]:
        self._last_det_ballistic_delta = None
        self._last_ballistic_time_s = None
        if not self.ballistic_enable:
            return pitch_deg, source
        if source != "PNP" or self._last_pnp_tvec is None:
            return pitch_deg, source
        solution = self._solve_ballistic_pitch(self._last_pnp_tvec)
        if solution is None:
            return pitch_deg, source
        ballistic_pitch_deg, flight_time_s = solution
        self._last_det_ballistic_delta = ballistic_pitch_deg - pitch_deg
        self._last_ballistic_time_s = flight_time_s
        if self.show_tx:
            print(
                f"弹道补偿: pitch={ballistic_pitch_deg:.2f} "
                f"dpitch={self._last_det_ballistic_delta:.2f} t={flight_time_s:.3f}s"
            )
        return ballistic_pitch_deg, f"{source}+BALL"

    def _compute_pnp_angles(
        self,
        bbox: Tuple[int, int, int, int],
        frame: np.ndarray,
        label: str,
        target_type: str,
        corners: Optional[np.ndarray] = None,
    ) -> Tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        log_comp = abs(self.gun_offset_y) > 0.0 and self.show_tx
        intrinsics = self._get_intrinsics_for_frame(frame.shape)
        (
            yaw,
            pitch,
            err,
            elapsed_ms,
            comp_yaw,
            comp_pitch,
            distance_m,
            tvec,
            rvec,
        ) = solve_angles_from_bbox(
            frame,
            bbox,
            target_type=target_type,
            intrinsics=intrinsics,
            max_pnp_error=self.max_pnp_error,
            gun_offset_y=self.gun_offset_y,
            use_corners=self.use_corners,
            corners=corners,
            bbox_shrink=self.bbox_shrink,
            return_comp=True,
            return_distance=True,
            return_tvec=True,
            return_rvec=True,
            log_comp=log_comp,
            log_label=label,
        )
        if elapsed_ms is not None:
            self._last_pnp_ms = elapsed_ms
        if distance_m is not None:
            self._last_distance_m = distance_m
        if yaw is not None and pitch is not None and tvec is not None and rvec is not None:
            self._last_pnp_tvec = tvec
            self._last_pnp_rvec = rvec
        else:
            self._last_pnp_tvec = None
            self._last_pnp_rvec = None
        return yaw, pitch, err, comp_yaw, comp_pitch

    def _resolve_angles(
        self,
        bbox: Tuple[int, int, int, int],
        x_center: float,
        y_center: float,
        frame: np.ndarray,
        label: str,
        target_type: str,
        corners: Optional[np.ndarray] = None,
        force_pixel: bool = False,
    ) -> Tuple[float, float, str, Optional[float], Optional[float], Optional[float]]:
        pnp_yaw = None
        pnp_pitch = None
        pnp_err = None
        comp_yaw = None
        comp_pitch = None
        if not force_pixel:
            pnp_yaw, pnp_pitch, pnp_err, comp_yaw, comp_pitch = self._compute_pnp_angles(
                bbox, frame, label, target_type, corners=corners
            )
        pixel_yaw, pixel_pitch = self._pixel_to_angle(x_center, y_center, frame.shape)
        source = "PNP"
        yaw_deg = pnp_yaw
        pitch_deg = pnp_pitch

        if force_pixel or yaw_deg is None or pitch_deg is None:
            source = "PIX"
            yaw_deg = pixel_yaw
            pitch_deg = pixel_pitch
            comp_yaw = None
            comp_pitch = None

        if self.invert_yaw:
            yaw_deg = -yaw_deg
            if comp_yaw is not None:
                comp_yaw = -comp_yaw
        if self.invert_pitch:
            pitch_deg = -pitch_deg
            if comp_pitch is not None:
                comp_pitch = -comp_pitch

        if self.show_tx:
            print(
                f"{label}角度: src={source} yaw={yaw_deg:.2f} pitch={pitch_deg:.2f} "
                f"(pnp={pnp_yaw if pnp_yaw is not None else 'NA'} "
                f"{pnp_pitch if pnp_pitch is not None else 'NA'}, "
                f"pix={pixel_yaw:.2f} {pixel_pitch:.2f})"
            )
        return yaw_deg, pitch_deg, source, pnp_err, comp_yaw, comp_pitch

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
    def _normalize_class_ids(
        class_ids: Optional[Sequence[int]],
        label: str,
    ) -> Optional[Tuple[int, ...]]:
        if class_ids is None:
            return None
        if isinstance(class_ids, str):
            parts = [part.strip() for part in class_ids.split(",")]
            values = [part for part in parts if part]
        else:
            try:
                values = list(class_ids)
            except TypeError:
                values = [class_ids]

        normalized = []
        seen = set()
        for value in values:
            try:
                class_id = int(value)
            except Exception as exc:
                raise ValueError(f"{label} must contain integers: {value!r}") from exc
            if class_id in seen:
                continue
            seen.add(class_id)
            normalized.append(class_id)
        return tuple(normalized) if normalized else None

    @staticmethod
    def _normalize_target_class_ids(
        target_class_ids: Optional[Sequence[int]],
    ) -> Optional[Tuple[int, ...]]:
        return AimPipeline._normalize_class_ids(target_class_ids, "target_class_ids")

    def _target_color_allows_detection(self, detection) -> bool:
        if self._target_color_class_ids is None and not self._excluded_class_ids:
            return True
        if not isinstance(detection, dict):
            return False
        class_id = detection.get("class")
        if class_id is None:
            return False
        try:
            class_id = int(class_id)
        except Exception:
            return False
        if self._target_color_class_ids is not None and class_id not in self._target_color_class_ids:
            return False
        if class_id in self._excluded_class_ids:
            return False
        return True

    def _filter_detections_by_target_color(self, detections):
        if not detections or self._target_color_class_ids is None:
            return detections
        return [
            detection
            for detection in detections
            if self._target_color_allows_detection(detection)
        ]

    def _select_best_detection(self, detections):
        if not detections:
            return None
        filtered = self._filter_detections_by_target_color(detections)
        if not filtered:
            return None
        return max(filtered, key=lambda item: item.get("confidence", 0))

    @staticmethod
    def _describe_detection_class(detection) -> str:
        if not isinstance(detection, dict):
            return "unknown"
        parts = []
        class_name = detection.get("class_name")
        if class_name not in (None, ""):
            parts.append(str(class_name))
        class_id = detection.get("class")
        if class_id is not None:
            parts.append(f"id={class_id}")
        return " ".join(parts) if parts else "unknown"

    def _resolve_target_type_from_detection(
        self,
        detection,
        phase: str,
        last_target_type: Optional[str],
    ) -> str:
        target_type = self.target_type
        if not self.auto_target_type:
            return target_type
        target_type = choose_target_type_by_detection(detection, self.target_type)
        if self.show_tx and target_type != last_target_type:
            class_desc = self._describe_detection_class(detection)
            print(f"自动目标类型({phase}): {target_type} [{class_desc}]")
        return target_type

    def _bbox_from_normalized(
        self, coords: np.ndarray, frame_shape: Tuple[int, int, int]
    ) -> Tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        x_center, y_center, w, h = coords
        x_center_px = x_center * width
        y_center_px = y_center * height
        w_px = max(1.0, w * width)
        h_px = max(1.0, h * height)
        x1 = int(max(0, min(width - 1, x_center_px - w_px / 2)))
        y1 = int(max(0, min(height - 1, y_center_px - h_px / 2)))
        x2 = int(max(0, min(width - 1, x_center_px + w_px / 2)))
        y2 = int(max(0, min(height - 1, y_center_px + h_px / 2)))
        if x2 <= x1:
            x2 = min(width - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(height - 1, y1 + 1)
        return x1, y1, x2, y2

    @staticmethod
    def _normalize_detection_corners(corners) -> Optional[np.ndarray]:
        if corners is None:
            return None
        pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
            return None
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _build_packet(
        self,
        yaw_deg: float,
        pitch_deg: float,
        x_center: float,
        y_center: float,
        has_target: bool,
        fire_cmd: int = 0x00,
    ) -> bytes:
        return build_armor_packet(
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            armor_x=int(round(x_center)),
            armor_y=int(round(y_center)),
            armor_cmd=0x01 if has_target else 0x00,
            fire_cmd=fire_cmd,
            angle_scale=self.angle_scale,
        )

    def _build_prediction_fallback_packet(
        self,
        det_yaw: float,
        det_pitch: float,
        x_center: float,
        y_center: float,
    ) -> Tuple[bytes, Tuple[float, float, float, float, str]]:
        source = "DET_WAIT_PRED"
        fire_cmd = 0x00
        if self.spin_aware and self._spin_active:
            source = "DET_SPIN_FALLBACK"
            fire_cmd = self._decide_fire_cmd(has_prediction=True)
        else:
            self._last_fire_reason = "NO_PRED"
        packet = self._build_packet(det_yaw, det_pitch, x_center, y_center, True, fire_cmd=fire_cmd)
        return packet, (det_yaw, det_pitch, x_center, y_center, source)

    def _decide_fire_cmd(self, has_prediction: bool) -> int:
        if not has_prediction:
            self._last_fire_reason = "NO_PRED"
            return 0x00

        now = time.time()
        conf = self._latest_detection_conf
        conf_val = float(conf) if conf is not None else 0.0

        if conf_val >= self.fire_confidence_threshold:
            self._last_fire_ts = now
            self._last_fire_reason = f"CONF({conf_val:.2f})"
            return 0x01

        if self.fire_force_interval_s > 0.0 and (now - self._last_fire_ts) >= self.fire_force_interval_s:
            self._last_fire_ts = now
            self._last_fire_reason = f"FORCE({conf_val:.2f})"
            return 0x01

        self._last_fire_reason = f"HOLD({conf_val:.2f})"
        return 0x00

    def _update_packets(self, frame: np.ndarray, detection, frame_id: int):
        start_time = time.perf_counter()
        self._last_pnp_ms = 0.0
        self._last_det_angle_ms = 0.0
        self._last_det_ballistic_ms = 0.0
        self._last_det_ec_ms = 0.0
        self._last_det_rate_limit_ms = 0.0
        self._last_det_packet_ms = 0.0
        if not self.pred_async:
            self._last_pred_ms = 0.0
        self._last_pred_resolve_ms = 0.0
        self._last_pred_ec_ms = 0.0
        self._last_pred_rate_limit_ms = 0.0
        self._last_pred_packet_ms = 0.0
        self._last_fire_decide_ms = 0.0
        self._last_det_corner_ms = 0.0
        self._last_pred_corner_ms = 0.0
        if detection is not None and not self._target_color_allows_detection(detection):
            detection = None
        if detection is None:
            self._lost_count += 1
            if self.pred_async:
                with self._pred_lock:
                    self._reset_prediction_buffers()
                with self._pred_result_lock:
                    self._pred_result = None
            if self.lost_threshold > 0 and self._lost_count < self.lost_threshold:
                height, width = frame.shape[:2]
                hold_x = float(width / 2.0)
                hold_y = float(height / 2.0)
                hold_packet = self._build_packet(0.0, 0.0, hold_x, hold_y, True, fire_cmd=0x00)
                self._latest_detection_packet = hold_packet
                self._latest_prediction_packet = hold_packet
                self._latest_detection_bbox = None
                self._latest_prediction_bbox = None
                self._latest_detection_info = (0.0, 0.0, hold_x, hold_y, "HOLD")
                self._latest_prediction_info = (0.0, 0.0, hold_x, hold_y, "HOLD")
                self._last_fire_reason = "HOLD"
                self._latest_detection_conf = None
                self._latest_detection_corners = None
                self._latest_prediction_corners = None
                self._last_det_comp = None
                self._last_det_ballistic_delta = None
                self._last_ballistic_time_s = None
                self._last_pnp_tvec = None
                self._last_update_ms = (time.perf_counter() - start_time) * 1000.0
                return
            if self.pred_async:
                with self._pred_lock:
                    self._reset_prediction_buffers()
            else:
                self._reset_prediction_buffers()
            self._latest_detection_packet = self._empty_packet
            self._latest_prediction_packet = self._empty_packet
            self._latest_detection_bbox = None
            self._latest_prediction_bbox = None
            self._latest_detection_info = None
            self._latest_prediction_info = None
            self._last_fire_reason = "NO_TARGET"
            self._latest_detection_conf = None
            self._latest_detection_corners = None
            self._latest_prediction_corners = None
            self._last_det_comp = None
            self._last_det_ballistic_delta = None
            self._last_ballistic_time_s = None
            self._last_pnp_tvec = None
            self._last_update_ms = (time.perf_counter() - start_time) * 1000.0
            return

        self._lost_count = 0
        x1, y1, x2, y2 = detection["bbox"]
        height, width = frame.shape[:2]
        x1 = max(0, min(int(x1), width - 1))
        x2 = max(0, min(int(x2), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        y2 = max(0, min(int(y2), height - 1))
        if x2 <= x1:
            x2 = min(width - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(height - 1, y1 + 1)

        x_center = float(x1 + (x2 - x1) / 2.0)
        y_center = float(y1 + (y2 - y1) / 2.0)
        self._last_detection_bbox = (x1, y1, x2, y2)
        self._latest_detection_conf = detection.get("confidence")
        detection_corners = self._normalize_detection_corners(detection.get("corners"))
        if detection_corners is not None:
            self._latest_detection_corners = detection_corners
            self._last_det_corner_ms = 0.0
        elif self.show_window and self.use_corners:
            corner_start = time.perf_counter()
            self._latest_detection_corners = detect_armor_corners(frame, (x1, y1, x2, y2))
            self._last_det_corner_ms = (time.perf_counter() - corner_start) * 1000.0
        else:
            self._latest_detection_corners = None

        det_target_type = self._resolve_target_type_from_detection(
            detection,
            "检测",
            getattr(self, "_last_det_target_type", None),
        )
        self._last_det_target_type = det_target_type

        det_angle_start = time.perf_counter()
        det_yaw, det_pitch, det_source, det_err, det_comp_yaw, det_comp_pitch = self._resolve_angles(
            (x1, y1, x2, y2),
            x_center,
            y_center,
            frame,
            "检测",
            det_target_type,
            corners=detection_corners,
        )
        self._last_det_angle_ms = (time.perf_counter() - det_angle_start) * 1000.0
        if det_err is not None and self.show_tx and det_source.startswith("PNP"):
            print(f"PnP重投影误差(检测): {det_err:.3f}")
        if det_source == "PNP" and det_comp_yaw is not None and det_comp_pitch is not None:
            self._last_det_comp = (det_comp_yaw, det_comp_pitch)
        else:
            self._last_det_comp = None

        motion_now = time.time()
        # 用检测角度更新“世界坐标”的目标角速度估计（剥离云台自转造成的画面运动）
        self._update_motion_state(det_yaw, det_pitch, (x1, y1, x2, y2), detection_corners, motion_now)

        det_ballistic_start = time.perf_counter()
        det_pitch, det_source = self._apply_ballistic_compensation(det_pitch, det_source)
        self._last_det_ballistic_ms = (time.perf_counter() - det_ballistic_start) * 1000.0
        det_ec_start = time.perf_counter()
        det_yaw, det_pitch, ec_tag = self._apply_ec_angle_lead(det_yaw, det_pitch)
        self._last_det_ec_ms = (time.perf_counter() - det_ec_start) * 1000.0
        if ec_tag is not None:
            det_source = f"{det_source}{ec_tag}"
        det_yaw, det_pitch, spin_tag = self._apply_spin_yaw_bias(det_yaw, det_pitch)
        if spin_tag is not None:
            det_source = f"{det_source}{spin_tag}"
        det_rate_start = time.perf_counter()
        det_yaw, det_pitch, self._last_det_angles, self._last_det_angle_time = self._apply_rate_limit(
            det_yaw, det_pitch, self._last_det_angles, self._last_det_angle_time
        )
        self._last_det_rate_limit_ms = (time.perf_counter() - det_rate_start) * 1000.0
        det_packet_start = time.perf_counter()
        self._latest_detection_packet = self._build_packet(
            det_yaw, det_pitch, x_center, y_center, True, fire_cmd=0x00
        )
        self._last_det_packet_ms = (time.perf_counter() - det_packet_start) * 1000.0
        self._latest_detection_bbox = (x1, y1, x2, y2)
        self._latest_detection_info = (det_yaw, det_pitch, x_center, y_center, det_source)

        if not self.enable_prediction:
            fire_start = time.perf_counter()
            det_fire_cmd = self._decide_fire_cmd(has_prediction=True)
            self._last_fire_decide_ms = (time.perf_counter() - fire_start) * 1000.0
            pred_packet_start = time.perf_counter()
            self._latest_prediction_packet = self._build_packet(
                det_yaw, det_pitch, x_center, y_center, True, fire_cmd=det_fire_cmd
            )
            self._last_pred_packet_ms = (time.perf_counter() - pred_packet_start) * 1000.0
            self._latest_prediction_bbox = self._latest_detection_bbox
            self._latest_prediction_info = self._latest_detection_info
            self._latest_prediction_corners = self._latest_detection_corners
            self._last_update_ms = (time.perf_counter() - start_time) * 1000.0
            return

        if self.pred_async:
            self._submit_pred_job(frame_id, frame, (x1, y1, x2, y2))
            pred_result = self._get_latest_pred_result()
            predicted_coords = None
            pred_frame_id = None
            pred_frame_shape = None
            if pred_result is not None:
                pred_frame_id, predicted_coords, pred_frame_shape = pred_result
            if pred_frame_id is None or predicted_coords is None:
                pred_packet_start = time.perf_counter()
                (
                    self._latest_prediction_packet,
                    self._latest_prediction_info,
                ) = self._build_prediction_fallback_packet(
                    det_yaw, det_pitch, x_center, y_center
                )
                self._last_pred_packet_ms = (time.perf_counter() - pred_packet_start) * 1000.0
                self._latest_prediction_bbox = self._latest_detection_bbox
                self._latest_prediction_corners = self._latest_detection_corners
                self._last_update_ms = (time.perf_counter() - start_time) * 1000.0
                return
            lag = frame_id - pred_frame_id
            self._last_pred_lag = lag

            use_frame_shape = pred_frame_shape or frame.shape
            pred_bbox = self._bbox_from_normalized(predicted_coords, use_frame_shape)
            self._update_prediction_state(pred_bbox)
            comp_bbox, comp_applied = self._apply_time_compensation(pred_bbox, use_frame_shape)
            px1, py1, px2, py2 = comp_bbox
            pred_x_center = float(px1 + (px2 - px1) / 2.0)
            pred_y_center = float(py1 + (py2 - py1) / 2.0)
            if self.show_window and self.use_corners and not comp_applied:
                corner_start = time.perf_counter()
                self._latest_prediction_corners = detect_armor_corners(frame, pred_bbox)
                self._last_pred_corner_ms = (time.perf_counter() - corner_start) * 1000.0
            else:
                self._latest_prediction_corners = None

            pred_target_type = self._resolve_target_type_from_detection(
                detection,
                "预测",
                getattr(self, "_last_pred_target_type", None),
            )
            self._last_pred_target_type = pred_target_type

            pred_resolve_start = time.perf_counter()
            pred_yaw, pred_pitch, pred_source, pred_err, _, _ = self._resolve_angles(
                comp_bbox,
                pred_x_center,
                pred_y_center,
                frame,
                "预测",
                pred_target_type,
                force_pixel=True,
            )
            self._last_pred_resolve_ms = (time.perf_counter() - pred_resolve_start) * 1000.0
            if pred_err is not None and self.show_tx and pred_source.startswith("PNP"):
                print(f"PnP重投影误差(预测): {pred_err:.3f}")
            if pred_source == "PIX" and self._last_det_comp is not None:
                comp_yaw, comp_pitch = self._last_det_comp
                pred_yaw += comp_yaw
                pred_pitch += comp_pitch
                pred_source = "PIX+COMP"
                if self.show_tx:
                    print(
                        f"预测补偿(复用检测): dyaw={comp_yaw:.2f} dpitch={comp_pitch:.2f}"
                    )
            if self._last_det_ballistic_delta is not None:
                pred_pitch += self._last_det_ballistic_delta
                pred_source = f"{pred_source}+BALL"
                if self.show_tx:
                    print(
                        f"预测补偿(弹道): dpitch={self._last_det_ballistic_delta:.2f}"
                    )
            pred_ec_start = time.perf_counter()
            pred_yaw, pred_pitch, ec_tag = self._apply_ec_angle_lead(pred_yaw, pred_pitch)
            self._last_pred_ec_ms = (time.perf_counter() - pred_ec_start) * 1000.0
            if ec_tag is not None:
                pred_source = f"{pred_source}{ec_tag}"
            pred_yaw, pred_pitch, spin_tag = self._apply_spin_yaw_bias(pred_yaw, pred_pitch)
            if spin_tag is not None:
                pred_source = f"{pred_source}{spin_tag}"
            pred_rate_start = time.perf_counter()
            pred_yaw, pred_pitch, self._last_pred_angles, self._last_pred_angle_time = self._apply_rate_limit(
                pred_yaw, pred_pitch, self._last_pred_angles, self._last_pred_angle_time
            )
            self._last_pred_rate_limit_ms = (time.perf_counter() - pred_rate_start) * 1000.0
            fire_start = time.perf_counter()
            pred_fire_cmd = self._decide_fire_cmd(has_prediction=True)
            self._last_fire_decide_ms = (time.perf_counter() - fire_start) * 1000.0
            pred_packet_start = time.perf_counter()
            self._latest_prediction_packet = self._build_packet(
                pred_yaw, pred_pitch, pred_x_center, pred_y_center, True, fire_cmd=pred_fire_cmd
            )
            self._last_pred_packet_ms = (time.perf_counter() - pred_packet_start) * 1000.0
            self._latest_prediction_bbox = comp_bbox
            self._latest_prediction_info = (
                pred_yaw,
                pred_pitch,
                pred_x_center,
                pred_y_center,
                pred_source,
            )
            self._last_update_ms = (time.perf_counter() - start_time) * 1000.0
            return

        pred_start = time.perf_counter()
        predicted_coords = self._predict_coordinates(frame, (x1, y1, x2, y2))
        self._last_pred_ms = (time.perf_counter() - pred_start) * 1000.0
        if predicted_coords is None:
            pred_packet_start = time.perf_counter()
            (
                self._latest_prediction_packet,
                self._latest_prediction_info,
            ) = self._build_prediction_fallback_packet(
                det_yaw, det_pitch, x_center, y_center
            )
            self._last_pred_packet_ms = (time.perf_counter() - pred_packet_start) * 1000.0
            self._latest_prediction_bbox = self._latest_detection_bbox
            self._latest_prediction_corners = self._latest_detection_corners
            self._last_update_ms = (time.perf_counter() - start_time) * 1000.0
            return

        pred_bbox = self._bbox_from_normalized(predicted_coords, frame.shape)
        self._update_prediction_state(pred_bbox)
        comp_bbox, comp_applied = self._apply_time_compensation(pred_bbox, frame.shape)
        px1, py1, px2, py2 = comp_bbox
        pred_x_center = float(px1 + (px2 - px1) / 2.0)
        pred_y_center = float(py1 + (py2 - py1) / 2.0)
        if self.show_window and self.use_corners and not comp_applied:
            corner_start = time.perf_counter()
            self._latest_prediction_corners = detect_armor_corners(frame, pred_bbox)
            self._last_pred_corner_ms = (time.perf_counter() - corner_start) * 1000.0
        else:
            self._latest_prediction_corners = None

        pred_target_type = self._resolve_target_type_from_detection(
            detection,
            "预测",
            getattr(self, "_last_pred_target_type", None),
        )
        self._last_pred_target_type = pred_target_type

        pred_resolve_start = time.perf_counter()
        pred_yaw, pred_pitch, pred_source, pred_err, _, _ = self._resolve_angles(
            comp_bbox,
            pred_x_center,
            pred_y_center,
            frame,
            "预测",
            pred_target_type,
            force_pixel=True,
        )
        self._last_pred_resolve_ms = (time.perf_counter() - pred_resolve_start) * 1000.0
        if pred_err is not None and self.show_tx and pred_source.startswith("PNP"):
            print(f"PnP重投影误差(预测): {pred_err:.3f}")
        if pred_source == "PIX" and self._last_det_comp is not None:
            comp_yaw, comp_pitch = self._last_det_comp
            pred_yaw += comp_yaw
            pred_pitch += comp_pitch
            pred_source = "PIX+COMP"
            if self.show_tx:
                print(
                    f"预测补偿(复用检测): dyaw={comp_yaw:.2f} dpitch={comp_pitch:.2f}"
                )
        if self._last_det_ballistic_delta is not None:
            pred_pitch += self._last_det_ballistic_delta
            pred_source = f"{pred_source}+BALL"
            if self.show_tx:
                print(
                    f"预测补偿(弹道): dpitch={self._last_det_ballistic_delta:.2f}"
                )
        pred_ec_start = time.perf_counter()
        pred_yaw, pred_pitch, ec_tag = self._apply_ec_angle_lead(pred_yaw, pred_pitch)
        self._last_pred_ec_ms = (time.perf_counter() - pred_ec_start) * 1000.0
        if ec_tag is not None:
            pred_source = f"{pred_source}{ec_tag}"
        pred_yaw, pred_pitch, spin_tag = self._apply_spin_yaw_bias(pred_yaw, pred_pitch)
        if spin_tag is not None:
            pred_source = f"{pred_source}{spin_tag}"
        pred_rate_start = time.perf_counter()
        pred_yaw, pred_pitch, self._last_pred_angles, self._last_pred_angle_time = self._apply_rate_limit(
            pred_yaw, pred_pitch, self._last_pred_angles, self._last_pred_angle_time
        )
        self._last_pred_rate_limit_ms = (time.perf_counter() - pred_rate_start) * 1000.0
        fire_start = time.perf_counter()
        pred_fire_cmd = self._decide_fire_cmd(has_prediction=True)
        self._last_fire_decide_ms = (time.perf_counter() - fire_start) * 1000.0
        pred_packet_start = time.perf_counter()
        self._latest_prediction_packet = self._build_packet(
            pred_yaw, pred_pitch, pred_x_center, pred_y_center, True, fire_cmd=pred_fire_cmd
        )
        self._last_pred_packet_ms = (time.perf_counter() - pred_packet_start) * 1000.0
        self._latest_prediction_bbox = comp_bbox
        self._latest_prediction_info = (pred_yaw, pred_pitch, pred_x_center, pred_y_center, pred_source)
        self._last_update_ms = (time.perf_counter() - start_time) * 1000.0

    def _tick_fps(self):
        self._fps_count += 1
        now = time.time()
        elapsed = now - self._fps_last_time
        if elapsed >= 1.0:
            self._fps_value = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_last_time = now

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        drawn, draw_ms = self._render_overlay_snapshot(self._build_display_snapshot(frame))
        self._last_draw_ms = draw_ms
        return drawn

    def run(self) -> int:
        if not self._open_camera():
            return 1
        if not self._open_serial():
            # Allow running without serial for testing.
            print("⚠️ 串口未打开，继续运行但不发送数据")
            self.serial = None
        self._log_perf_profile_hint()
        self._start_ec_feedback_worker()
        self._start_pred_worker()
        self._start_display_worker()
        self._start_send_worker()

        win_start = time.time()
        last_tx_count = 0
        last_tx_bytes = 0

        print("🚀 Aim scheduler 启动")
        print("   仅发送预测框数据，锁定以预测框中心为准")
        print(f"   发送频率: {self.send_rate:.2f} Hz")
        print("   每周期发送一帧（预测）")
        print(
            f"   开火门控: conf>={self.fire_confidence_threshold:.2f}, "
            f"force={self.fire_force_interval_s:.2f}s（仅预测可用时）"
        )
        print("   按 Ctrl+C 退出")
        if self.max_yaw_rate > 0 or self.max_pitch_rate > 0:
            print(
                f"   角速度限制: yaw={self.max_yaw_rate:.1f} deg/s, "
                f"pitch={self.max_pitch_rate:.1f} deg/s"
            )
        if self.invert_yaw or self.invert_pitch:
            print(f"   角度翻转: yaw={self.invert_yaw}, pitch={self.invert_pitch}")
        if self._target_color_class_ids is not None:
            if self.target_class_ids is not None:
                print(f"   类别过滤: allow={list(self._target_color_class_ids)}")
            else:
                print(f"   颜色过滤: {self.target_color} allow={list(self._target_color_class_ids)}")
        if self.exclude_class_ids is not None:
            print(f"   类别黑名单: exclude={list(self.exclude_class_ids)}")
        if self.enable_ec_feedback:
            img_comp = "OFF" if self.disable_image_time_comp_with_feedback else "ON"
            print(
                f"   EC反馈: ON t0={self.ec_t0_s * 1000.0:.1f}ms "
                f"add={self.ec_additional_predict_time_s * 1000.0:.1f}ms "
                f"invYaw={self.ec_feedback_invert_yaw} invPitch={self.ec_feedback_invert_pitch} "
                f"imgComp={img_comp}"
            )
        else:
            print("   EC反馈: OFF")
        if self.spin_aware:
            print(
                f"   小陀螺感知: ON enter={self.spin_enter_threshold:.2f} "
                f"exit={self.spin_exit_threshold:.2f} "
                f"yawBias={self.spin_yaw_reverse_bias_deg:.2f} "
                f"dirLock(conf={self.spin_yaw_dir_lock_min_conf:.2f},"
                f"rate={self.spin_yaw_dir_min_rate_dps:.1f},"
                f"score={self.spin_yaw_dir_lock_threshold:.1f}) "
                f"dirSwitch(conf={self.spin_yaw_dir_switch_min_conf:.2f},"
                f"rate={self.spin_yaw_dir_switch_min_rate_dps:.1f},"
                f"score={self.spin_yaw_dir_switch_threshold:.1f})"
            )
        else:
            print("   小陀螺感知: OFF")
        if self.show_window:
            print(
                f"   显示窗口已开启，独立线程限频到 {self.display_max_fps:.1f} Hz，按 q 退出"
            )
        if self.record_video:
            print(
                f"   比赛录制: ON -> {self._record_path_resolved} "
                f"({self.record_fps:.1f} FPS, {self.record_fourcc})"
            )

        try:
            while not self._stop_requested.is_set():
                loop_start = time.perf_counter()
                read_start = time.perf_counter()
                frame = self._read_frame()
                self._last_read_ms = (time.perf_counter() - read_start) * 1000.0
                if frame is None:
                    time.sleep(0.005)
                    continue
                self._update_frame_period_estimate(time.time())
                self._frame_id += 1
                frame_id = self._frame_id
                if self.swap_rb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self._write_record_frame(frame)
                self._tick_fps()

                yolo_start = time.perf_counter()
                detections = self.detector.detect_objects(frame, classes=self._detector_class_ids)
                self._last_yolo_ms = (time.perf_counter() - yolo_start) * 1000.0
                self._update_detector_runtime_stats()
                select_start = time.perf_counter()
                detection = self._select_best_detection(detections)
                self._last_select_ms = (time.perf_counter() - select_start) * 1000.0
                self._update_packets(frame, detection, frame_id)
                self._publish_send_snapshot()
                if self.show_window:
                    self._publish_display_snapshot(frame)
                else:
                    self._last_draw_ms = 0.0
                    self._last_present_ms = 0.0
                    self._last_waitkey_ms = 0.0
                    self._last_imshow_ms = 0.0

                self._last_loop_ms = (time.perf_counter() - loop_start) * 1000.0
                now = time.time()
                if (now - win_start) >= 1.0:
                    if not self.show_window:
                        print(
                            f"FPS: {self._fps_value:.1f} "
                            f"Loop:{self._last_loop_ms:.1f} "
                            f"YOLO:{self._last_yolo_ms:.1f} "
                            f"Pred:{self._last_pred_ms:.1f}"
                        )
                    if self.serial is not None and self.serial.is_open:
                        with self._tx_lock:
                            tx_count = self._tx_total_count
                            tx_bytes = self._tx_total_bytes
                        print(
                            f"TX 速率: {tx_count - last_tx_count} pkt/s, "
                            f"{tx_bytes - last_tx_bytes} B/s"
                        )
                        last_tx_count = tx_count
                        last_tx_bytes = tx_bytes
                    if not self.show_window and self.enable_ec_feedback:
                        ec = self._get_ec_feedback()
                        if ec is not None and abs(now - float(ec[-1])) <= 0.2:
                            yaw_deg, pitch_deg, yaw_rate_dps, pitch_rate_dps, mode, shoot, _ = ec
                            print(
                                f"EC: yaw={float(yaw_deg):.1f} pitch={float(pitch_deg):.1f} "
                                f"yawV={float(yaw_rate_dps):.1f} pitchV={float(pitch_rate_dps):.1f} "
                                f"mode={mode} shoot={shoot}"
                            )
                    win_start = now
                self._record_perf_sample(detection is not None)
                self._maybe_log_perf(now)

        except KeyboardInterrupt:
            print("\n⏹️ 用户中断，正在退出...")
        finally:
            self._stop_requested.set()
            self._stop_display_worker()
            self._stop_send_worker()
            self._stop_pred_worker()
            self._stop_ec_feedback_worker()
            if self.show_window:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            self._close_resources()
        return 0


__all__ = [
    "AimPipeline",
    "DEFAULT_YOLO_MODEL",
    "DEFAULT_COORD_MODEL",
    "DEFAULT_DAHENG_CONFIG",
    "SERIAL_PORT_DEFAULT",
    "SERIAL_BAUD_DEFAULT",
    "TargetGeometry",
]
