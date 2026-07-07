import sys
import types
import unittest
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ultralytics = types.ModuleType("ultralytics")


class _DummyYOLO:
    def __init__(self, *args, **kwargs):
        pass


ultralytics.YOLO = _DummyYOLO  # type: ignore[attr-defined]
sys.modules["ultralytics"] = ultralytics

serial = types.ModuleType("serial")
serial.Serial = object  # type: ignore[attr-defined]
sys.modules["serial"] = serial

torch = types.ModuleType("torch")
torch.cuda = types.SimpleNamespace(is_available=lambda: False, synchronize=lambda: None)
sys.modules["torch"] = torch

coordinate_prediction_model = types.ModuleType("coordinate_prediction_model")
class _DummyCoordinatePredictionModel:
    def __init__(self, *args, **kwargs):
        pass
coordinate_prediction_model.CoordinatePredictionModel = _DummyCoordinatePredictionModel
sys.modules["coordinate_prediction_model"] = coordinate_prediction_model

industrial_camera_processor = types.ModuleType("camera_adaptation.industrial_camera_processor")
class _DummyIndustrialCameraProcessor:
    def __init__(self, *args, **kwargs):
        pass
industrial_camera_processor.IndustrialCameraProcessor = _DummyIndustrialCameraProcessor
sys.modules["camera_adaptation.industrial_camera_processor"] = industrial_camera_processor

from camera_adaptation.aim_pipeline import AimPipeline, MotionObservation


def _make_pipeline() -> AimPipeline:
    pipeline = AimPipeline.__new__(AimPipeline)
    pipeline.spin_aware = True
    pipeline.spin_enter_threshold = 0.65
    pipeline.spin_exit_threshold = 0.45
    pipeline.invert_yaw = False
    pipeline.enable_ec_feedback = False
    pipeline.disable_image_time_comp_with_feedback = False
    pipeline._get_ec_feedback = lambda: None
    pipeline.target_color = None
    pipeline.target_class_ids = None
    pipeline.exclude_class_ids = None
    pipeline._target_color_class_ids = None
    pipeline._excluded_class_ids = set()
    pipeline._detector_class_ids = None

    pipeline._motion_history = deque(maxlen=32)
    pipeline._spin_confidence = 0.0
    pipeline._spin_active = False
    pipeline._spin_yaw_direction_score = 0.0
    pipeline._spin_yaw_direction_locked = 0
    pipeline._spin_yaw_fake_rate_dps = 0.0
    pipeline._last_spin_yaw_bias_deg = 0.0
    pipeline.spin_yaw_reverse_bias_deg = 2.0
    pipeline.spin_yaw_dir_lock_min_conf = 0.75
    pipeline.spin_yaw_dir_min_rate_dps = 10.0
    pipeline.spin_yaw_dir_lock_threshold = 4.0
    pipeline.spin_yaw_dir_switch_min_conf = 0.85
    pipeline.spin_yaw_dir_switch_min_rate_dps = 20.0
    pipeline.spin_yaw_dir_switch_threshold = 7.0
    pipeline._last_valid_motion_rvec_ts = None
    pipeline._last_rel_angles_for_rate = None
    pipeline._last_rel_angle_time_for_rate = None

    pipeline._target_rate_fast_alpha = 0.25
    pipeline._target_rate_slow_alpha = 0.08
    pipeline._target_yaw_rate_fast_dps = 0.0
    pipeline._target_pitch_rate_fast_dps = 0.0
    pipeline._target_yaw_rate_slow_dps = 0.0
    pipeline._target_pitch_rate_slow_dps = 0.0
    pipeline._target_yaw_rate_effective_dps = 0.0
    pipeline._target_pitch_rate_effective_dps = 0.0
    pipeline._target_yaw_rate_dps = 0.0
    pipeline._target_pitch_rate_dps = 0.0

    pipeline._last_distance_m = None
    pipeline._last_pnp_tvec = None
    pipeline._last_pnp_rvec = None
    pipeline._last_image_time_comp_scale = 1.0
    pipeline._pred_velocity = (0.0, 0.0)
    pipeline._compute_time_compensation = lambda: (0.2, 0.1, 6)

    pipeline._last_fire_ts = 0.0
    pipeline._last_fire_reason = "INIT"
    pipeline.fire_confidence_threshold = 0.6
    pipeline.fire_force_interval_s = 0.0
    pipeline._latest_detection_conf = 0.9

    pipeline.coord_buffer = np.zeros((4, 4), dtype=np.float32)
    pipeline._buffer_idx = 2
    pipeline._buffer_len = 2
    pipeline._last_buffer_len = 2
    pipeline._pred_ready = True
    pipeline._last_pred_center = (12.0, 34.0)
    pipeline._last_pred_size = (20.0, 10.0)
    pipeline._last_pred_time = 1.0
    pipeline._last_pred_lag = 1
    pipeline.lost_threshold = 10

    pipeline._last_raw_distance_m = None
    pipeline._last_comp_distance_m = None
    pipeline._last_comp_time_s = 0.0
    pipeline._last_extra_time_s = 0.0
    pipeline._last_lead_frames = 0
    pipeline._last_lag_time_s = 0.0
    pipeline._last_lag_frames = 0
    pipeline._last_flight_time_s = None
    pipeline._last_det_ballistic_delta = None
    pipeline._last_ballistic_time_s = None
    pipeline.angle_scale = 100.0
    return pipeline


def _make_rvec_y(yaw_deg: float) -> np.ndarray:
    return np.array([[0.0], [np.radians(yaw_deg)], [0.0]], dtype=np.float32)


def _make_corners(x: float, y: float, w: float, h: float) -> np.ndarray:
    return np.array(
        [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ],
        dtype=np.float32,
    )


class SpinAwareMotionTests(unittest.TestCase):
    def test_linear_translation_keeps_fast_rate_and_large_img_scale(self):
        pipeline = _make_pipeline()

        for idx, now_s in enumerate(np.arange(0.0, 0.30, 0.05)):
            pipeline._last_pnp_rvec = _make_rvec_y(5.0)
            pipeline._last_pnp_tvec = np.array([[0.0], [0.0], [3.0]], dtype=np.float32)
            pipeline._last_distance_m = 3.0
            bbox = (100 + idx * 4, 200, 160 + idx * 4, 240)
            corners = _make_corners(100 + idx * 4, 200, 60, 40)
            AimPipeline._update_motion_state(
                pipeline,
                rel_yaw_deg=float(idx * 2.5),
                rel_pitch_deg=0.0,
                bbox=bbox,
                corners=corners,
                now_s=float(now_s),
            )

        self.assertFalse(pipeline._spin_active)
        diff = abs(pipeline._target_yaw_rate_effective_dps - pipeline._target_yaw_rate_fast_dps)
        self.assertLessEqual(diff, 0.1 * max(1.0, abs(pipeline._target_yaw_rate_fast_dps)))

        pipeline._pred_velocity = (120.0, 0.0)
        AimPipeline._apply_time_compensation(pipeline, (120, 200, 180, 240), (720, 1280, 3))
        self.assertGreater(pipeline._last_image_time_comp_scale, 0.8)

    def test_spin_motion_prefers_slow_rate_and_suppresses_img_comp(self):
        pipeline = _make_pipeline()

        for idx, now_s in enumerate(np.arange(0.0, 0.30, 0.05)):
            pipeline._last_pnp_rvec = _make_rvec_y(idx * 15.0)
            pipeline._last_pnp_tvec = np.array([[0.0], [0.0], [3.0]], dtype=np.float32)
            pipeline._last_distance_m = 3.0
            width = 60 + idx * 6
            bbox = (200, 180, 200 + width, 220)
            corners = _make_corners(200, 180, width, 40)
            AimPipeline._update_motion_state(
                pipeline,
                rel_yaw_deg=float(idx * 8.0),
                rel_pitch_deg=0.0,
                bbox=bbox,
                corners=corners,
                now_s=float(now_s),
            )

        self.assertTrue(pipeline._spin_active)
        self.assertLess(
            abs(pipeline._target_yaw_rate_effective_dps - pipeline._target_yaw_rate_slow_dps),
            abs(pipeline._target_yaw_rate_effective_dps - pipeline._target_yaw_rate_fast_dps),
        )

        pipeline._pred_velocity = (120.0, 0.0)
        original_bbox = (200, 180, 260, 220)
        comp_bbox, comp_applied = AimPipeline._apply_time_compensation(
            pipeline, original_bbox, (720, 1280, 3)
        )
        self.assertEqual(pipeline._last_image_time_comp_scale, 0.0)
        self.assertFalse(comp_applied)
        self.assertEqual(comp_bbox, original_bbox)

    def test_spin_hysteresis_enters_and_exits_cleanly(self):
        pipeline = _make_pipeline()

        for idx, now_s in enumerate(np.arange(0.0, 0.30, 0.05)):
            pipeline._last_pnp_rvec = _make_rvec_y(idx * 15.0)
            pipeline._last_pnp_tvec = np.array([[0.0], [0.0], [3.0]], dtype=np.float32)
            pipeline._last_distance_m = 3.0
            width = 60 + idx * 6
            bbox = (220, 180, 220 + width, 220)
            corners = _make_corners(220, 180, width, 40)
            AimPipeline._update_motion_state(
                pipeline,
                rel_yaw_deg=float(idx * 8.0),
                rel_pitch_deg=0.0,
                bbox=bbox,
                corners=corners,
                now_s=float(now_s),
            )

        self.assertTrue(pipeline._spin_active)

        for idx, now_s in enumerate(np.arange(0.35, 0.75, 0.05)):
            pipeline._last_pnp_rvec = _make_rvec_y(5.0)
            pipeline._last_pnp_tvec = np.array([[0.0], [0.0], [3.0]], dtype=np.float32)
            pipeline._last_distance_m = 3.0
            bbox = (260 + idx * 2, 180, 320 + idx * 2, 220)
            corners = _make_corners(260 + idx * 2, 180, 60, 40)
            AimPipeline._update_motion_state(
                pipeline,
                rel_yaw_deg=float(48.0 + idx * 1.0),
                rel_pitch_deg=0.0,
                bbox=bbox,
                corners=corners,
                now_s=float(now_s),
            )

        self.assertFalse(pipeline._spin_active)
        self.assertLess(pipeline._spin_confidence, 0.45)

    def test_missing_corners_do_not_reenter_bbox_mode(self):
        pipeline = _make_pipeline()
        pipeline._spin_active = True
        pipeline._spin_confidence = 0.9
        pipeline._last_valid_motion_rvec_ts = 0.0

        for idx, now_s in enumerate(np.arange(0.3, 0.75, 0.1)):
            pipeline._last_pnp_rvec = _make_rvec_y(20.0)
            pipeline._last_pnp_tvec = np.array([[0.0], [0.0], [3.0]], dtype=np.float32)
            pipeline._last_distance_m = 3.0
            AimPipeline._update_motion_state(
                pipeline,
                rel_yaw_deg=float(20.0 + idx),
                rel_pitch_deg=0.0,
                bbox=(200, 180, 260, 220),
                corners=None,
                now_s=float(now_s),
            )

        self.assertFalse(pipeline._spin_active)
        self.assertLess(pipeline._spin_confidence, 0.45)

    def test_prediction_fallback_can_fire_during_spin(self):
        pipeline = _make_pipeline()
        pipeline._spin_active = True
        pipeline._spin_confidence = 0.9
        pipeline._latest_detection_conf = 0.95
        pipeline._build_packet = lambda yaw, pitch, x, y, has_target, fire_cmd=0x00: bytes(
            [0xA5, 0x5A, fire_cmd]
        )

        packet, info = AimPipeline._build_prediction_fallback_packet(
            pipeline,
            det_yaw=1.5,
            det_pitch=-0.5,
            x_center=320.0,
            y_center=240.0,
        )

        self.assertEqual(packet[2], 0x01)
        self.assertEqual(info[-1], "DET_SPIN_FALLBACK")
        self.assertTrue(pipeline._last_fire_reason.startswith("CONF"))

    def test_spin_yaw_direction_locks_switches_and_applies_reverse_yaw_bias(self):
        pipeline = _make_pipeline()
        pipeline._spin_active = True
        pipeline._spin_confidence = 0.9

        for _ in range(8):
            pipeline._target_yaw_rate_fast_dps = 60.0
            pipeline._target_yaw_rate_slow_dps = 10.0
            AimPipeline._update_spin_yaw_direction_lock(pipeline)

        self.assertEqual(pipeline._spin_yaw_direction_locked, 1)

        biased_yaw, biased_pitch, tag = AimPipeline._apply_spin_yaw_bias(
            pipeline, yaw_deg=5.0, pitch_deg=-1.5
        )
        self.assertEqual(biased_yaw, 3.0)
        self.assertEqual(biased_pitch, -1.5)
        self.assertEqual(pipeline._last_spin_yaw_bias_deg, -2.0)
        self.assertEqual(tag, "+SPIN(-2.0)")

        pipeline._spin_confidence = 0.95
        for _ in range(20):
            pipeline._target_yaw_rate_fast_dps = -60.0
            pipeline._target_yaw_rate_slow_dps = -10.0
            AimPipeline._update_spin_yaw_direction_lock(pipeline)

        self.assertEqual(pipeline._spin_yaw_direction_locked, -1)

        switched_yaw, switched_pitch, switched_tag = AimPipeline._apply_spin_yaw_bias(
            pipeline, yaw_deg=5.0, pitch_deg=-1.5
        )
        self.assertEqual(switched_yaw, 7.0)
        self.assertEqual(switched_pitch, -1.5)
        self.assertEqual(pipeline._last_spin_yaw_bias_deg, 2.0)
        self.assertEqual(switched_tag, "+SPIN(+2.0)")

    def test_no_prediction_mode_can_fire_from_detection_result(self):
        pipeline = _make_pipeline()
        pipeline.enable_prediction = False
        pipeline.pred_async = False
        pipeline._lost_count = 0
        pipeline.show_window = False
        pipeline.use_corners = False
        pipeline.show_tx = False
        pipeline.target_type = "small"
        pipeline.auto_target_type = False
        pipeline._last_det_angles = None
        pipeline._last_det_angle_time = None
        pipeline._build_packet = lambda yaw, pitch, x, y, has_target, fire_cmd=0x00: bytes(
            [0xA5, 0x5A, fire_cmd]
        )
        pipeline._resolve_angles = (
            lambda bbox, x_center, y_center, frame, label, target_type, corners=None, force_pixel=False: (
                1.5,
                -0.5,
                "PNP",
                None,
                None,
                None,
            )
        )
        pipeline._apply_ballistic_compensation = lambda pitch, source: (pitch, source)
        pipeline._apply_ec_angle_lead = lambda yaw, pitch: (yaw, pitch, None)
        pipeline._apply_rate_limit = (
            lambda yaw, pitch, last_angles, last_time: (yaw, pitch, last_angles, last_time)
        )
        pipeline._update_motion_state = lambda *args, **kwargs: None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detection = {"bbox": (100, 120, 160, 180), "confidence": 0.95, "corners": None}

        AimPipeline._update_packets(pipeline, frame, detection, frame_id=1)

        self.assertEqual(pipeline._latest_detection_packet[2], 0x00)
        self.assertEqual(pipeline._latest_prediction_packet[2], 0x01)
        self.assertEqual(pipeline._latest_prediction_info[-1], "PNP")
        self.assertTrue(pipeline._last_fire_reason.startswith("CONF"))

    def test_target_color_filter_prefers_requested_rm4pt_team(self):
        pipeline = _make_pipeline()
        pipeline.target_color = "blue"
        pipeline._target_color_class_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8)

        best = AimPipeline._select_best_detection(
            pipeline,
            [
                {"confidence": 0.99, "class": 12},
                {"confidence": 0.80, "class": 4},
                {"confidence": 0.75, "class": 1},
            ],
        )

        self.assertIsNotNone(best)
        self.assertEqual(best["class"], 4)

    def test_target_color_filter_rejects_other_team_and_unknown_labels(self):
        pipeline = _make_pipeline()
        pipeline.target_color = "red"
        pipeline._target_color_class_ids = (9, 10, 11, 12, 13, 14, 15, 16, 17)

        self.assertTrue(AimPipeline._target_color_allows_detection(pipeline, {"class": 13}))
        self.assertFalse(AimPipeline._target_color_allows_detection(pipeline, {"class": 4}))
        self.assertFalse(AimPipeline._target_color_allows_detection(pipeline, {"class_name": "R4"}))

        best = AimPipeline._select_best_detection(
            pipeline,
            [
                {"confidence": 0.99, "class": 4},
                {"confidence": 0.98, "class": 26},
            ],
        )

        self.assertIsNone(best)

    def test_target_class_id_filter_accepts_blue2_and_red2(self):
        pipeline = _make_pipeline()
        pipeline.target_color = None
        pipeline.target_class_ids = (2, 11)
        pipeline._target_color_class_ids = (2, 11)
        pipeline._excluded_class_ids = set()
        pipeline._detector_class_ids = (2, 11)

        self.assertTrue(AimPipeline._target_color_allows_detection(pipeline, {"class": 2}))
        self.assertTrue(AimPipeline._target_color_allows_detection(pipeline, {"class": 11}))
        self.assertFalse(AimPipeline._target_color_allows_detection(pipeline, {"class": 4}))
        self.assertFalse(AimPipeline._target_color_allows_detection(pipeline, {"class_name": "B2"}))

    def test_target_color_blacklist_excludes_blue2_without_whitelist(self):
        pipeline = _make_pipeline()
        pipeline.target_color = "blue"
        pipeline.target_class_ids = None
        pipeline.exclude_class_ids = (2,)
        pipeline._target_color_class_ids = (0, 1, 3, 4, 5, 6, 7, 8)
        pipeline._excluded_class_ids = {2}
        pipeline._detector_class_ids = (0, 1, 3, 4, 5, 6, 7, 8)

        self.assertFalse(AimPipeline._target_color_allows_detection(pipeline, {"class": 2}))
        self.assertTrue(AimPipeline._target_color_allows_detection(pipeline, {"class": 4}))

        best = AimPipeline._select_best_detection(
            pipeline,
            [
                {"confidence": 0.99, "class": 2},
                {"confidence": 0.80, "class": 4},
            ],
        )

        self.assertIsNotNone(best)
        self.assertEqual(best["class"], 4)

    def test_reset_prediction_buffers_clears_spin_and_pose_state(self):
        pipeline = _make_pipeline()
        pipeline.coord_buffer.fill(1.0)
        pipeline._motion_history.append(
            MotionObservation(
                ts=1.0,
                yaw_deg=10.0,
                pitch_deg=0.0,
                quad_area_px=2400.0,
                quad_aspect=1.5,
                distance_m=3.0,
                normal_yaw_deg=30.0,
            )
        )
        pipeline._spin_confidence = 0.9
        pipeline._spin_active = True
        pipeline._spin_yaw_direction_score = 7.5
        pipeline._spin_yaw_direction_locked = -1
        pipeline._spin_yaw_fake_rate_dps = -42.0
        pipeline._last_spin_yaw_bias_deg = 2.0
        pipeline._last_distance_m = 3.0
        pipeline._last_pnp_tvec = np.ones((3, 1), dtype=np.float32)
        pipeline._last_pnp_rvec = np.ones((3, 1), dtype=np.float32)
        pipeline._target_yaw_rate_fast_dps = 10.0
        pipeline._target_yaw_rate_slow_dps = 5.0
        pipeline._target_yaw_rate_effective_dps = 6.0

        AimPipeline._reset_prediction_buffers(pipeline)

        self.assertEqual(len(pipeline._motion_history), 0)
        self.assertEqual(pipeline._spin_confidence, 0.0)
        self.assertFalse(pipeline._spin_active)
        self.assertEqual(pipeline._spin_yaw_direction_score, 0.0)
        self.assertEqual(pipeline._spin_yaw_direction_locked, 0)
        self.assertEqual(pipeline._spin_yaw_fake_rate_dps, 0.0)
        self.assertEqual(pipeline._last_spin_yaw_bias_deg, 0.0)
        self.assertIsNone(pipeline._last_distance_m)
        self.assertIsNone(pipeline._last_pnp_tvec)
        self.assertIsNone(pipeline._last_pnp_rvec)
        self.assertEqual(pipeline._target_yaw_rate_fast_dps, 0.0)
        self.assertEqual(pipeline._target_yaw_rate_slow_dps, 0.0)
        self.assertEqual(pipeline._target_yaw_rate_effective_dps, 0.0)
        self.assertTrue(np.allclose(pipeline.coord_buffer, 0.0))


if __name__ == "__main__":
    unittest.main()
