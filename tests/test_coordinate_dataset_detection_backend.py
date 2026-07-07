import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = types.ModuleType("torch")
torch.Tensor = object  # type: ignore[attr-defined]
torch.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules["torch"] = torch

nn = types.ModuleType("torch.nn")
nn.Module = object  # type: ignore[attr-defined]
torch.nn = nn  # type: ignore[attr-defined]
sys.modules["torch.nn"] = nn

lstm_module = types.ModuleType("lstm_module")
class _DummyLSTMFeatureExtractor:
    def __init__(self, *args, **kwargs):
        pass
lstm_module.LSTMFeatureExtractor = _DummyLSTMFeatureExtractor
sys.modules["lstm_module"] = lstm_module

kalman_filter = types.ModuleType("kalman_filter")
class _DummyAdaptiveKalmanFilter:
    def __init__(self, *args, **kwargs):
        pass
kalman_filter.AdaptiveKalmanFilter = _DummyAdaptiveKalmanFilter
sys.modules["kalman_filter"] = kalman_filter

fusion_module = types.ModuleType("fusion_module")
class _DummyFusionModule:
    def __init__(self, *args, **kwargs):
        pass
fusion_module.FusionModule = _DummyFusionModule
sys.modules["fusion_module"] = fusion_module

rm4pt_runtime = types.ModuleType("camera_adaptation.rm4pt_runtime")
rm4pt_runtime.Legacy4PointDetector = object  # type: ignore[attr-defined]
rm4pt_runtime.Legacy4PointTensorRTDetector = object  # type: ignore[attr-defined]
rm4pt_runtime.looks_like_legacy_rm4pt_engine = lambda path: False  # type: ignore[attr-defined]
rm4pt_runtime.looks_like_legacy_rm4pt_weight = lambda path: False  # type: ignore[attr-defined]
sys.modules["camera_adaptation.rm4pt_runtime"] = rm4pt_runtime

import coordinate_prediction_model as coordinate_prediction_model_module
from coordinate_prediction_model import CoordinateDataset


class CoordinateDatasetDetectionBackendTests(unittest.TestCase):
    def _make_dataset(self) -> CoordinateDataset:
        dataset = CoordinateDataset.__new__(CoordinateDataset)
        dataset.yolo_model_path = "/tmp/model.pt"
        dataset.frame_shape = (480, 640, 3)
        dataset.target_type = "armor_small"
        dataset.auto_target_type = True
        dataset.max_pnp_error = 5.0
        dataset._intrinsics_base = object()
        dataset._intrinsics_scaled = None
        dataset._intrinsics_scaled_shape = None
        dataset.scale_intrinsics = False
        dataset.default_fps = 60.0
        dataset.video_fps = 120.0
        dataset.bullet_speed_mps = 20.0
        dataset.system_latency_s = 0.05
        dataset.min_lead_frames = 1
        dataset.max_lead_frames = 20
        dataset.default_lead_frames = 8
        return dataset

    def test_choose_detection_backend_uses_rm4pt_sidecar_for_engine(self):
        dataset = CoordinateDataset.__new__(CoordinateDataset)
        dataset.yolo_model_path = "/tmp/model.engine"

        with patch.object(
            coordinate_prediction_model_module,
            "looks_like_legacy_rm4pt_engine",
            return_value=False,
        ), patch.object(
            coordinate_prediction_model_module,
            "looks_like_legacy_rm4pt_weight",
            side_effect=lambda path: path.endswith("model.pt"),
        ):
            self.assertEqual(dataset._choose_detection_backend(), "rm4pt")

    def test_build_detection_record_keeps_bbox_class_and_corners(self):
        dataset = self._make_dataset()
        corners = np.array(
            [[100, 120], [200, 120], [200, 220], [100, 220]],
            dtype=np.float32,
        )
        detection = dataset._build_detection_record(
            dataset.frame_shape,
            bbox=(100, 120, 200, 220),
            confidence=0.9,
            class_id=1,
            class_name="B1",
            corners=corners,
            backend="rm4pt",
        )

        self.assertIsNotNone(detection)
        self.assertEqual(detection["class"], 1)
        self.assertEqual(detection["class_name"], "B1")
        self.assertEqual(detection["backend"], "rm4pt")
        np.testing.assert_array_equal(detection["corners"], corners)
        self.assertAlmostEqual(float(detection["normalized_bbox"][0]), 150.0 / 640.0)
        self.assertAlmostEqual(float(detection["normalized_bbox"][1]), 170.0 / 480.0)

    def test_estimate_distance_prefers_corners_before_bbox(self):
        dataset = self._make_dataset()
        detection = {
            "corners": np.array([[1, 1], [2, 1], [2, 2], [1, 2]], dtype=np.float32),
            "bbox": [1, 1, 2, 2],
            "class": 1,
            "class_name": "B1",
        }

        with patch.object(
            dataset,
            "_estimate_distance_from_image_points",
            return_value=3.5,
        ) as image_points_mock, patch.object(
            dataset,
            "_estimate_distance_from_bbox",
            return_value=1.0,
        ) as bbox_mock:
            self.assertEqual(dataset._estimate_distance_from_detection(detection), 3.5)
            image_points_mock.assert_called_once()
            bbox_mock.assert_not_called()

    def test_estimate_distance_falls_back_to_bbox_when_corners_fail(self):
        dataset = self._make_dataset()
        detection = {
            "corners": np.array([[1, 1], [2, 1], [2, 2], [1, 2]], dtype=np.float32),
            "bbox": [10, 20, 40, 60],
            "class": 10,
            "class_name": "R1",
        }

        with patch.object(
            dataset,
            "_estimate_distance_from_image_points",
            return_value=None,
        ) as image_points_mock, patch.object(
            dataset,
            "_estimate_distance_from_bbox",
            return_value=2.25,
        ) as bbox_mock:
            self.assertEqual(dataset._estimate_distance_from_detection(detection), 2.25)
            image_points_mock.assert_called_once()
            bbox_mock.assert_called_once_with(
                [10, 20, 40, 60],
                target_class=10,
                target_class_name="R1",
            )


if __name__ == "__main__":
    unittest.main()
