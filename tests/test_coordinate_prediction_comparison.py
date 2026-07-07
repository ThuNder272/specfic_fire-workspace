import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coordinate_prediction_comparison import (
    build_baseline_predictions,
    compute_regression_metrics,
)


class _FakeDataset:
    def __init__(self):
        self.input_sequence_length = 3
        self.sequences = [
            {
                "coordinates": [
                    np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float32),
                    np.array([0.20, 0.30, 0.40, 0.50], dtype=np.float32),
                    np.array([0.30, 0.40, 0.50, 0.60], dtype=np.float32),
                    np.array([0.40, 0.50, 0.60, 0.70], dtype=np.float32),
                ],
                "target_index": 3,
            }
        ]


class CoordinatePredictionComparisonTests(unittest.TestCase):
    def test_compute_metrics_for_perfect_prediction(self):
        targets = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
        predictions = targets.copy()

        metrics = compute_regression_metrics(targets, predictions)

        self.assertEqual(metrics["num_samples"], 2)
        self.assertAlmostEqual(metrics["mse"], 0.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)
        self.assertTrue(math.isinf(metrics["psnr"]))

    def test_build_baselines_last_input_and_linear(self):
        dataset = _FakeDataset()

        targets, baselines = build_baseline_predictions(dataset)

        np.testing.assert_allclose(
            targets,
            np.array([[0.40, 0.50, 0.60, 0.70]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            baselines["Baseline-LastInput"],
            np.array([[0.30, 0.40, 0.50, 0.60]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            baselines["Baseline-LinearExtrapolation"],
            np.array([[0.40, 0.50, 0.60, 0.70]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
