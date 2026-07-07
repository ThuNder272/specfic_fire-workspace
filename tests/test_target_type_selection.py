import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camera_adaptation.pnp_solver import (
    TargetGeometry,
    choose_target_type_by_class,
    choose_target_type_by_detection,
    get_rm4pt_class_name,
)


class TargetTypeSelectionTest(unittest.TestCase):
    def test_known_rm4pt_class_names(self):
        self.assertEqual(get_rm4pt_class_name(1), "B1")
        self.assertEqual(get_rm4pt_class_name(10), "R1")
        self.assertEqual(get_rm4pt_class_name(17), "RBb")

    def test_big_armor_is_selected_for_class_1_and_10(self):
        self.assertEqual(
            choose_target_type_by_class(class_id=1),
            TargetGeometry.ARMOR_BIG,
        )
        self.assertEqual(
            choose_target_type_by_class(class_id=10),
            TargetGeometry.ARMOR_BIG,
        )

    def test_other_known_classes_default_to_small_armor(self):
        self.assertEqual(
            choose_target_type_by_class(class_id=2),
            TargetGeometry.ARMOR_SMALL,
        )
        self.assertEqual(
            choose_target_type_by_class(class_id=15),
            TargetGeometry.ARMOR_SMALL,
        )

    def test_class_name_fallback_matches_rm4pt_mapping(self):
        self.assertEqual(
            choose_target_type_by_class(class_name="B1"),
            TargetGeometry.ARMOR_BIG,
        )
        self.assertEqual(
            choose_target_type_by_class(class_name="R4"),
            TargetGeometry.ARMOR_SMALL,
        )

    def test_unknown_class_falls_back_to_default_type(self):
        self.assertEqual(
            choose_target_type_by_class(
                class_name="person",
                default_type=TargetGeometry.ARMOR_BIG,
            ),
            TargetGeometry.ARMOR_BIG,
        )

    def test_detection_wrapper_uses_class_and_class_name(self):
        self.assertEqual(
            choose_target_type_by_detection({"class": 1, "class_name": "B1"}),
            TargetGeometry.ARMOR_BIG,
        )
        self.assertEqual(
            choose_target_type_by_detection({"class": 13, "class_name": "R4"}),
            TargetGeometry.ARMOR_SMALL,
        )


if __name__ == "__main__":
    unittest.main()
