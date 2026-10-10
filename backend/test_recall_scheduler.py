import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recall_scheduler import Candidate, adaptive_soft_nms, map_tile_candidate, staged_recall_detect, validate_high_risk_candidates


class RecallSchedulerTests(unittest.TestCase):
    def test_soft_nms_keeps_overlapping_small_instances(self) -> None:
        candidates = [
            Candidate((10, 10, 30, 30), 0.91, "tomato"),
            Candidate((14, 12, 34, 32), 0.82, "tomato"),
        ]
        result = adaptive_soft_nms(candidates, (100, 100), min_score=0.03, max_count=10)
        self.assertEqual(2, len(result))

    def test_tile_mapping_restores_global_coordinates(self) -> None:
        mapped = map_tile_candidate(Candidate((4, 8, 24, 30), 0.6, "car", 3), 100, 200, "1-1")
        self.assertEqual((104, 208, 124, 230), mapped.bbox)
        self.assertEqual("1-1", mapped.tile_id)

    def test_stage_two_is_label_driven(self) -> None:
        calls: list[str] = []

        def detector(_: np.ndarray, prompt: str) -> list[Candidate]:
            calls.append(prompt)
            return [Candidate((1, 1, 20, 20), 0.8, "car")] if prompt == "car." else []

        raw, stage_one = staged_recall_detect(
            np.zeros((64, 64, 3), dtype=np.uint8), ("car.",), detector, full_mode=False, enable_tiles=False,
        )
        self.assertEqual(1, len(stage_one))
        self.assertEqual(1, len(raw))
        self.assertTrue(any("vehicle rear" in prompt for prompt in calls))

    def test_plate_requires_vehicle_context(self) -> None:
        image = np.full((100, 100, 3), [50, 150, 50], dtype=np.uint8)
        invalid = validate_high_risk_candidates(image, [Candidate((20, 20, 60, 28), 0.9, "license plate")])
        self.assertEqual([], invalid)
        valid = validate_high_risk_candidates(image, [
            Candidate((10, 10, 90, 80), 0.9, "car"),
            Candidate((34, 50, 54, 56), 0.9, "license plate"),
        ])
        self.assertEqual({"car", "license plate"}, {item.label for item in valid})

    def test_large_street_regions_cannot_be_manhole_or_air_conditioner(self) -> None:
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = validate_high_risk_candidates(image, [
            Candidate((5, 80, 195, 195), 0.9, "manhole cover"),
            Candidate((5, 5, 180, 150), 0.9, "air conditioner"),
        ])
        self.assertEqual([], result)

    def test_wide_red_sign_cannot_be_tomato(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[30:60, 10:90] = [220, 40, 30]
        result = validate_high_risk_candidates(image, [Candidate((10, 30, 90, 60), 0.9, "tomato")])
        self.assertEqual([], result)


if __name__ == "__main__":
    unittest.main()
