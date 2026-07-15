import unittest

import torch

from utils.training_utils import (
    evenly_spaced_holdout_indices,
    foreground_edge_l1,
    foreground_weighted_l1,
    natural_image_key,
)


class TrainingUtilsTests(unittest.TestCase):
    def test_evenly_spaced_five_percent_holdout(self):
        self.assertEqual(
            evenly_spaced_holdout_indices(240, 0.05),
            {10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230},
        )

    def test_natural_sort_uses_capture_indices(self):
        names = ["frame_10.jpg", "frame_2.jpg", "frame_1.jpg"]
        self.assertEqual(sorted(names, key=natural_image_key), ["frame_1.jpg", "frame_2.jpg", "frame_10.jpg"])

    def test_foreground_weighted_l1_emphasises_foreground(self):
        rendered = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]])
        target = torch.zeros_like(rendered)
        mask = torch.tensor([[[1.0, 0.0]]])
        self.assertAlmostEqual(foreground_weighted_l1(rendered, target, mask, 3.0).item(), 0.8, places=6)

    def test_foreground_edge_l1_detects_a_blurred_bts_edge(self):
        rendered = torch.tensor([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
        target = torch.tensor([[[0.0, 1.0, 1.0]], [[0.0, 1.0, 1.0]], [[0.0, 1.0, 1.0]]])
        mask = torch.tensor([[[0.0, 1.0, 1.0]]])
        self.assertAlmostEqual(foreground_edge_l1(rendered, target, mask).item(), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
