import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multimodal_viewer import (
    RegionSpec,
    _display_range,
    load_manifest,
    side_by_side_offsets,
    specs_from_arguments,
)


class MultimodalViewerTests(unittest.TestCase):
    def test_side_by_side_offsets_prevent_x_overlap(self):
        bounds = [
            ("CT", np.asarray([[10.0, 0.0, 0.0], [30.0, 5.0, 5.0]])),
            ("MRI", np.asarray([[-5.0, 0.0, 0.0], [5.0, 5.0, 5.0]])),
        ]
        offsets = side_by_side_offsets(bounds, gap=4.0)
        ct = bounds[0][1][:, 0] + offsets["CT"][0]
        mr = bounds[1][1][:, 0] + offsets["MRI"][0]
        self.assertAlmostEqual(mr[0] - ct[1], 4.0)

    def test_named_surface_joins_matching_volume_region(self):
        args = argparse.Namespace(
            ct=["CT=scan.nii.gz"],
            mr=["MRI=scan.mha"],
            surface=["CT=lobes.stl", "MRI=outline.ply", "airway.stl"],
        )
        specs = specs_from_arguments(args)
        self.assertEqual([spec.name for spec in specs], ["CT", "MRI", "airway"])
        self.assertEqual(specs[0].surface_paths[0].name, "lobes.stl")
        self.assertEqual(specs[1].surface_paths[0].name, "outline.ply")

    def test_manifest_paths_are_relative_to_manifest(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            manifest = root / "demo.json"
            manifest.write_text(
                json.dumps(
                    {
                        "layout": "overlay",
                        "gap": 12,
                        "regions": [
                            {
                                "name": "MRI",
                                "modality": "MR",
                                "volume": "images/mr.nii.gz",
                                "surfaces": ["surfaces/lung.stl"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            specs, layout, gap = load_manifest(manifest)
            self.assertEqual(layout, "overlay")
            self.assertEqual(gap, 12.0)
            self.assertEqual(specs[0].volume_path, root / "images" / "mr.nii.gz")
            self.assertEqual(specs[0].surface_paths, [root / "surfaces" / "lung.stl"])

    def test_ct_and_mr_use_different_default_windows(self):
        values = np.asarray([-1200.0, -500.0, 0.0, 1000.0])
        self.assertEqual(_display_range(values, "CT"), (-1000.0, 400.0))
        mr_low, mr_high = _display_range(values, "MR")
        self.assertGreater(mr_low, -1200.0)
        self.assertLess(mr_high, 1000.0)


if __name__ == "__main__":
    unittest.main()
