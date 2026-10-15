import tempfile
import unittest
from pathlib import Path

import numpy as np

from demo_io import (
    PUBLIC_DATA,
    load_triangle_surface,
    prepare_ventilation_animation,
    require_private_data,
    scale_transform_about_centre,
    validate_public_bundle,
)
from multimodal_viewer import RegionSpec, load_medical_volume


class DemoDataTests(unittest.TestCase):
    def test_public_bundle_is_complete_and_readable(self):
        summary = validate_public_bundle()
        self.assertGreater(summary["image_voxels"], 0)
        self.assertGreater(summary["tree_nodes"], 0)
        self.assertGreater(summary["growth_frames"], 0)
        self.assertGreater(summary["ventilation_frames"], 0)
        self.assertGreater(summary["surface_faces"], 0)

    def test_public_ply_loads_without_vtk(self):
        vertices, faces = load_triangle_surface(PUBLIC_DATA / "LUL.ply")
        self.assertEqual(vertices.shape[1], 3)
        self.assertEqual(faces.shape[1], 3)

    def test_public_npz_loads_as_ct_without_simpleitk(self):
        volume = load_medical_volume(
            RegionSpec(
                "Public CT",
                modality="CT",
                volume_path=PUBLIC_DATA / "raw_3D_RAS.npz",
            )
        )
        self.assertEqual(volume.values.ndim, 3)
        self.assertEqual(volume.spacing.shape, (3,))
        self.assertEqual(volume.transform[2, 2], -1.0)

    def test_ventilation_emphasis_preserves_totals_and_expands_full_range(self):
        values = np.asarray([[2.0, 4.0], [3.0, 5.0], [4.0, 6.0]])
        emphasized, totals, scales = prepare_ventilation_animation(values)
        np.testing.assert_allclose(totals, (6.0, 8.0, 10.0))
        np.testing.assert_allclose(scales, (0.90, 1.00, 1.10))
        np.testing.assert_allclose(emphasized[2] - emphasized[1], (3.0, 3.0))

    def test_ventilation_emphasis_rejects_invalid_bounds(self):
        with self.assertRaisesRegex(ValueError, "scale bounds"):
            prepare_ventilation_animation(np.ones((2, 2)), scale_min=1.1, scale_max=0.9)

    def test_tissue_and_airway_can_share_one_breathing_transform(self):
        centre = np.asarray([1.0, 2.0, 3.0])
        tissue = np.asarray([[2.0, 4.0, 6.0]])
        airway = np.asarray([[0.0, 0.0, 0.0]])
        transform = scale_transform_about_centre(centre, 1.1)

        def apply(points):
            homogeneous = np.column_stack((points, np.ones(len(points))))
            return (homogeneous @ transform.T)[:, :3]

        np.testing.assert_allclose(
            apply(tissue),
            [[2.1, 4.2, 6.3]],
        )
        np.testing.assert_allclose(
            apply(airway),
            [[-0.1, -0.2, -0.3]],
        )

    def test_missing_private_data_has_actionable_message(self):
        with self.assertRaisesRegex(SystemExit, "private ps_demo_data"):
            require_private_data("file-that-is-not-public.example")

    def test_simpleitk_volume_loader_preserves_geometry(self):
        import SimpleITK as sitk

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "mr.mha"
            image = sitk.GetImageFromArray(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
            image.SetSpacing((1.5, 2.0, 2.5))
            image.SetOrigin((10.0, 20.0, 30.0))
            sitk.WriteImage(image, str(path))
            volume = load_medical_volume(
                RegionSpec("MRI", modality="MR", volume_path=path)
            )
        self.assertEqual(volume.values.shape, (4, 3, 2))
        np.testing.assert_allclose(volume.spacing, (1.5, 2.0, 2.5))
        np.testing.assert_allclose(volume.transform[:3, 3], (10.0, 20.0, 30.0))


if __name__ == "__main__":
    unittest.main()
