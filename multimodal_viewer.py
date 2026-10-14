"""Interactive CT/MR and surface comparison demo for Polyscope.

Each named region owns one optional medical image and any number of STL/PLY
surfaces. Regions are registered with unique Polyscope structure names and can
be shown in their native coordinates or arranged side by side.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from demo_io import PUBLIC_DATA, load_triangle_surface, validate_public_bundle


SURFACE_SUFFIXES = {".stl", ".ply"}
LAYOUTS = ("side-by-side", "overlay")


@dataclass
class RegionSpec:
    name: str
    modality: str | None = None
    volume_path: Path | None = None
    surface_paths: list[Path] = field(default_factory=list)


@dataclass
class MedicalVolume:
    name: str
    modality: str
    path: Path
    values: np.ndarray
    spacing: np.ndarray
    transform: np.ndarray
    display_range: tuple[float, float]

    @property
    def world_axes(self) -> np.ndarray:
        axes = np.asarray(self.transform[:3, :3], dtype=float)
        norms = np.linalg.norm(axes, axis=0)
        norms[norms == 0.0] = 1.0
        return axes / norms


@dataclass
class SliceView:
    name: str
    axis: int
    index: int
    structure: object
    visible: bool = False


@dataclass
class SurfaceView:
    path: Path
    structure: object
    visible: bool = True
    opacity: float = 0.55


@dataclass
class RegionView:
    spec: RegionSpec
    bounds: np.ndarray
    volume: MedicalVolume | None = None
    slices: list[SliceView] = field(default_factory=list)
    surfaces: list[SurfaceView] = field(default_factory=list)
    volume_enabled: bool = True
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))


def _named_path(value: str, fallback: str) -> tuple[str, Path]:
    """Parse ``NAME=PATH`` while still accepting an ordinary path."""

    if "=" in value:
        name, raw_path = value.split("=", 1)
        if name.strip() and raw_path.strip():
            return name.strip(), Path(raw_path).expanduser()
    path = Path(value).expanduser()
    return fallback or path.stem, path


def _unique_name(requested: str, used: set[str]) -> str:
    name = requested.strip() or "region"
    if name not in used:
        used.add(name)
        return name
    index = 2
    while f"{name} ({index})" in used:
        index += 1
    result = f"{name} ({index})"
    used.add(result)
    return result


def _normalise_specs(specs: Iterable[RegionSpec]) -> list[RegionSpec]:
    used: set[str] = set()
    result: list[RegionSpec] = []
    for spec in specs:
        result.append(
            RegionSpec(
                name=_unique_name(spec.name, used),
                modality=spec.modality.upper() if spec.modality else None,
                volume_path=spec.volume_path,
                surface_paths=list(spec.surface_paths),
            )
        )
    return result


def load_manifest(path: str | Path) -> tuple[list[RegionSpec], str, float | None]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    layout = payload.get("layout", "side-by-side")
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {', '.join(LAYOUTS)}")
    gap = payload.get("gap")
    base = source.parent
    specs: list[RegionSpec] = []
    for item in payload.get("regions", []):
        volume = item.get("volume")
        surfaces = item.get("surfaces", [])
        specs.append(
            RegionSpec(
                name=str(item.get("name") or item.get("modality") or "region"),
                modality=str(item["modality"]).upper() if item.get("modality") else None,
                volume_path=(base / volume).resolve() if volume else None,
                surface_paths=[(base / value).resolve() for value in surfaces],
            )
        )
    if not specs:
        raise ValueError("The manifest does not contain any regions")
    return _normalise_specs(specs), layout, float(gap) if gap is not None else None


def specs_from_arguments(args: argparse.Namespace) -> list[RegionSpec]:
    specs: list[RegionSpec] = []
    region_by_name: dict[str, RegionSpec] = {}

    def add_volume(raw: str, modality: str, ordinal: int) -> None:
        fallback = modality if ordinal == 1 else f"{modality} {ordinal}"
        name, path = _named_path(raw, fallback)
        spec = RegionSpec(name=name, modality=modality, volume_path=path.resolve())
        specs.append(spec)
        region_by_name.setdefault(name, spec)

    for index, raw in enumerate(args.ct, start=1):
        add_volume(raw, "CT", index)
    for index, raw in enumerate(args.mr, start=1):
        add_volume(raw, "MR", index)
    for raw in args.surface:
        name, path = _named_path(raw, "")
        target = region_by_name.get(name)
        if target is None:
            target = RegionSpec(name=name)
            specs.append(target)
            region_by_name[name] = target
        target.surface_paths.append(path.resolve())
    return _normalise_specs(specs)


def _display_range(values: np.ndarray, modality: str) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return (0.0, 1.0)
    if modality.upper() == "CT":
        return (-1000.0, 400.0)
    low, high = np.percentile(finite, (1.0, 99.0))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def load_medical_volume(spec: RegionSpec) -> MedicalVolume:
    if spec.volume_path is None or spec.modality is None:
        raise ValueError(f"Region {spec.name!r} does not define a medical image")

    source = spec.volume_path
    if source.suffix.lower() == ".npz":
        with np.load(source) as payload:
            if "array" not in payload or "spacing" not in payload:
                raise ValueError(
                    f"NPZ image {source} must contain 'array' and 'spacing' entries"
                )
            values = np.asarray(payload["array"], dtype=float)
            spacing = np.asarray(payload["spacing"], dtype=float)
        if values.ndim != 3 or spacing.shape != (3,):
            raise ValueError(f"Expected a 3D array and three spacing values in {source}")
        # The bundled hologram array is stored as [Z, Y, X], with the patient Z
        # axis pointing in the negative display direction.
        values = values.transpose(2, 1, 0)
        transform = np.eye(4, dtype=float)
        transform[2, 2] = -1.0
        return MedicalVolume(
            name=spec.name,
            modality=spec.modality,
            path=source,
            values=values,
            spacing=spacing,
            transform=transform,
            display_range=_display_range(values, spec.modality),
        )

    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("Medical-image loading requires SimpleITK") from exc

    if source.is_dir():
        series_ids = list(sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(source)) or [])
        if not series_ids:
            raise ValueError(f"No DICOM series found in {source}")
        candidates = [
            (series_id, sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(source), series_id))
            for series_id in series_ids
        ]
        _series_id, filenames = max(candidates, key=lambda item: len(item[1]))
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(filenames)
        image = reader.Execute()
    else:
        image = sitk.ReadImage(str(source))

    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D image for {source}, found {image.GetDimension()}D")
    values = np.asarray(sitk.GetArrayFromImage(image), dtype=float).transpose(2, 1, 0)
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = direction
    transform[:3, 3] = np.asarray(image.GetOrigin(), dtype=float)
    return MedicalVolume(
        name=spec.name,
        modality=spec.modality,
        path=source,
        values=values,
        spacing=spacing,
        transform=transform,
        display_range=_display_range(values, spec.modality),
    )


def load_surface(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    source = Path(path)
    if source.suffix.lower() not in SURFACE_SUFFIXES:
        raise ValueError(f"Expected STL or PLY surface, found {source.name}")
    return load_triangle_surface(source)


def _volume_bounds(volume: MedicalVolume) -> np.ndarray:
    high = (np.asarray(volume.values.shape, dtype=float) - 1.0) * volume.spacing
    corners = np.asarray(
        [[x, y, z, 1.0] for x in (0.0, high[0]) for y in (0.0, high[1]) for z in (0.0, high[2])]
    )
    world = corners @ volume.transform.T
    return np.vstack((world[:, :3].min(axis=0), world[:, :3].max(axis=0)))


def combine_bounds(items: Sequence[np.ndarray]) -> np.ndarray:
    if not items:
        return np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    return np.vstack(
        (
            np.min([item[0] for item in items], axis=0),
            np.max([item[1] for item in items], axis=0),
        )
    )


def side_by_side_offsets(
    named_bounds: Sequence[tuple[str, np.ndarray]], gap: float | None = None
) -> dict[str, np.ndarray]:
    """Return X translations which lay regions out without overlap."""

    if not named_bounds:
        return {}
    widths = [max(float(bounds[1, 0] - bounds[0, 0]), 1.0) for _, bounds in named_bounds]
    spacing = float(gap) if gap is not None else max(widths) * 0.15
    cursor = 0.0
    placements: dict[str, float] = {}
    for (name, bounds), width in zip(named_bounds, widths):
        placements[name] = cursor - float(bounds[0, 0])
        cursor += width + spacing
    layout_min = min(
        float(bounds[0, 0]) + placements[name] for name, bounds in named_bounds
    )
    layout_max = max(
        float(bounds[1, 0]) + placements[name] for name, bounds in named_bounds
    )
    centre = (layout_min + layout_max) * 0.5
    return {
        name: np.asarray([translation - centre, 0.0, 0.0], dtype=float)
        for name, translation in placements.items()
    }


def _display_indices(size: int, maximum: int = 384) -> np.ndarray:
    if size <= maximum:
        return np.arange(size, dtype=int)
    return np.unique(np.linspace(0, size - 1, maximum).round().astype(int))


def _slice_geometry(volume: MedicalVolume, axis: int, index: int):
    plane_axes = [candidate for candidate in range(3) if candidate != axis]
    first_indices = _display_indices(volume.values.shape[plane_axes[0]])
    second_indices = _display_indices(volume.values.shape[plane_axes[1]])
    first, second = np.meshgrid(first_indices, second_indices, indexing="ij")
    vertices = np.zeros((first.size, 3), dtype=float)
    vertices[:, axis] = index * volume.spacing[axis]
    vertices[:, plane_axes[0]] = first.ravel() * volume.spacing[plane_axes[0]]
    vertices[:, plane_axes[1]] = second.ravel() * volume.spacing[plane_axes[1]]

    lookup = [np.full(first.size, index, dtype=int) for _ in range(3)]
    lookup[plane_axes[0]] = first.ravel()
    lookup[plane_axes[1]] = second.ravel()
    values = volume.values[tuple(lookup)]
    grid = np.arange(first.size, dtype=int).reshape(first.shape)
    faces = np.column_stack(
        (
            grid[:-1, :-1].ravel(),
            grid[1:, :-1].ravel(),
            grid[1:, 1:].ravel(),
            grid[:-1, 1:].ravel(),
        )
    )
    return vertices, faces, values


def _grayscale(values: np.ndarray, display_range: tuple[float, float]) -> np.ndarray:
    low, high = display_range
    scale = high - low if high > low else 1.0
    intensity = np.clip((values - low) / scale, 0.0, 1.0)
    intensity = np.nan_to_num(intensity, nan=0.0)
    return np.repeat(intensity[:, None], 3, axis=1)


def _anatomical_plane_axes(volume: MedicalVolume):
    directions = np.abs(volume.world_axes)
    voxel_axes = max(
        permutations(range(3)),
        key=lambda assignment: sum(
            directions[world_axis, voxel_axis]
            for world_axis, voxel_axis in enumerate(assignment)
        ),
    )
    return (("Axial", voxel_axes[2]), ("Coronal", voxel_axes[1]), ("Sagittal", voxel_axes[0]))


class MultimodalViewer:
    def __init__(
        self,
        specs: Sequence[RegionSpec],
        *,
        layout: str = "side-by-side",
        gap: float | None = None,
    ) -> None:
        if layout not in LAYOUTS:
            raise ValueError(f"Unknown layout {layout!r}")
        self.specs = list(specs)
        self.layout = layout
        self.layout_index = LAYOUTS.index(layout)
        self.gap = gap
        self.regions: list[RegionView] = []
        self.message = ""

    @staticmethod
    def _translation(offset: np.ndarray) -> np.ndarray:
        transform = np.eye(4, dtype=float)
        transform[:3, 3] = offset
        return transform

    def _update_slice(self, region: RegionView, item: SliceView) -> None:
        assert region.volume is not None
        vertices, _faces, values = _slice_geometry(region.volume, item.axis, item.index)
        item.structure.update_vertex_positions(vertices)
        item.structure.add_color_quantity(
            f"{region.volume.modality} grayscale",
            _grayscale(values, region.volume.display_range),
            defined_on="vertices",
            enabled=True,
        )

    def _set_layout(self, layout: str) -> None:
        self.layout = layout
        offsets = (
            side_by_side_offsets([(region.spec.name, region.bounds) for region in self.regions], self.gap)
            if layout == "side-by-side"
            else {region.spec.name: np.zeros(3, dtype=float) for region in self.regions}
        )
        for region in self.regions:
            region.offset = offsets[region.spec.name]
            translation = self._translation(region.offset)
            for surface in region.surfaces:
                surface.structure.set_transform(translation)
            if region.volume is not None:
                volume_transform = translation @ region.volume.transform
                for item in region.slices:
                    item.structure.set_transform(volume_transform)

    def register_scene(self) -> None:
        """Register this viewer inside an already initialized Polyscope app."""

        import polyscope as ps

        self.clear_scene()
        for spec in self.specs:
            bounds: list[np.ndarray] = []
            volume = load_medical_volume(spec) if spec.volume_path else None
            if volume is not None:
                bounds.append(_volume_bounds(volume))
            surface_payloads: list[tuple[Path, np.ndarray, np.ndarray]] = []
            for path in spec.surface_paths:
                vertices, faces = load_surface(path)
                surface_payloads.append((path, vertices, faces))
                bounds.append(np.vstack((vertices.min(axis=0), vertices.max(axis=0))))

            region = RegionView(spec=spec, bounds=combine_bounds(bounds), volume=volume)
            if volume is not None:
                for plane_index, (name, axis) in enumerate(_anatomical_plane_axes(volume)):
                    index = volume.values.shape[axis] // 2
                    vertices, faces, values = _slice_geometry(volume, axis, index)
                    structure = ps.register_surface_mesh(
                        f"{spec.name} / {volume.modality} / {name}",
                        vertices,
                        faces,
                        enabled=plane_index == 0,
                        edge_width=0.0,
                        smooth_shade=False,
                        back_face_policy="identical",
                    )
                    structure.add_color_quantity(
                        f"{volume.modality} grayscale",
                        _grayscale(values, volume.display_range),
                        defined_on="vertices",
                        enabled=True,
                    )
                    region.slices.append(
                        SliceView(name, axis, index, structure, visible=plane_index == 0)
                    )

            used_structures: set[str] = set()
            for path, vertices, faces in surface_payloads:
                label = _unique_name(path.stem, used_structures)
                structure = ps.register_surface_mesh(
                    f"{spec.name} / surface / {label}",
                    vertices,
                    faces,
                    smooth_shade=True,
                    transparency=0.55,
                )
                region.surfaces.append(SurfaceView(path, structure))
            self.regions.append(region)

        self._set_layout(self.layout)
        self.message = f"Loaded {len(self.regions)} independent region(s)."

    def clear_scene(self) -> None:
        """Remove only structures owned by this viewer."""

        for region in self.regions:
            for item in region.slices:
                try:
                    item.structure.remove()
                except RuntimeError:
                    pass
            for surface in region.surfaces:
                try:
                    surface.structure.remove()
                except RuntimeError:
                    pass
        self.regions.clear()

    def _draw_region(self, psim, region: RegionView) -> None:
        if not psim.TreeNode(region.spec.name):
            return
        try:
            if region.volume is not None:
                volume = region.volume
                changed, enabled = psim.Checkbox(
                    f"Show {volume.modality}##{region.spec.name}-volume",
                    region.volume_enabled,
                )
                if changed:
                    region.volume_enabled = enabled
                    for item in region.slices:
                        item.structure.set_enabled(enabled and item.visible)
                psim.TextUnformatted(
                    f"{volume.values.shape[0]} x {volume.values.shape[1]} x {volume.values.shape[2]} voxels"
                )
                for item in region.slices:
                    changed, visible = psim.Checkbox(
                        f"{item.name}##{region.spec.name}-{item.name}", item.visible
                    )
                    if changed:
                        item.visible = visible
                        item.structure.set_enabled(region.volume_enabled and visible)
                    changed, index = psim.SliderInt(
                        f"{item.name} slice##{region.spec.name}-{item.name}-slice",
                        item.index,
                        0,
                        volume.values.shape[item.axis] - 1,
                    )
                    if changed:
                        item.index = index
                        self._update_slice(region, item)

            for index, surface in enumerate(region.surfaces):
                suffix = f"{region.spec.name}-{index}"
                changed, visible = psim.Checkbox(
                    f"{surface.path.name}##surface-{suffix}", surface.visible
                )
                if changed:
                    surface.visible = visible
                    surface.structure.set_enabled(visible)
                changed, opacity = psim.SliderFloat(
                    f"Opacity##surface-{suffix}", surface.opacity, 0.0, 1.0
                )
                if changed:
                    surface.opacity = float(opacity)
                    surface.structure.set_transparency(surface.opacity)
        finally:
            psim.TreePop()

    def draw_controls(self, psim, *, show_title: bool = True) -> None:
        """Draw controls for standalone or embedded use."""

        if show_title:
            psim.TextUnformatted("Multimodal lung viewer")
        psim.TextWrapped(self.message)
        changed, index = psim.Combo("Layout", self.layout_index, list(LAYOUTS))
        if changed:
            self.layout_index = index
            self._set_layout(LAYOUTS[index])
        psim.TextWrapped(
            "Side-by-side separates named regions along X. Overlay keeps native patient/world coordinates."
        )
        for region in self.regions:
            self._draw_region(psim, region)

    def callback(self) -> None:
        import polyscope.imgui as psim

        self.draw_controls(psim)

    def run(self, *, smoke: bool = False) -> None:
        import polyscope as ps

        ps.set_program_name("Polyscope multimodal lung demo")
        ps.init()
        ps.set_ground_plane_mode("none")
        ps.set_navigation_style("free")
        self.register_scene()
        ps.set_user_callback(self.callback)
        ps.reset_camera_to_home_view()
        ps.show(1 if smoke else None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare CT, MR, and STL/PLY lung regions in Polyscope."
    )
    parser.add_argument(
        "--manifest",
        help="JSON manifest describing named regions. Paths are relative to the manifest.",
    )
    parser.add_argument(
        "--ct",
        action="append",
        default=[],
        metavar="[NAME=]PATH",
        help="CT file or DICOM directory; repeat for more than one CT region.",
    )
    parser.add_argument(
        "--mr",
        action="append",
        default=[],
        metavar="[NAME=]PATH",
        help="MR file or DICOM directory; repeat for more than one MR region.",
    )
    parser.add_argument(
        "--surface",
        "--stl",
        "--ply",
        action="append",
        default=[],
        metavar="[REGION=]PATH",
        help="STL/PLY surface. A matching REGION shares the CT/MR layout slot.",
    )
    parser.add_argument("--layout", choices=LAYOUTS, default="side-by-side")
    parser.add_argument(
        "--gap",
        type=float,
        help="Side-by-side gap in the input coordinate units (default: 15%% of max width).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate dependencies and bundled public data without opening a window.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _public_surface_specs() -> list[RegionSpec]:
    paths = [
        PUBLIC_DATA / name
        for name in ("LUL.ply", "LLL.ply", "RUL.ply", "RML.ply", "RLL.ply")
    ]
    available = [path for path in paths if path.exists()]
    if not available:
        return []
    return [RegionSpec("Public lung surfaces", surface_paths=available)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.check:
        summary = validate_public_bundle()
        print("Public multimodal demo data is valid:")
        for name, value in summary.items():
            print(f"  {name}: {value}")
        return 0
    if args.manifest:
        specs, manifest_layout, manifest_gap = load_manifest(args.manifest)
        layout = manifest_layout
        gap = manifest_gap
    else:
        specs = specs_from_arguments(args)
        layout = args.layout
        gap = args.gap
    if not specs:
        specs = _public_surface_specs()
    if not specs:
        parser.error("Provide --ct, --mr, --surface/--stl, or --manifest")
    MultimodalViewer(specs, layout=layout, gap=gap).run(smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
