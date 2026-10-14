"""Shared, lightweight data loading helpers for the public demos."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PUBLIC_DATA = ROOT / "hologram_data"
PRIVATE_DATA = ROOT / "ps_demo_data"


def load_triangle_surface(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one STL/PLY as vertices and triangular faces without VTK."""

    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - installation problem
        raise RuntimeError(
            "Surface loading requires trimesh; run `python -m pip install -r requirements.txt`."
        ) from exc

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Surface file not found: {source}")
    loaded = trimesh.load_mesh(source, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"Surface contains no geometry: {source}")
        loaded = trimesh.util.concatenate(geometries)
    vertices = np.asarray(loaded.vertices, dtype=float)
    faces = np.asarray(loaded.faces, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f"Surface contains no 3D vertices: {source}")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError(f"Surface contains no triangular faces: {source}")
    return vertices, faces


def require_private_data(*names: str) -> list[Path]:
    """Resolve private demo files and raise one actionable error if absent."""

    paths = [PRIVATE_DATA / name for name in names]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            "This legacy demo needs private ps_demo_data files which are not in the "
            f"public repository: {joined}. Use hologram.py or multimodal_viewer.py "
            "for a demo that runs with the included public data."
        )
    return paths


def validate_public_bundle() -> dict[str, int]:
    """Load every public demo asset and return a compact shape summary."""

    required = {
        "image": PUBLIC_DATA / "raw_3D_RAS.npz",
        "tree": PUBLIC_DATA / "tree.npz",
        "growth": PUBLIC_DATA / "growing.pkl",
        "ventilation": PUBLIC_DATA / "ventilation.npz",
    }
    missing = [path.name for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Public demo data is incomplete: " + ", ".join(sorted(missing))
        )

    with np.load(required["image"]) as image:
        image_values = np.asarray(image["array"])
        spacing = np.asarray(image["spacing"])
    if image_values.ndim != 3 or spacing.size != 3:
        raise ValueError("raw_3D_RAS.npz has invalid image data")

    with np.load(required["tree"]) as tree:
        nodes = np.asarray(tree["nodes"])
        edges = np.asarray(tree["edges"])
        radius = np.asarray(tree["radius"])
    if nodes.ndim != 2 or nodes.shape[1] != 3 or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("tree.npz has invalid geometry")
    if len(radius) not in {len(nodes), len(edges)}:
        raise ValueError("tree.npz radius length does not match nodes or edges")

    with required["growth"].open("rb") as stream:
        frames = pickle.load(stream)
    if not frames or not {"nodes", "edges", "radius"}.issubset(frames[0]):
        raise ValueError("growing.pkl does not contain valid animation frames")

    with np.load(required["ventilation"]) as ventilation:
        acinus = np.asarray(ventilation["acinus"])
        dvdt = np.asarray(ventilation["dvdt"])
    if acinus.ndim != 2 or acinus.shape[1] != 3 or dvdt.ndim != 2:
        raise ValueError("ventilation.npz has invalid point or frame data")

    surface_faces = 0
    for name in ("LUL.ply", "LLL.ply", "RUL.ply", "RML.ply", "RLL.ply"):
        _vertices, faces = load_triangle_surface(PUBLIC_DATA / name)
        surface_faces += len(faces)

    return {
        "image_voxels": int(image_values.size),
        "tree_nodes": int(len(nodes)),
        "tree_edges": int(len(edges)),
        "growth_frames": int(len(frames)),
        "ventilation_points": int(len(acinus)),
        "ventilation_frames": int(len(dvdt)),
        "surface_faces": int(surface_faces),
    }
