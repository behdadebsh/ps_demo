"""Build and load the anonymized pulmonary perfusion demo dataset."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PerfusionData:
    nodes: np.ndarray
    edges: np.ndarray
    edge_radius: np.ndarray
    terminal_node_indices: np.ndarray
    terminal_element_ids: np.ndarray
    terminal_steady_flow: np.ndarray
    terminal_pressure: np.ndarray
    terminal_flow: np.ndarray
    cycle_phase: np.ndarray


def prepare_perfusion_animation(
    values: np.ndarray, *, value_gain: float = 4.0
) -> np.ndarray:
    """Emphasize pulsatile terminal flow while preserving no-flow terminals."""

    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 2 or not source.size:
        raise ValueError("Perfusion values must be a non-empty frame-by-terminal array")
    if not np.all(np.isfinite(source)):
        raise ValueError("Perfusion values contain non-finite values")
    if value_gain < 0.0:
        raise ValueError("Perfusion value gain cannot be negative")

    reference = np.mean(source, axis=0)
    emphasized = reference + value_gain * (source - reference)
    return np.clip(emphasized, 0.0, None).astype(np.float32, copy=False)


def _parse_exnode(path: str | Path, value_count: int) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    current: int | None = None
    values: list[float] = []

    def store() -> None:
        if current is None:
            return
        if len(values) != value_count:
            raise ValueError(
                f"Node {current} in {path} has {len(values)} values; expected {value_count}"
            )
        result[current] = np.asarray(values, dtype=float)

    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = re.match(r"\s*Node:\s*(\d+)", line)
            if match:
                store()
                current = int(match.group(1))
                values = []
            elif current is not None and len(values) < value_count:
                parts = line.split()
                if len(parts) == 1:
                    try:
                        values.append(float(parts[0]))
                    except ValueError:
                        pass
    store()
    if not result:
        raise ValueError(f"No nodes found in {path}")
    return result


def _parse_exelem_connectivity(path: str | Path) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    current: int | None = None
    read_nodes = False
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = re.match(r"\s*Element:\s*(\d+)", line)
            if match:
                current = int(match.group(1))
                read_nodes = False
            elif current is not None and line.strip() == "Nodes:":
                read_nodes = True
            elif read_nodes:
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    result[current] = (int(parts[0]), int(parts[1]))
                    read_nodes = False
    if not result:
        raise ValueError(f"No element connectivity found in {path}")
    return result


def _parse_exelem_scalar(path: str | Path, maximum_id: int) -> dict[int, float]:
    result: dict[int, float] = {}
    current: int | None = None
    read_values = False
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = re.match(r"\s*Element:\s*(\d+)", line)
            if match:
                current = int(match.group(1))
                read_values = False
                if current > maximum_id:
                    break
            elif current is not None and line.strip() == "Values:":
                read_values = True
            elif read_values:
                values = [float(value) for value in line.split()]
                if values:
                    result[current] = float(np.mean(values))
                    read_values = False
    return result


def build_perfusion_data(
    node_path: str | Path,
    element_path: str | Path,
    radius_path: str | Path,
    terminal_path: str | Path,
    flow_path: str | Path,
) -> PerfusionData:
    """Parse EX files and map each terminal node to its incident flow element."""

    node_values = _parse_exnode(node_path, 3)
    element_nodes = _parse_exelem_connectivity(element_path)
    terminal_values = _parse_exnode(terminal_path, 5)

    node_ids = sorted(node_values)
    element_ids = sorted(element_nodes)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    nodes = np.asarray([node_values[node_id] for node_id in node_ids], dtype=np.float32)
    try:
        edges = np.asarray(
            [
                [node_index[first], node_index[second]]
                for first, second in (element_nodes[element_id] for element_id in element_ids)
            ],
            dtype=np.int32,
        )
    except KeyError as exc:
        raise ValueError(f"Element references missing node {exc.args[0]}") from exc

    radius_by_element = _parse_exelem_scalar(radius_path, max(element_ids))
    missing_radii = sorted(set(element_ids) - set(radius_by_element))
    if missing_radii:
        raise ValueError(f"Missing radius for {len(missing_radii)} vessel elements")
    edge_radius = np.asarray(
        [radius_by_element[element_id] for element_id in element_ids], dtype=np.float32
    )

    incident: dict[int, list[int]] = {}
    for element_id, connected in element_nodes.items():
        for node_id in connected:
            incident.setdefault(node_id, []).append(element_id)

    terminal_node_ids = sorted(terminal_values)
    terminal_element_ids: list[int] = []
    for node_id in terminal_node_ids:
        connected = incident.get(node_id, [])
        if len(connected) != 1:
            raise ValueError(
                f"Terminal node {node_id} touches {len(connected)} elements; expected one"
            )
        terminal_element_ids.append(connected[0])
        if node_id not in node_index:
            raise ValueError(f"Terminal node {node_id} is absent from the full vessel tree")
        if not np.allclose(
            node_values[node_id], terminal_values[node_id][:3], rtol=0.0, atol=1e-5
        ):
            raise ValueError(f"Coordinates disagree for terminal node {node_id}")

    flow_table = np.loadtxt(flow_path)
    if flow_table.ndim != 2 or flow_table.shape[1] < 2:
        raise ValueError("Flow table must contain an element ID and cardiac samples")
    if not np.all(np.isfinite(flow_table)):
        raise ValueError("Flow table contains non-finite values")
    flow_ids = flow_table[:, 0].astype(np.int64)
    if not np.array_equal(flow_table[:, 0], flow_ids):
        raise ValueError("Flow table element IDs must be integers")
    if len(np.unique(flow_ids)) != len(flow_ids):
        raise ValueError("Flow table contains duplicate element IDs")
    flow_row = {int(element_id): index for index, element_id in enumerate(flow_ids)}
    missing_flows = sorted(set(terminal_element_ids) - set(flow_row))
    if missing_flows:
        raise ValueError(f"Missing time-dependent flow for {len(missing_flows)} terminals")
    terminal_flow = np.asarray(
        [flow_table[flow_row[element_id], 1:] for element_id in terminal_element_ids],
        dtype=np.float32,
    ).T

    terminal_steady_flow = np.asarray(
        [terminal_values[node_id][3] for node_id in terminal_node_ids],
        dtype=np.float32,
    )
    terminal_pressure = np.asarray(
        [terminal_values[node_id][4] for node_id in terminal_node_ids],
        dtype=np.float32,
    )
    if not np.allclose(
        np.mean(terminal_flow, axis=0),
        terminal_steady_flow,
        rtol=5e-4,
        atol=5e-6,
    ):
        raise ValueError("Terminal waveform means do not match steady-state terminal flow")

    return PerfusionData(
        nodes=nodes,
        edges=edges,
        edge_radius=edge_radius,
        terminal_node_indices=np.asarray(
            [node_index[node_id] for node_id in terminal_node_ids], dtype=np.int32
        ),
        terminal_element_ids=np.asarray(terminal_element_ids, dtype=np.int32),
        terminal_steady_flow=terminal_steady_flow,
        terminal_pressure=terminal_pressure,
        terminal_flow=terminal_flow,
        cycle_phase=np.linspace(
            0.0, 1.0, terminal_flow.shape[0], endpoint=False, dtype=np.float32
        ),
    )


def save_perfusion_data(data: PerfusionData, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        nodes=data.nodes,
        edges=data.edges,
        edge_radius=data.edge_radius,
        terminal_node_indices=data.terminal_node_indices,
        terminal_element_ids=data.terminal_element_ids,
        terminal_steady_flow=data.terminal_steady_flow,
        terminal_pressure=data.terminal_pressure,
        terminal_flow=data.terminal_flow,
        cycle_phase=data.cycle_phase,
    )


def load_perfusion_data(path: str | Path) -> PerfusionData:
    source = Path(path)
    with np.load(source) as payload:
        data = PerfusionData(
            nodes=np.asarray(payload["nodes"], dtype=np.float32),
            edges=np.asarray(payload["edges"], dtype=np.int32),
            edge_radius=np.asarray(payload["edge_radius"], dtype=np.float32),
            terminal_node_indices=np.asarray(
                payload["terminal_node_indices"], dtype=np.int32
            ),
            terminal_element_ids=np.asarray(
                payload["terminal_element_ids"], dtype=np.int32
            ),
            terminal_steady_flow=np.asarray(
                payload["terminal_steady_flow"], dtype=np.float32
            ),
            terminal_pressure=np.asarray(
                payload["terminal_pressure"], dtype=np.float32
            ),
            terminal_flow=np.asarray(payload["terminal_flow"], dtype=np.float32),
            cycle_phase=np.asarray(payload["cycle_phase"], dtype=np.float32),
        )
    _validate_perfusion_data(data, source)
    return data


def _validate_perfusion_data(data: PerfusionData, source: Path) -> None:
    if data.nodes.ndim != 2 or data.nodes.shape[1] != 3:
        raise ValueError(f"Invalid vessel nodes in {source}")
    if data.edges.ndim != 2 or data.edges.shape[1] != 2:
        raise ValueError(f"Invalid vessel edges in {source}")
    if len(data.edge_radius) != len(data.edges):
        raise ValueError(f"Vessel radius count does not match edges in {source}")
    numeric_arrays = (
        data.nodes,
        data.edge_radius,
        data.terminal_steady_flow,
        data.terminal_pressure,
        data.terminal_flow,
        data.cycle_phase,
    )
    if any(not np.all(np.isfinite(values)) for values in numeric_arrays):
        raise ValueError(f"Perfusion data contains non-finite values in {source}")
    if np.any(data.edge_radius <= 0):
        raise ValueError(f"Vessel radii must be positive in {source}")
    if np.any(data.edges < 0) or np.any(data.edges >= len(data.nodes)):
        raise ValueError(f"Vessel edges reference invalid nodes in {source}")
    terminal_count = len(data.terminal_node_indices)
    if np.any(data.terminal_node_indices < 0) or np.any(
        data.terminal_node_indices >= len(data.nodes)
    ):
        raise ValueError(f"Terminal indices are invalid in {source}")
    if data.terminal_flow.ndim != 2 or data.terminal_flow.shape[1] != terminal_count:
        raise ValueError(f"Terminal flow dimensions are invalid in {source}")
    if any(
        len(values) != terminal_count
        for values in (
            data.terminal_element_ids,
            data.terminal_steady_flow,
            data.terminal_pressure,
        )
    ):
        raise ValueError(f"Terminal metadata dimensions are invalid in {source}")
    if len(data.cycle_phase) != data.terminal_flow.shape[0]:
        raise ValueError(f"Cardiac phase count does not match flow frames in {source}")
    if len(data.cycle_phase) < 2:
        raise ValueError(f"Perfusion data needs at least two cardiac frames in {source}")
    if not np.allclose(
        np.mean(data.terminal_flow, axis=0),
        data.terminal_steady_flow,
        rtol=5e-4,
        atol=5e-6,
    ):
        raise ValueError(
            f"Terminal waveform means do not match steady-state flow in {source}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert pulmonary perfusion EX/text inputs to an anonymized NPZ demo."
    )
    parser.add_argument("--nodes", required=True, help="Full vessel-tree EXNODE")
    parser.add_argument("--elements", required=True, help="Full vessel-tree EXELEM")
    parser.add_argument("--radii", required=True, help="Element-radius EXELEM")
    parser.add_argument("--terminals", required=True, help="Terminal EXNODE")
    parser.add_argument("--flows", required=True, help="Element-by-time flow text file")
    parser.add_argument("--output", required=True, help="Output NPZ path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = build_perfusion_data(
        args.nodes, args.elements, args.radii, args.terminals, args.flows
    )
    save_perfusion_data(data, args.output)
    print(
        f"Saved {len(data.nodes)} nodes, {len(data.edges)} vessels, "
        f"{len(data.terminal_node_indices)} terminals, and "
        f"{len(data.cycle_phase)} cardiac frames to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
