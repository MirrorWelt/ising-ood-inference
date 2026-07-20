"""File-name discovery/parsing utilities.

Consolidates the regex-based ``.pt``/``.pth`` discovery logic that was
reimplemented, with small inconsistencies, in nearly every evaluation script.
"""

from __future__ import annotations

import glob
import os
import re
from collections import defaultdict

SUPPORTED_MODEL_TYPES = ("cnn", "improved_cnn", "transformer", "gnn", "hybrid")

_TEMPERATURE_PATTERNS = [
    (re.compile(r"^tensor_coupling_dataset_beta_([-+eE0-9.]+)_class(\d+)_samples_(\d+)\.pt$"), "beta"),
    (re.compile(r"^tensor_coupling_dataset_T_([-+eE0-9.]+)_class(\d+)_samples_(\d+)\.pt$"), "temperature"),
    (re.compile(r"^tensor_coupling_dataset_temp_([-+eE0-9.]+)_class(\d+)_samples_(\d+)\.pt$"), "temperature"),
    (
        re.compile(r"^tensor_coupling_dataset_temperature_([-+eE0-9.]+)_class(\d+)_samples_(\d+)\.pt$"),
        "temperature",
    ),
]


def safe_float(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def discover_temperature_files(data_dir: str) -> dict[int, list[dict]]:
    """Find all beta/T-sweep dataset files and group them by class (topology id).

    Returns:
        {class_id: [{"file_path", "x_type", "x_value_raw", "T"}, ...]},
        sorted by ascending T within each class.
    """
    index: dict[int, list[dict]] = defaultdict(list)

    for fp in glob.glob(os.path.join(data_dir, "*.pt")):
        name = os.path.basename(fp)
        for rgx, x_type in _TEMPERATURE_PATTERNS:
            m = rgx.match(name)
            if not m:
                continue
            val_str, cls_str, _ = m.groups()
            x_raw = safe_float(val_str)
            if x_raw is None:
                break

            if x_type == "beta":
                if x_raw == 0:
                    break
                t_val = 1.0 / x_raw
            else:
                t_val = x_raw

            index[int(cls_str)].append(
                {"file_path": fp, "x_type": x_type, "x_value_raw": x_raw, "T": t_val}
            )
            break

    for cls in index:
        index[cls] = sorted(index[cls], key=lambda z: z["T"])
    return index


def parse_model_type_from_name(filename: str, supported_types=SUPPORTED_MODEL_TYPES) -> str | None:
    low = filename.lower()
    for mt in supported_types:
        if f"regression_{mt}_" in low or low.startswith(f"regression_{mt}_"):
            return mt
    return None


def parse_class_count_from_name(filename: str) -> int | None:
    m = re.search(r"_(\d+)classes", filename.lower())
    return int(m.group(1)) if m else None


def discover_models(
    model_dir: str,
    only_model_types: list[str] | None = None,
    only_class_count: int | None = None,
) -> dict[str, str]:
    """Discover the best available checkpoint for each model type.

    Prefers files with ``_best.pth`` in the name; otherwise, the most recently
    modified matching file. ``_simple.pth`` (weights-only) files are skipped
    in favour of full checkpoints.
    """
    supported_types = list(SUPPORTED_MODEL_TYPES)
    if only_model_types is not None:
        supported_types = [m for m in supported_types if m in only_model_types]

    grouped: dict[str, list[str]] = defaultdict(list)
    for fp in glob.glob(os.path.join(model_dir, "*.pth")):
        name = os.path.basename(fp).lower()
        if "_simple.pth" in name:
            continue
        mt = parse_model_type_from_name(name, supported_types)
        if mt is None:
            continue
        cc = parse_class_count_from_name(name)
        if only_class_count is not None and cc != only_class_count:
            continue
        grouped[mt].append(fp)

    selected: dict[str, str] = {}
    for mt, files in grouped.items():
        def score(p: str):
            nm = os.path.basename(p).lower()
            return (1 if "_best.pth" in nm else 0, os.path.getmtime(p))

        selected[mt] = sorted(files, key=score, reverse=True)[0]
    return selected


def get_available_class_ids(data_dir: str, dataset_mode: str = "nodes") -> list[int]:
    """Return the sorted, de-duplicated list of topology ids present on disk."""
    if dataset_mode == "nodes":
        pattern = os.path.join(data_dir, "tensor_coupling_dataset_nodes_*_samples_*.pt")
        key = "nodes"
    elif dataset_mode == "33links":
        pattern = os.path.join(data_dir, "tensor_coupling_dataset_33links_matrix_*_samples_*.pt")
        key = "matrix"
    else:
        raise ValueError(f"Unsupported dataset_mode: {dataset_mode}")

    class_ids: list[int] = []
    for fp in glob.glob(pattern):
        parts = os.path.basename(fp).split("_")
        for i, part in enumerate(parts):
            if part == key and i + 1 < len(parts):
                try:
                    class_ids.append(int(parts[i + 1]))
                except ValueError:
                    pass
                break
    return sorted(set(class_ids))


def get_common_class_ids(nodes_dir: str, links33_dir: str) -> tuple[list[int], list[int], list[int]]:
    nodes_ids = set(get_available_class_ids(nodes_dir, "nodes"))
    links33_ids = set(get_available_class_ids(links33_dir, "33links"))
    common = sorted(nodes_ids & links33_ids)
    return common, sorted(nodes_ids), sorted(links33_ids)