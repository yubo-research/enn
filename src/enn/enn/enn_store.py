from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from enn._rust import InputHistory

INPUT_TRANSFORM_FILE = "input_transform.npz"
OUTPUT_PROJECTION_FILE = "output_projection.npz"


def _save_atomically(artifact: Any, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    artifact.save(temporary)
    temporary.replace(path)


def save_store_artifacts(work_dir: Any, input_transform, output_projection) -> None:
    root = Path(work_dir)
    if input_transform is not None:
        _save_atomically(input_transform, root.joinpath(INPUT_TRANSFORM_FILE))
    if output_projection is not None:
        _save_atomically(output_projection, root.joinpath(OUTPUT_PROJECTION_FILE))


def _require_recorded(actual: str, recorded: str, name: str) -> None:
    if actual != recorded:
        raise ValueError(f"{name} does not match the fingerprint in the store")


def open_input_transform(rust_model: Any, work_dir: Any) -> tuple[Any, Any]:
    recorded, semantic_only, _, _ = rust_model.store_settings
    if recorded is None:
        return None, None
    from enn.semantic import SemanticInputTransform

    transform = SemanticInputTransform.load(
        Path(work_dir).joinpath(INPUT_TRANSFORM_FILE)
    )
    actual = transform.fingerprint(semantic_only=semantic_only)
    _require_recorded(actual, recorded, INPUT_TRANSFORM_FILE)
    no_rows = np.zeros((0, transform.bounds.shape[0]), dtype=float)
    return transform, InputHistory(no_rows, rust_model)


def open_output_projection(rust_model: Any, work_dir: Any) -> Any:
    _, _, recorded, alpha = rust_model.store_settings
    if recorded is None:
        return None
    from enn.semantic_outputs import SemanticOutputProjection

    projection = SemanticOutputProjection.load(
        Path(work_dir).joinpath(OUTPUT_PROJECTION_FILE)
    )
    _require_recorded(projection.fingerprint(), recorded, OUTPUT_PROJECTION_FILE)
    rust_model.set_semantic_outputs(projection.loadings, alpha)
    return projection
