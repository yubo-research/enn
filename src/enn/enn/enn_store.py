from __future__ import annotations

from typing import Any

from enn._rust import semantic_artifact_paths


def _require_recorded(actual: str, recorded: str, name: str) -> None:
    if actual != recorded:
        raise ValueError(f"{name} does not match the fingerprint in the store")


def open_input_transform(rust_model: Any) -> Any:
    recorded, semantic_only, _, _ = rust_model.store_settings
    if recorded is None:
        return None
    from enn.semantic import SemanticInputTransform

    transform = SemanticInputTransform.load(semantic_artifact_paths(rust_model)[0])
    actual = transform.fingerprint(semantic_only=semantic_only)
    _require_recorded(actual, recorded, "input_transform.npz")
    return transform


def open_output_projection(rust_model: Any) -> Any:
    _, _, recorded, alpha = rust_model.store_settings
    if recorded is None:
        return None
    from enn.semantic_outputs import SemanticOutputProjection

    projection = SemanticOutputProjection.load(semantic_artifact_paths(rust_model)[1])
    _require_recorded(projection.fingerprint(), recorded, "output_projection.npz")
    rust_model.set_semantic_outputs(projection.loadings, alpha)
    return projection
