import json

import numpy as np
import pytest

from enn import EpistemicNearestNeighbors
from enn.semantic import SemanticInputTransform

IDENTITY = "store_identity.json"
ORIGINAL_INPUTS = "original_inputs.bin"


class _Encoder:
    def encode(self, texts):
        return np.random.RandomState(0).normal(size=(len(texts), 8))


def _transform():
    names = ["a", "b", "c"]
    return SemanticInputTransform(names, np.array([[0.0, 1.0]] * 3), encoder=_Encoder())


def _rows(n, seed):
    x = np.random.RandomState(seed).random((n, 3))
    return x, x.sum(axis=1, keepdims=True)


def _store(work_dir):
    x, y = _rows(10, 1)
    return EpistemicNearestNeighbors(
        x, y, input_transform=_transform(), work_dir=work_dir
    )


def _assert_aligned(model):
    x, y, _ = model.train_rows_at(list(range(len(model))))
    np.testing.assert_allclose(x.sum(axis=1, keepdims=True), y)


def _leave_row_that_never_reached_the_model(work_dir):
    path = work_dir / ORIGINAL_INPUTS
    data = bytearray(path.read_bytes())
    count = int.from_bytes(data[16:24], "little")
    data[16:24] = (count + 1).to_bytes(8, "little")
    data.extend(np.array([9.0, 9.0, 9.0]).tobytes())
    path.write_bytes(data)


def _mark_unfinished(work_dir):
    path = work_dir / IDENTITY
    identity = json.loads(path.read_text())
    identity["committed"] = False
    path.write_text(json.dumps(identity))


def test_original_row_without_a_model_row_is_dropped_on_open(tmp_path):
    work_dir = tmp_path / "store"
    model = _store(work_dir)
    model.persist_index_to_disk()
    del model
    _leave_row_that_never_reached_the_model(work_dir)
    reopened = EpistemicNearestNeighbors.open(work_dir)
    assert len(reopened) == 10
    x, y = _rows(2, 2)
    reopened.add(x, y)
    _assert_aligned(reopened)
    np.testing.assert_allclose(reopened.train_rows_at([10, 11])[0], x)


def test_a_store_whose_creation_stopped_can_be_created_again(tmp_path):
    work_dir = tmp_path / "store"
    model = _store(work_dir)
    model.persist_index_to_disk()
    del model
    _mark_unfinished(work_dir)
    with pytest.raises(ValueError, match="never finished"):
        EpistemicNearestNeighbors.open(work_dir)
    recreated = _store(work_dir)
    _assert_aligned(recreated)
    recreated.persist_index_to_disk()
    del recreated
    _assert_aligned(EpistemicNearestNeighbors.open(work_dir))


def test_a_finished_store_is_not_replaced(tmp_path):
    work_dir = tmp_path / "store"
    model = _store(work_dir)
    with pytest.raises(ValueError, match="already holds a store"):
        _store(work_dir)
    assert len(model) == 10


def test_rows_cannot_reach_the_model_without_their_originals(tmp_path):
    model = _store(tmp_path / "store")
    x, y = _rows(1, 3)
    encoded = model.transform_inputs(x)
    with pytest.raises(ValueError, match="requires them"):
        model.rust_backend.add(encoded, y)
    assert len(model) == 10


def test_a_store_without_a_transform_refuses_original_rows(tmp_path):
    x, y = _rows(4, 4)
    model = EpistemicNearestNeighbors(x, y, work_dir=tmp_path / "plain")
    with pytest.raises(ValueError, match="keeps no original inputs"):
        model.rust_backend.add(x, y, None, x)
    assert not (tmp_path / "plain" / ORIGINAL_INPUTS).exists()
