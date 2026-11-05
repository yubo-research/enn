import json

import numpy as np
import pytest

from enn import EpistemicNearestNeighbors

IDENTITY = "store_identity.json"


def _rows(n, seed):
    x = np.random.RandomState(seed).random((n, 3))
    return x, x.sum(axis=1, keepdims=True)


def _store(work_dir):
    x, y = _rows(10, 1)
    return EpistemicNearestNeighbors(x, y, work_dir=work_dir)


def _assert_aligned(model):
    x, y, _ = model.train_rows_at(list(range(len(model))))
    np.testing.assert_allclose(x.sum(axis=1, keepdims=True), y)


def _mark_unfinished(work_dir):
    path = work_dir / IDENTITY
    identity = json.loads(path.read_text())
    identity["committed"] = False
    path.write_text(json.dumps(identity))


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
