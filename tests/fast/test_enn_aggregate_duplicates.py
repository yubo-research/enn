from __future__ import annotations

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags
from evals.harness.work_dirs import fresh_work_dir


def _repeated_data(copies: int = 20, sites: int = 12):
    rng = np.random.default_rng(0)
    site_x = rng.standard_normal((sites, 2))
    train_x = np.repeat(site_x, copies, axis=0)
    train_y = rng.standard_normal((sites * copies, 1))
    return train_x, train_y


def test_flag_defaults_off_and_is_remembered_on_reopen():
    train_x, train_y = _repeated_data(copies=2)
    work_dir = fresh_work_dir()
    off = EpistemicNearestNeighbors(train_x, train_y, work_dir=work_dir)
    assert off.aggregate_duplicates is False
    on_dir = fresh_work_dir()
    on = EpistemicNearestNeighbors(
        train_x, train_y, work_dir=on_dir, aggregate_duplicates=True
    )
    assert on.aggregate_duplicates is True
    del on
    assert EpistemicNearestNeighbors.open(on_dir).aggregate_duplicates is True


def test_len_counts_observations_and_neighbors_are_distinct_x():
    train_x, train_y = _repeated_data()
    model = EpistemicNearestNeighbors(
        train_x, train_y, work_dir=fresh_work_dir(), aggregate_duplicates=True
    )
    assert len(model) == len(train_x)
    neighbors = model.neighbors(train_x[:1], k=8, exclude_nearest=False)
    assert neighbors.shape == (1, 8)
    assert len({tuple(train_x[i]) for i in neighbors[0]}) == 8
    assert neighbors[0, 0] == 0


def test_exclude_nearest_removes_every_observation_at_the_nearest_x():
    train_x, train_y = _repeated_data()
    model = EpistemicNearestNeighbors(
        train_x, train_y, work_dir=fresh_work_dir(), aggregate_duplicates=True
    )
    query = train_x[40:41]
    neighbors = model.neighbors(query, k=5, exclude_nearest=True)
    assert all(not np.array_equal(train_x[i], query[0]) for i in neighbors[0])


def test_repeats_keep_reducing_the_epistemic_error():
    params = ENNParams(
        k_num_neighbors=3, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.5
    )
    flags = PosteriorFlags()
    se_by_setting = {}
    for aggregate in (False, True):
        ses = []
        for copies in (4, 40):
            train_x = np.concatenate(
                [np.zeros((copies, 1)), np.array([[1.0], [2.0], [3.0], [4.0]])]
            )
            train_y = np.concatenate(
                [np.ones((copies, 1)), np.array([[0.0], [3.0], [-1.0], [2.0]])]
            )
            model = EpistemicNearestNeighbors(
                train_x,
                train_y,
                work_dir=fresh_work_dir(),
                aggregate_duplicates=aggregate,
            )
            post = model.posterior(np.zeros((1, 1)), params=params, flags=flags)
            ses.append(float(post.se_epi[0, 0]) / float(model._y_scale.ravel()[0]))
        se_by_setting[aggregate] = ses
    assert se_by_setting[True][1] < se_by_setting[True][0]
    assert abs(se_by_setting[False][1] - se_by_setting[False][0]) < 1e-9
