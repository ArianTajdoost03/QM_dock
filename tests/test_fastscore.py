import numpy as np
import pytest

torch = pytest.importorskip("torch")
from config import Config
from fastscore import Scorer, diverse, rodrigues, search


def make_scorer(cfg):
    grid = np.array([[x, y, 0.0] for x in np.arange(-6, 6.1, 1.5) for y in np.arange(-6, 6.1, 1.5)])
    return Scorer(np.array(["C"] * len(grid)), grid, ["C", "C", "O"], torch.device("cpu"), cfg)


def test_rodrigues_is_rotation():
    R = rodrigues(torch.randn(5, 3))
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3).expand(5, 3, 3), atol=1e-5)


def test_clash_detected():
    cfg = Config("r", "l")
    sc = make_scorer(cfg)
    X = torch.tensor([[[0.0, 0.0, 0.5], [1.2, 0.0, 0.5], [2.4, 0.0, 0.5]],
                      [[0.0, 0.0, 4.0], [1.2, 0.0, 4.0], [2.4, 0.0, 4.0]]])
    _, clash = sc.evaluate(X)
    assert clash.tolist() == [True, False]


def test_search_returns_clash_free_poses():
    cfg = Config("r", "l", n_random=2000, n_optimize=100, opt_steps=30, n_qm=5, search_radius=4.0)
    sc = make_scorer(cfg)
    conf = np.array([[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.4, 0.9, 0.0]]])
    ps = search(sc, conf, np.array([True, True, True]), np.array([0.0, 0.0, 4.0]), cfg, 1, 0)
    assert len(ps) > 0
    _, clash = sc.evaluate(torch.tensor(ps.coords, dtype=torch.float32))
    assert not clash.any()


def test_diverse_filters_close_poses():
    a = np.zeros((3, 4, 3))
    a[1] += 0.1
    a[2] += 5.0
    assert diverse(a, [0, 1, 2], 1.0, 10) == [0, 2]


def test_mutation_never_worsens_best_score():
    base = dict(n_random=2000, n_optimize=100, opt_steps=30, n_qm=5, search_radius=4.0)
    conf = np.array([[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.4, 0.9, 0.0]]])
    heavy = np.array([True, True, True])
    plain = Config("r", "l", **base)
    mutated = Config("r", "l", mutation_rounds=2, mutation_parents=10, mutation_children=4, mutation_steps=15, **base)
    a = search(make_scorer(plain), conf, heavy, np.array([0.0, 0.0, 4.0]), plain, 1, 0)
    b = search(make_scorer(mutated), conf, heavy, np.array([0.0, 0.0, 4.0]), mutated, 1, 0)
    assert b.diag["mutation"]["children"] == 80
    assert b.score.min() <= a.score.min() + 1e-6
    _, clash = make_scorer(mutated).evaluate(torch.tensor(b.coords, dtype=torch.float32))
    assert not clash.any()
