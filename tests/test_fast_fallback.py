from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qmdock.config import Config
from qmdock.pipeline.fast import guarded_search
from qmdock.search.sampling import exact_clash

CPU = SimpleNamespace(type="cpu")


def test_out_of_memory_retries_with_a_smaller_budget_then_without_the_grid():
    seen = []

    def run(scorer, cfg):
        seen.append((cfg.pair_budget, cfg.grid_spacing))
        if len(seen) < 3:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return "poses"

    cfg = Config("r", "l", pair_budget=8e7, grid_spacing=0.25)
    assert guarded_search(cfg, lambda c: c, run, CPU) == "poses"
    assert seen == [(8e7, 0.25), (4e7, 0.25), (2e7, 0.0)]


def test_other_errors_and_a_third_out_of_memory_are_not_swallowed():
    def other(scorer, cfg):
        raise RuntimeError("CUDA error: unknown error")

    with pytest.raises(RuntimeError, match="unknown error"):
        guarded_search(Config("r", "l"), lambda c: c, other, CPU)
    calls = []

    def always(scorer, cfg):
        calls.append(1)
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        guarded_search(Config("r", "l"), lambda c: c, always, CPU)
    assert len(calls) == 3


def test_a_successful_first_attempt_is_untouched():
    cfg = Config("r", "l", pair_budget=6e7)
    seen = []
    assert guarded_search(cfg, lambda c: c, lambda s, c: seen.append(c.pair_budget) or "ok", CPU) == "ok"
    assert seen == [6e7]


def test_the_exact_clash_probe_is_evaluated_in_memory_bounded_chunks():
    sizes = []

    class FakeScorer:
        def chunk(self, grad, full):
            return 7

        def evaluate(self, X, full=False):
            sizes.append(len(X))
            return torch.zeros(len(X)), torch.zeros(len(X), dtype=torch.bool)

    out = exact_clash(FakeScorer(), torch.zeros(20, 3, 3))
    assert len(out) == 20 and max(sizes) == 7 and sizes == [7, 7, 6]
