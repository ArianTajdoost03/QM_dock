import pytest

from qmdock.cli import build_parser, config_from_args
from qmdock.config import Config
from qmdock.qm.jobs import energy_job, relax_job, strain_job

import numpy as np


def parse(*extra):
    return build_parser("x").parse_args(["--receptor", "r", "--ligand", "l", "--out", "o", *extra])


def test_accuracy_options_are_accepted_and_reach_the_config():
    cfg = config_from_args(parse("--accuracy", "3", "--relax-accuracy", "1.5"), "l")
    assert cfg.accuracy == 3.0 and cfg.relax_accuracy == 1.5


def test_defaults_keep_the_current_numerics():
    cfg = config_from_args(parse(), "l")
    assert cfg.accuracy == 1.0 and cfg.relax_accuracy == 1.0


def test_non_positive_accuracy_is_rejected():
    with pytest.raises(SystemExit):
        config_from_args(parse("--accuracy", "0"), "l")
    with pytest.raises(SystemExit):
        config_from_args(parse("--relax-accuracy", "-1"), "l")


def test_energy_jobs_use_accuracy_while_relaxation_and_strain_use_relax_accuracy():
    cfg = Config("r", "l", accuracy=3.0, relax_accuracy=1.0)
    numbers, pos = np.array([8, 1, 1], dtype=np.int32), np.zeros((3, 3))
    assert energy_job("k", numbers, pos, 0, cfg)["accuracy"] == 3.0
    assert relax_job("k", numbers, pos, np.arange(3), 0, cfg)["accuracy"] == 1.0
    assert strain_job("k", numbers, pos, np.arange(3), 0, cfg)["accuracy"] == 1.0
