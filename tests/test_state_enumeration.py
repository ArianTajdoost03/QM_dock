import time

import pytest
from rdkit import Chem

from qmdock.chem.pka import parse_overrides
from qmdock.chem.protonation import ligand_states
from qmdock.cli import build_parser, config_from_args
from qmdock.config import Config
from qmdock.pipeline.run import enumerate_states


def states(smi, n=4, window=5.0, ph=7.4, overrides=None, mode="all", log=None):
    return ligand_states(Chem.MolFromSmiles(smi), mode, 6.4, 8.4, n, 7, ph=ph, window=window, overrides=overrides,
                         log=log)


def charges(smi, **kw):
    return [s["charge"] for s in states(smi, **kw)]


def test_the_dominant_phosphonate_dianion_survives_a_small_max_states():
    assert charges("CP(=O)(O)O", n=2) == [-2, -1]
    assert charges("CP(=O)(O)O", n=1) == [-2]


def test_dominant_state_comes_first_with_zero_penalty_and_penalties_never_decrease():
    out = states("NCC(=O)O")
    assert out[0]["charge"] == 0 and out[0]["penalty"] == 0.0
    penalties = [s["penalty"] for s in out]
    assert penalties == sorted(penalties)


def test_implausible_states_outside_the_window_are_dropped():
    out = states("NCC(=O)O")
    assert sorted(s["charge"] for s in out) == [-1, 0, 1]
    assert all(s["penalty"] <= 5.0 for s in out)


def test_window_zero_keeps_every_state_up_to_max_states():
    assert len(states("NCC(=O)O", window=0.0)) == 4
    assert len(states("NCC(=O)O", window=0.0, n=2)) == 2


def test_a_tight_window_leaves_only_the_dominant_state():
    assert charges("NCC(=O)O", window=1.0) == [0]


def test_ph_moves_the_dominant_state():
    assert charges("CC(=O)O", n=1, ph=7.4) == [-1]
    assert charges("CC(=O)O", n=1, ph=2.0) == [0]


def test_pka_overrides_change_which_state_dominates():
    assert charges("CC(=O)O", n=1, overrides=parse_overrides(["carboxylate=9.0"])) == [0]


def test_the_given_state_is_reused_with_its_own_coordinates_when_it_survives():
    base = Chem.AddHs(Chem.MolFromSmiles("[NH3+]CC(=O)[O-]"))
    from rdkit.Chem import AllChem

    AllChem.EmbedMolecule(base, randomSeed=5)
    out = ligand_states(base, "all", 6.4, 8.4, 4, 7, ph=7.4)
    given = [s for s in out if s["label"] == "as given"]
    assert len(given) == 1 and given[0]["mol"] is base


def test_every_returned_state_has_three_dimensional_coordinates():
    for s in states("CP(=O)(O)O"):
        assert s["mol"].GetNumConformers() == 1


def test_more_than_the_site_limit_of_titratable_groups_is_handled_quickly():
    smi = "OC(=O)CC(CC(=O)O)CC(CC(=O)O)CC(CC(=O)O)CC(CC(=O)O)CC(CC(=O)O)CC(=O)O"
    t0 = time.time()
    out = states(smi, n=3)
    assert time.time() - t0 < 30
    assert out[0]["charge"] == -7 and out[0]["penalty"] == 0.0


def test_none_mode_returns_only_the_given_state():
    out = states("NCC(=O)O", mode="none")
    assert len(out) == 1 and "penalty" not in out[0]


def test_enumeration_is_reported_through_the_log():
    lines = []
    states("NCC(=O)O", log=lines.append)
    assert any("kept 3 of 4" in line for line in lines)
    assert any("dropped" in line for line in lines)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        states("CCO", mode="bogus")


def test_dimorphite_candidates_are_ranked_by_penalty_not_by_absolute_charge():
    pytest.importorskip("dimorphite_dl")
    out = states("CP(=O)(O)O", mode="pka", n=2)
    assert out[0]["penalty"] == 0.0
    assert out[0]["charge"] <= out[1]["charge"] or out[0]["penalty"] <= out[1]["penalty"]


def test_config_and_cli_carry_the_window_and_ph_into_the_enumeration():
    a = build_parser("x").parse_args(["--receptor", "r", "--ligand", "l", "--out", "o", "--state-window", "3.5",
                                      "--state-ph", "6.8", "--max-states", "2"])
    cfg = config_from_args(a, "l")
    assert cfg.state_window == 3.5 and cfg.state_ph == 6.8
    cfg = Config("r", "l", protonation="all", max_states=1, state_window=5.0)
    assert [s["charge"] for s in enumerate_states(cfg, Chem.MolFromSmiles("CP(=O)(O)O"))] == [-2]
