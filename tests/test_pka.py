import math
import sys
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

from qmdock.chem.pka import KT_LN10, ionized_sites, parse_overrides, penalties_from_smiles, state_penalties
from qmdock.chem.protonation import ligand_states
from qmdock.config import Config
from qmdock.pipeline.stateselect import choose_state, corrected_energy, finalize, prepare
from qmdock.screening.report import choose_best_state

sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "bench_test_1"))
import collect_rmsd


def free(smi, ph=7.4):
    return penalties_from_smiles([smi], ph)[0]["free_energy"]


def test_kt_ln10_is_the_textbook_value():
    assert KT_LN10 == pytest.approx(1.364, abs=0.001)


def test_carboxylic_acid_is_ionized_at_physiological_ph_and_neutral_in_strong_acid():
    low, high = penalties_from_smiles(["CC(=O)O", "CC(=O)[O-]"], 7.4)
    assert low["penalty"] == pytest.approx(KT_LN10 * (7.4 - 4.5))
    assert high["penalty"] == 0.0
    low, high = penalties_from_smiles(["CC(=O)O", "CC(=O)[O-]"], 2.0)
    assert low["penalty"] == 0.0
    assert high["penalty"] == pytest.approx(KT_LN10 * (4.5 - 2.0))


def test_a_site_at_its_pka_costs_nothing_to_ionize():
    assert free("CC(=O)[O-]", ph=4.5) == pytest.approx(0.0)


def test_amine_protonated_state_is_favoured_at_ph_7_4():
    neutral, charged = penalties_from_smiles(["CCN", "CC[NH3+]"], 7.4)
    assert charged["penalty"] == 0.0
    assert neutral["penalty"] == pytest.approx(KT_LN10 * (10.6 - 7.4))


def test_zwitterion_adds_both_site_contributions():
    zw = free("[NH3+]CC(=O)[O-]")
    assert zw == pytest.approx(KT_LN10 * (4.5 - 7.4) + KT_LN10 * (7.4 - 10.6))


def test_phosphate_second_ionization_uses_its_own_pka():
    sites, _ = ionized_sites(Chem.MolFromSmiles("COP(=O)([O-])[O-]"))
    assert sorted(s["pka"] for s in sites) == [2.0, 7.2]
    mono, di = penalties_from_smiles(["COP(=O)(O)[O-]", "COP(=O)([O-])[O-]"], 7.4)
    assert di["free_energy"] - mono["free_energy"] == pytest.approx(KT_LN10 * (7.2 - 7.4))


def test_permanent_charges_add_no_penalty_and_are_not_flagged():
    for smi in ("C[N+](C)(C)C", "c1ccccc1[N+](=O)[O-]", "C[n+]1ccccc1"):
        row = state_penalties([Chem.MolFromSmiles(smi)], 7.4)[0]
        assert row["free_energy"] == 0.0 and row["unclassified"] == []


def test_unknown_charged_group_is_reported():
    row = state_penalties([Chem.MolFromSmiles("C[Se+](C)C")], 7.4)[0]
    assert row["free_energy"] == 0.0 and len(row["unclassified"]) == 1


def test_overrides_replace_the_table_value_and_reject_unknown_names():
    base = free("CC(=O)[O-]")
    shifted = penalties_from_smiles(["CC(=O)[O-]"], 7.4, parse_overrides(["carboxylate=3.0"]))[0]["free_energy"]
    assert shifted == pytest.approx(KT_LN10 * (3.0 - 7.4))
    assert shifted != base
    with pytest.raises(ValueError):
        parse_overrides(["nonsense=3"])
    with pytest.raises(ValueError):
        parse_overrides(["carboxylate=abc"])


def test_penalties_are_non_negative_and_the_most_populated_state_has_zero():
    rows = penalties_from_smiles(["CCN", "CC[NH3+]", "CC[NH3+]"], 7.4)
    assert min(r["penalty"] for r in rows) == 0.0 and all(r["penalty"] >= 0 for r in rows)


def test_enumerated_states_of_a_real_ligand_get_a_penalty_for_the_neutral_amine():
    base = Chem.MolFromSmiles("Nc1ncnc2c([C@@H]3N[C@H](CO)[C@@H](O)[C@H]3O)c[nH]c12")
    states = ligand_states(base, "all", 6.4, 8.4, 4, 7)
    rows = state_penalties([s["mol"] for s in states], 7.4)
    by_charge = {s["charge"]: r for s, r in zip(states, rows)}
    assert by_charge[1]["penalty"] == 0.0
    assert by_charge[0]["penalty"] > 3.0


def row(state, charge, energy, penalty, level=2, **extra):
    return dict(state=state, charge=charge, best_e_final_kcal=energy, best_level=level, pka_penalty_kcal=penalty,
                smiles="x", dir=f"d{state}", **extra)


def test_penalty_can_overturn_the_raw_energy_ranking():
    rows = [row(0, 0, -25.5, 4.6), row(1, 1, -24.8, 0.0)]
    assert choose_state(rows)["state"] == 1
    assert choose_best_state(rows)["state"] == 1
    assert corrected_energy(rows[0]) == pytest.approx(-20.9)


def test_selection_still_prefers_the_higher_level_and_skips_failed_states():
    rows = [row(0, 0, -30.0, 0.0, level=1), row(1, 1, -10.0, 0.0, level=2), {"state": 2, "error": "boom"}]
    assert choose_state(rows)["state"] == 1
    assert choose_state([{"state": 0, "error": "x"}]) is None


def test_rows_without_penalty_fall_back_to_the_raw_energy():
    rows = [dict(state=0, charge=0, best_e_final_kcal=-5.0, best_level=2, smiles="a", dir="a"),
            dict(state=1, charge=1, best_e_final_kcal=-7.0, best_level=2, smiles="b", dir="b")]
    assert choose_state(rows)["state"] == 1


def test_finalize_marks_the_selected_state_and_adds_the_corrected_energy():
    rows = [row(0, 0, -25.5, 4.6), row(1, 1, -24.8, 0.0)]
    out = finalize(rows, Config("r", "l"))
    assert [r["selected"] for r in out] == [False, True]
    assert out[0]["best_e_corrected_kcal"] == pytest.approx(-20.9)


def test_prepare_is_off_for_one_state_or_when_disabled():
    states = [{"mol": Chem.MolFromSmiles("CCN")}, {"mol": Chem.MolFromSmiles("CC[NH3+]")}]
    assert prepare(Config("r", "l"), states[:1]) == [None]
    assert prepare(Config("r", "l", pka_penalty=False), states) == [None, None]
    pens = prepare(Config("r", "l"), states)
    assert pens[0]["penalty"] > 0 and pens[1]["penalty"] == 0.0


def test_benchmark_reads_penalties_from_states_csv_or_recomputes_them(tmp_path):
    table = pd.DataFrame({"state": [0, 1], "smiles": ["CCN", "CC[NH3+]"]})
    table.to_csv(tmp_path / "states.csv", index=False)
    computed = collect_rmsd.run_penalties(str(tmp_path), 7.4)
    assert computed["state_0"] > 0 and computed["state_1"] == 0.0
    table["pka_penalty_kcal"] = [1.5, 0.25]
    table.to_csv(tmp_path / "states.csv", index=False)
    assert collect_rmsd.run_penalties(str(tmp_path), 7.4) == {"state_0": 1.5, "state_1": 0.25}
    assert collect_rmsd.run_penalties(str(tmp_path / "missing"), 7.4) == {}
