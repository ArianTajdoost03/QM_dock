import numpy as np

from qmdock.config import Config
from qmdock.qm.engine import run_job
from qmdock.qm.jobs import energy_job


def hexaaqua(metal_z, distance):
    numbers, coords = [metal_z], [np.zeros(3)]
    for axis in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]):
        d = np.array(axis, dtype=float)
        o = d * distance
        perp = np.cross(d, [0.3, 0.5, 0.8])
        perp /= np.linalg.norm(perp)
        numbers += [8, 1, 1]
        coords += [o, o + 0.96 * (np.cos(0.9) * d + np.sin(0.9) * perp), o + 0.96 * (np.cos(0.9) * d - np.sin(0.9) * perp)]
    return np.array(numbers, dtype=np.int32), np.array(coords)


def energy(cfg, uhf, charge=2, metal=26, distance=2.12):
    numbers, pos = hexaaqua(metal, distance)
    out = run_job(energy_job(("e", uhf), numbers, pos, charge, cfg, uhf))
    assert out["ok"], out.get("error")
    return out["energy"]


def test_spin_polarization_makes_the_high_spin_state_of_aqua_iron_ii_the_lowest():
    on = Config("r", "l", solvent="water")
    off = Config("r", "l", solvent="water", spin_polarization=0.0)
    assert energy(on, 4) < energy(on, 0)
    assert energy(off, 0) < energy(off, 4)


def test_the_term_changes_open_shell_energies_but_not_closed_shell_ones():
    on = Config("r", "l", solvent="water")
    off = Config("r", "l", solvent="water", spin_polarization=0.0)
    assert energy(on, 4) != energy(off, 4)
    assert energy(on, 0) == energy(off, 0)


def test_manganese_ii_and_cobalt_ii_also_prefer_their_high_spin_states_with_the_term():
    cfg = Config("r", "l", solvent="water")
    assert energy(cfg, 5, metal=25, distance=2.18) < energy(cfg, 1, metal=25, distance=2.18)
    assert energy(cfg, 3, metal=27, distance=2.10) < energy(cfg, 1, metal=27, distance=2.10)
