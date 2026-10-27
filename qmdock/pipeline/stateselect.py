from qmdock.chem.pka import parse_overrides, state_penalties
from qmdock.qm.pool import log


def prepare(cfg, states):
    if not cfg.pka_penalty or len(states) < 2:
        return [None] * len(states)
    return state_penalties([s["mol"] for s in states], cfg.state_ph, parse_overrides(cfg.pka_overrides))


def apply_penalty(row, penalty):
    if penalty is not None:
        row.update(pka_penalty_kcal=round(penalty["penalty"], 3), ionized_sites=penalty["sites_text"],
                   pka_unclassified=len(penalty["unclassified"]))
    return row


def corrected_energy(row):
    energy = row.get("best_e_final_kcal")
    if energy is None:
        return None
    return energy + (row.get("pka_penalty_kcal") or 0.0)


def choose_state(rows):
    ok = [r for r in rows if "error" not in r and r.get("best_e_final_kcal") is not None]
    if not ok:
        return None
    top = max(r["best_level"] for r in ok)
    return min((r for r in ok if r["best_level"] == top), key=corrected_energy)


def finalize(rows, cfg):
    for r in rows:
        value = corrected_energy(r)
        if value is not None:
            r["best_e_corrected_kcal"] = round(value, 3)
    best = choose_state(rows)
    for r in rows:
        r["selected"] = r is best
    if not any("pka_penalty_kcal" in r for r in rows):
        return rows
    log(f"protonation states ranked by e_final plus a pKa penalty at pH {cfg.state_ph:g} "
        "(approximate site pKa table, independent sites):")
    for r in rows:
        if r.get("best_e_final_kcal") is None:
            continue
        sites = r.get("ionized_sites") or "neutral"
        mark = "  <- selected" if r["selected"] else ""
        log(f"  state {r['state']}  charge {r['charge']:+d}  e_final {r['best_e_final_kcal']:8.2f}  "
            f"penalty {r.get('pka_penalty_kcal', 0.0):5.2f}  corrected {r['best_e_corrected_kcal']:8.2f}  {sites}{mark}")
        if r.get("pka_unclassified"):
            log(f"    {r['pka_unclassified']} charged atom(s) in state {r['state']} are not in the pKa table and add no penalty")
    scored = [r for r in rows if r.get("best_e_final_kcal") is not None and "error" not in r]
    if best is not None and scored:
        raw = min((r for r in scored if r["best_level"] == best["best_level"]), key=lambda r: r["best_e_final_kcal"])
        if raw is not best:
            log(f"  the penalty changed the selected state from {raw['state']} (lowest raw energy) to {best['state']}")
    return rows
