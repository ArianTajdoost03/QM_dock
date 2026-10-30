from dataclasses import dataclass, field

import numpy as np

from qmdock.chem.elements import METAL_CHARGE

GROUP = {"Mn": 7, "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11}
VARIABLE = tuple(GROUP)
CODE_STATES = {
    "CU1": ("Cu", 1), "CU": ("Cu", 2), "CU3": ("Cu", 3),
    "FE2": ("Fe", 2), "FE": ("Fe", 3),
    "MN": ("Mn", 2), "MN3": ("Mn", 3),
    "CO": ("Co", 2), "3CO": ("Co", 3),
    "NI": ("Ni", 2), "3NI": ("Ni", 3),
}
DONORS = ("N", "O", "S", "Cl", "Br", "I", "Se")
DONOR_CUTOFF = 2.8
STRICT_DONOR_CUTOFF = 2.6
COUPLING_DISTANCE = 4.0
PLANAR_TOLERANCE = 0.4
SCAN_MARGIN_KCAL = 10.0
ALWAYS_LOW_SPIN = (("Co", 3),)


@dataclass
class MetalSite:
    atom: int
    element: str
    oxidation: int
    source: str
    confidence: str
    donors: dict
    coordination: int
    planar: bool
    candidates: tuple
    group: int = -1
    label: str = ""
    partners: tuple = field(default_factory=tuple)

    @property
    def d_electrons(self):
        return GROUP[self.element] - self.oxidation

    @property
    def paramagnetic(self):
        return self.candidates != (0,)


@dataclass
class SpinPlan:
    uhf: int
    notes: list
    confidence: str
    scan_site: object = None
    scan_options: tuple = ()


def parse_state_overrides(items):
    out = []
    for item in items or ():
        left, _, value = str(item).partition("=")
        element, _, selector = left.strip().partition("@")
        element = element.strip().capitalize()
        if element not in VARIABLE:
            raise ValueError(f"--metal-state supports {', '.join(VARIABLE)}; got '{element}'")
        try:
            state = int(value)
        except ValueError:
            raise ValueError(f"metal state '{item}' must look like Cu=1 or Cu@A401=2") from None
        if not 0 <= state <= 6:
            raise ValueError(f"oxidation state in '{item}' must be between 0 and 6")
        out.append((element, selector.strip().upper() or None, state))
    return out


def match_override(overrides, element, chain, resseq):
    general = None
    for el, selector, state in overrides:
        if el != element:
            continue
        if selector is None:
            general = state if general is None else general
        elif selector in (f"{chain}{resseq}".upper(), str(resseq).upper()):
            return state
    return general


def spin_candidates(element, oxidation, coordination, n_nitrogen, planar):
    d = GROUP[element] - oxidation
    if d <= 0 or d >= 10:
        return (0,)
    if d <= 3:
        return (d,)
    if d == 9:
        return (1,)
    strong = coordination >= 6 and n_nitrogen >= 5
    if d == 8:
        return (0, 2) if coordination == 4 and planar else (2, 0)
    if d == 7:
        return (1, 3) if element == "Ni" else (3, 1)
    if d == 6:
        return (0, 4, 2) if strong or (element, oxidation) in ALWAYS_LOW_SPIN else (4, 0, 2)
    if d == 5:
        return (1, 5, 3) if strong else (5, 1, 3)
    return (4, 2, 0)


def planar_four(metal_xyz, donor_xyz):
    pts = np.vstack([donor_xyz, metal_xyz])
    centered = pts - pts.mean(axis=0)
    normal = np.linalg.svd(centered)[2][-1]
    return float(np.abs(centered @ normal).max()) <= PLANAR_TOLERANCE


def infer_oxidation(element, resname, formal, override, strict_donors):
    if override is not None:
        return override, "user option", "high"
    if formal and formal > 0:
        return int(formal), "PDB charge column", "high"
    code = CODE_STATES.get(resname.upper())
    if code and code[0] == element:
        state, source, confidence = code[1], f"PDB ligand code {resname}", "medium"
    else:
        state, source, confidence = METAL_CHARGE[element], "assumed, no state information in the file", "low"
    if element == "Cu" and state == 2 and confidence != "high":
        n = len(strict_donors)
        if 2 <= n <= 3 and "O" not in strict_donors:
            return 1, f"coordination ({n} N/S donors, no O)", "low"
    return state, source, confidence


def assign_metal_sites(receptor, overrides=(), prior=None):
    formal = getattr(receptor, "formal", None)
    sites = {}
    for atom in np.where(np.isin(receptor.elements, VARIABLE))[0]:
        atom = int(atom)
        element = str(receptor.elements[atom])
        chain, resseq, _, resname = receptor.residues[receptor.residue_of_atom[atom]]
        near = [int(i) for i in receptor.tree.query_ball_point(receptor.coords[atom], DONOR_CUTOFF)
                if i != atom and receptor.elements[i] in DONORS]
        dist = {i: float(np.linalg.norm(receptor.coords[i] - receptor.coords[atom])) for i in near}
        strict = [str(receptor.elements[i]) for i in near if dist[i] <= STRICT_DONOR_CUTOFF]
        override = match_override(overrides, element, chain, resseq)
        known = None
        if prior is not None and override is None:
            known = nearest_prior(prior, receptor.coords[atom], element)
        if known is not None:
            state, source, confidence = known.oxidation, known.source, known.confidence
        else:
            state, source, confidence = infer_oxidation(
                element, resname, int(formal[atom]) if formal is not None else 0, override, strict)
        counts = {}
        for i in near:
            counts[str(receptor.elements[i])] = counts.get(str(receptor.elements[i]), 0) + 1
        coordination = len(near)
        planar = bool(coordination == 4 and planar_four(receptor.coords[atom], receptor.coords[near]))
        cands = spin_candidates(element, state, coordination, counts.get("N", 0), planar)
        sites[atom] = MetalSite(atom, element, state, source, confidence, counts, coordination, planar, cands,
                                label=f"{element}({'I' * state if state < 4 else state}) {chain}{resseq}",
                                partners=tuple(near))
    link_coupled(receptor, sites)
    return sites


def nearest_prior(prior, xyz, element):
    best, best_d = None, 1.5
    for site in prior.values():
        if site.element != element:
            continue
        d = float(np.linalg.norm(prior_xyz(site, prior) - xyz))
        if d < best_d:
            best, best_d = site, d
    return best


def prior_xyz(site, prior):
    return getattr(prior, "xyz", {}).get(site.atom, np.full(3, 1e9))


def link_coupled(receptor, sites):
    magnetic = [s for s in sites.values() if s.paramagnetic]
    parent = {s.atom: s.atom for s in magnetic}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(magnetic):
        for b in magnetic[i + 1:]:
            close = np.linalg.norm(receptor.coords[a.atom] - receptor.coords[b.atom]) < COUPLING_DISTANCE
            if close or set(a.partners) & set(b.partners):
                parent[find(a.atom)] = find(b.atom)
    for s in magnetic:
        root = find(s.atom)
        members = [m for m in magnetic if find(m.atom) == root]
        s.group = root if len(members) > 1 else -1


class SiteTable(dict):
    def __init__(self, sites=(), xyz=None):
        super().__init__(sites)
        self.xyz = xyz or {}


def build_site_table(receptor, overrides=(), prior=None):
    sites = assign_metal_sites(receptor, overrides, prior)
    table = SiteTable(sites, {a: receptor.coords[a].copy() for a in sites})
    return table


def spin_name(unpaired):
    return {0: "closed-shell", 1: "doublet", 2: "triplet", 3: "quartet", 4: "quintet", 5: "sextet"}.get(
        unpaired, f"{unpaired} unpaired")


def cluster_plan(receptor, cluster):
    wanted = {int(a) for a in cluster.atoms}
    inside = [s for a, s in receptor.metals.items() if a in wanted]
    total, notes, confidence = 0, [], "high"
    singles, groups = [], {}
    for s in inside:
        if not s.paramagnetic:
            if s.element in VARIABLE:
                notes.append(f"{s.label} d{s.d_electrons} closed-shell ({s.source})")
                confidence = lower(confidence, s.confidence)
            continue
        if s.group >= 0:
            groups.setdefault(s.group, []).append(s)
        else:
            singles.append(s)
    for g, members in list(groups.items()):
        if len(members) < 2:
            singles.append(members[0])
            del groups[g]
    for s in singles:
        total += s.candidates[0]
        alt = f", alternative {spin_name(s.candidates[1])}" if len(s.candidates) > 1 else ""
        notes.append(f"{s.label} d{s.d_electrons} {spin_name(s.candidates[0])} assumed ({s.source}{alt})")
        confidence = lower(confidence, s.confidence)
        if len(s.candidates) > 1:
            confidence = lower(confidence, "medium")
    for members in groups.values():
        low = sum(s.candidates[0] for s in members) % 2
        total += low
        names = "-".join(s.label.split()[0] for s in members)
        notes.append(f"coupled {names}: lowest-spin (antiferromagnetic-like) state, closed-shell approximation")
        confidence = lower(confidence, "low")
        for s in members:
            confidence = lower(confidence, s.confidence)
    plan = SpinPlan(total, notes, confidence)
    scannable = [s for s in singles if len(s.candidates) > 1]
    if len(scannable) == 1:
        site = scannable[0]
        options = []
        for cand in site.candidates[:2]:
            options.append(total - site.candidates[0] + cand)
        plan.scan_site, plan.scan_options = site, tuple(dict.fromkeys(options))
    return plan


RANK = {"high": 2, "medium": 1, "low": 0}


def lower(a, b):
    return a if RANK[a] <= RANK[b] else b


def describe(receptor):
    lines = []
    for s in sorted(receptor.metals.values(), key=lambda s: s.atom):
        donors = ",".join(f"{n}{e}" for e, n in sorted(s.donors.items())) or "none"
        spin = spin_name(s.candidates[0])
        extra = ""
        if len(s.candidates) > 1:
            extra = f", alternative {spin_name(s.candidates[1])}"
        coupled = " coupled" if s.group >= 0 else ""
        lines.append(f"  metal {s.label}: oxidation state {s.oxidation} [{s.source}; {s.confidence} confidence], "
                     f"d{s.d_electrons}, {s.coordination} donors ({donors}), {spin}{extra}{coupled}")
    return lines
