UNPAIRED = {"Mn": 5, "Fe": 4, "Co": 3, "Ni": 2, "Cu": 1}
UNCERTAIN = ("Mn", "Fe", "Co", "Ni")


def parse_uhf(text):
    if str(text).strip().lower() == "auto":
        return "auto"
    value = int(text)
    if value < 0:
        raise ValueError("uhf must be 'auto' or a non-negative integer")
    return value


def uhf_for(elements, setting):
    if setting == "auto":
        return sum(UNPAIRED.get(str(e), 0) for e in elements)
    return int(setting)


def cluster_uhf(cluster, setting, receptor=None):
    if setting == "auto" and getattr(receptor, "metals", None) is not None:
        from qmdock.chem.metals import cluster_plan

        n = cluster_plan(receptor, cluster).uhf
    else:
        n = uhf_for(cluster.elements(), setting)
    return n if (cluster.electrons() - n) % 2 == 0 else None


def uncertain_metals(cluster):
    return sorted({str(e) for e in cluster.elements() if str(e) in UNCERTAIN})
