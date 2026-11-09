#!/usr/bin/env python3
"""Thermal Woodward–Hoffmann labels from cyclic active-orbital TS matrices.

This is an idealized symmetry dataset, not a transition-state geometry or
activation-barrier dataset. A candidate approach is part of every example.
The Hückel-style matrix includes the active π/σ/H orbitals needed by the
reaction class, with a signed cyclic overlap boundary. Labels are formal
thermal symmetry permission (1 or 0), not whether a reaction is fast or
experimentally observed.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import numpy as np


@dataclass(frozen=True)
class Template:
    name: str
    reactants: str
    sites: tuple[str, ...]
    edge_roles: tuple[str, ...]
    modes: tuple[tuple[str, int], tuple[str, int]]
    electron_sources: str

    def __post_init__(self) -> None:
        if len(self.sites) != len(self.edge_roles) or len(self.sites) % 2:
            raise ValueError(f"Invalid cyclic template: {self.name}")
        if {phase for _, phase in self.modes} != {0, 1}:
            raise ValueError(f"Both orbital phase parities required: {self.name}")


# Site order follows a continuous cyclic array. For each edge role at index i,
# the interaction is from site i to (i+1) mod n. A role called "forming" or
# "breaking" identifies its role along the proposed reaction coordinate.
TEMPLATES = (
    Template("diels_alder_4plus2", "C=CC=C.C=C", ("C",) * 6,
             ("pi", "conjugated", "pi", "forming_CC", "pi", "forming_CC"),
             (("supra_supra", 0), ("supra_antara", 1)), "pi4 + pi2"),
    Template("alder_ene", "C=CC.C=C", ("C", "C", "C", "H", "C", "C"),
             ("pi", "conjugated", "breaking_CH", "forming_CH", "pi", "forming_CC"),
             (("all_suprafacial", 0), ("one_antarafacial_component", 1)),
             "pi2 + pi2 + sigma_CH2"),
    Template("claisen_3_3", "C=CCOC=C", ("C", "C", "C", "O", "C", "C"),
             ("pi", "conjugated", "breaking_CO", "conjugated", "pi", "forming_CC"),
             (("supra_supra", 0), ("supra_antara", 1)), "pi2 + pi2 + sigma_CO2"),
    Template("cope_3_3", "C=CCCC=C", ("C",) * 6,
             ("pi", "conjugated", "breaking_CC", "conjugated", "pi", "forming_CC"),
             (("supra_supra", 0), ("supra_antara", 1)), "pi2 + pi2 + sigma_CC2"),
    Template("cycloaddition_2plus2", "C=C.C=C", ("C",) * 4,
             ("pi", "forming_CC", "pi", "forming_CC"),
             (("supra_supra", 0), ("supra_antara", 1)), "pi2 + pi2"),
    Template("cycloaddition_4plus4", "C=CC=C.C=CC=C", ("C",) * 8,
             ("pi", "conjugated", "pi", "forming_CC", "pi", "conjugated", "pi", "forming_CC"),
             (("supra_supra", 0), ("supra_antara", 1)), "pi4 + pi4"),
    Template("electrocyclic_4pi", "C=CC=C", ("C",) * 4,
             ("pi", "conjugated", "pi", "forming_CC"),
             (("disrotatory", 0), ("conrotatory", 1)), "pi4"),
    Template("electrocyclic_6pi", "C=CC=CC=C", ("C",) * 6,
             ("pi", "conjugated", "pi", "conjugated", "pi", "forming_CC"),
             (("disrotatory", 0), ("conrotatory", 1)), "pi6"),
)

SITE_ALPHA = {"C": 0.0, "O": -0.8, "H": -0.4}
ROLE_RANGE = {
    "pi": (0.90, 1.10),
    "conjugated": (0.60, 0.85),
    "breaking_CH": (0.25, 0.60),
    "breaking_CO": (0.25, 0.60),
    "breaking_CC": (0.25, 0.60),
    "forming_CH": (0.18, 0.52),
    "forming_CC": (0.18, 0.52),
}
SITE_COUPLING = {"C": 1.0, "O": 0.8, "H": 0.75}
MAX_SITES = max(len(t.sites) for t in TEMPLATES)

# Rooted, acyclic substituents placed on carbon atoms with an implicit H.
# The parent reaction center remains intact. Alkyl variants are included to
# provide graph diversity without requiring large or strained pi systems.
SUBSTITUENTS = (
    "none", "methyl", "ethyl", "propyl", "butyl", "pentyl", "hexyl",
    "fluoro", "chloro", "hydroxy", "methoxy", "cyano",
)
SITE_SHIFT = {
    "none": 0.0, "methyl": 0.025, "ethyl": 0.035, "propyl": 0.040,
    "butyl": 0.045, "pentyl": 0.050, "hexyl": 0.055,
    "fluoro": -0.080, "chloro": -0.060, "hydroxy": -0.075,
    "methoxy": -0.055, "cyano": -0.120,
}
VALENCE = {"C": 4, "O": 2, "N": 3, "H": 1, "F": 1, "Cl": 1}


def parent_graph(template: Template) -> nx.Graph:
    """Reactant connectivity from the template's existing active bonds."""
    graph = nx.Graph()
    for i, element in enumerate(template.sites):
        graph.add_node(i, element=element)
    n = len(template.sites)
    for i, role in enumerate(template.edge_roles):
        if role.startswith("forming"):
            continue
        graph.add_edge(i, (i + 1) % n, order=2 if role == "pi" else 1)
    assert nx.is_forest(graph)
    return graph


def _add_substituent(graph: nx.Graph, site: int, name: str) -> None:
    if name == "none":
        return
    if name in ("methyl", "ethyl", "propyl", "butyl", "pentyl", "hexyl"):
        length = ("methyl", "ethyl", "propyl", "butyl", "pentyl", "hexyl").index(name) + 1
        elements = ["C"] * length
        orders = [1] * (length - 1)
    elif name == "fluoro":
        elements, orders = ["F"], []
    elif name == "chloro":
        elements, orders = ["Cl"], []
    elif name == "hydroxy":
        elements, orders = ["O"], []
    elif name == "methoxy":
        elements, orders = ["O", "C"], [1]
    elif name == "cyano":
        elements, orders = ["C", "N"], [3]
    else:
        raise ValueError(name)
    previous = site
    for index, element in enumerate(elements):
        node = len(graph)
        graph.add_node(node, element=element)
        graph.add_edge(previous, node, order=1 if index == 0 else orders[index - 1])
        previous = node


def reactant_graph(template: Template, substitutions: tuple[str, ...]) -> nx.Graph:
    if len(substitutions) != len(template.sites):
        raise ValueError("One substituent choice is required per active site")
    graph = parent_graph(template)
    for site, name in enumerate(substitutions):
        if name != "none":
            if template.sites[site] != "C":
                raise ValueError("Only carbon sites can be decorated")
            _add_substituent(graph, site, name)
    for node in graph:
        element = graph.nodes[node]["element"]
        used = sum(graph.edges[node, neighbor]["order"] for neighbor in graph[node])
        implicit_h = VALENCE[element] - used
        if implicit_h < 0:
            raise ValueError(f"Valence exceeded at {element} site {node}")
        graph.nodes[node]["implicit_h"] = implicit_h
    assert nx.is_forest(graph)
    return graph


def reactant_smiles(graph: nx.Graph) -> str:
    """Write an acyclic, explicit-branch SMILES for the graph."""
    def visit(node: int, previous: int | None = None) -> str:
        element = graph.nodes[node]["element"]
        token = "[H]" if element == "H" else element
        children = sorted(neighbor for neighbor in graph[node] if neighbor != previous)
        fragments = []
        for child in children:
            order = graph.edges[node, child]["order"]
            fragments.append({1: "", 2: "=", 3: "#"}[order] + visit(child, node))
        if fragments:
            token += "".join("(" + fragment + ")" for fragment in fragments[:-1]) + fragments[-1]
        return token

    return ".".join(visit(min(component)) for component in nx.connected_components(graph))


def graph_record(graph: nx.Graph, sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "nodes": [{"id": int(i), "element": graph.nodes[i]["element"],
                   "implicit_h": int(graph.nodes[i]["implicit_h"])} for i in graph],
        "bonds": [[int(i), int(j), int(data["order"])]
                  for i, j, data in graph.edges(data=True)],
    }


def graph_hash(graph: nx.Graph) -> str:
    return nx.weisfeiler_lehman_graph_hash(graph, node_attr="element", edge_attr="order")


def sample_unique_reactant(
    template: Template, rng: np.random.Generator, seen: dict[str, list[nx.Graph]],
) -> tuple[tuple[str, ...], nx.Graph, str, str]:
    """Reject graph-isomorphic reactants, including across candidate modes."""
    node_match = nx.algorithms.isomorphism.categorical_node_match("element", None)
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("order", None)
    for _ in range(30_000):
        substitutions = tuple(
            str(rng.choice(SUBSTITUENTS)) if element == "C" else "none"
            for element in template.sites
        )
        graph = reactant_graph(template, substitutions)
        digest = graph_hash(graph)
        peers = seen.get(digest, [])
        if any(nx.is_isomorphic(graph, prior, node_match=node_match,
                               edge_match=edge_match) for prior in peers):
            continue
        graph_id = f"{digest}:{len(peers)}"
        seen.setdefault(digest, []).append(graph)
        return substitutions, graph, digest, graph_id
    raise RuntimeError(f"Could not find a new reactant graph for {template.name}")


def thermal_allowed(n_electrons: int, cycle_phase: int) -> int:
    """Formal ground-state Hückel/Möbius thermal selection rule.

    cycle_phase is +1 for an even number of orbital phase inversions and -1
    for an odd number. +1 allows 4k+2 electrons; -1 allows 4k electrons.
    """
    if n_electrons < 2 or n_electrons % 2 or cycle_phase not in (-1, 1):
        raise ValueError("Expected positive even electron count and phase ±1")
    return int((n_electrons % 4 == 2) == (cycle_phase == 1))


def cycle_phase_from_matrix(h: np.ndarray) -> int:
    """Gauge-invariant sign product of -H around the active orbital cycle."""
    n = h.shape[0]
    if h.shape != (n, n) or n < 4 or np.any(np.abs(h[np.arange(n), (np.arange(n) + 1) % n]) < 1e-8):
        raise ValueError("Expected a nonzero signed cyclic matrix")
    return int(np.prod(np.sign(-h[np.arange(n), (np.arange(n) + 1) % n])))


def build_candidate(
    template: Template, phase_inversions: int, rng: np.random.Generator,
    *, substitutions: tuple[str, ...] | None = None,
    randomize_gauge: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one Hückel-style cyclic TS matrix and its orbital annotations.

    All edge magnitudes are illustrative dimensionless overlap strengths.
    Orbital phase gauge and cyclic starting site are randomized; the loop
    sign and hence the symmetry label are invariant.
    """
    n = len(template.sites)
    h = np.diag([SITE_ALPHA[site] for site in template.sites]).astype(np.float64)
    if substitutions is not None:
        if len(substitutions) != n:
            raise ValueError("Substitution vector length does not match active sites")
        for i, name in enumerate(substitutions):
            h[i, i] += SITE_SHIFT[name]
    for i, role in enumerate(template.edge_roles):
        j = (i + 1) % n
        low, high = ROLE_RANGE[role]
        magnitude = float(rng.uniform(low, high))
        magnitude *= np.sqrt(SITE_COUPLING[template.sites[i]] * SITE_COUPLING[template.sites[j]])
        sign = -1.0 if phase_inversions and i == n - 1 else 1.0
        h[i, j] = h[j, i] = -magnitude * sign
    gauge = rng.choice(np.array([-1.0, 1.0]), size=n) if randomize_gauge else np.ones(n)
    h *= np.outer(gauge, gauge)
    order = np.arange(n)
    if randomize_gauge:
        order = np.roll(order, int(rng.integers(n)))
        if rng.integers(2):
            order = order[::-1]
    h = h[np.ix_(order, order)]
    site_codes = np.array([{"C": 1, "O": 2, "H": 3}[template.sites[i]] for i in order], dtype=np.uint8)
    assert np.allclose(h, h.T)
    assert cycle_phase_from_matrix(h) == (-1 if phase_inversions else 1)
    return h, site_codes, order.astype(np.uint8), gauge.astype(np.int8)


def make_plan(count: int, rng: np.random.Generator) -> list[tuple[Template, str, int]]:
    if count < 1:
        raise ValueError("count must be positive")
    choices = [(template, mode, parity) for template in TEMPLATES for mode, parity in template.modes]
    plan = [choices[i % len(choices)] for i in range(count)]
    rng.shuffle(plan)
    return plan


def generate(count: int = 10_000, seed: int = 20261029, outdir: Path = Path("pericyclic_thermal_dataset")) -> None:
    rng = np.random.default_rng(seed)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plan = make_plan(count, rng)
    h_all = np.zeros((count, MAX_SITES, MAX_SITES), dtype=np.float32)
    mask = np.zeros((count, MAX_SITES), dtype=bool)
    site_codes = np.zeros((count, MAX_SITES), dtype=np.uint8)
    site_order = np.full((count, MAX_SITES), 255, dtype=np.uint8)
    labels = np.empty(count, dtype=np.uint8)
    n_sites = np.empty(count, dtype=np.uint8)
    rows: list[dict] = []
    graph_rows: list[dict] = []
    seen_graphs: dict[str, list[nx.Graph]] = {}

    for idx, (template, mode, parity) in enumerate(plan):
        substitutions, reactants, digest, graph_id = sample_unique_reactant(template, rng, seen_graphs)
        h, codes, order, _ = build_candidate(template, parity, rng,
                                             substitutions=substitutions)
        n = h.shape[0]
        cycle_phase = cycle_phase_from_matrix(h)
        allowed = thermal_allowed(n, cycle_phase)
        eigenvalues = np.linalg.eigvalsh(h)
        homo = float(eigenvalues[n // 2 - 1])
        lumo = float(eigenvalues[n // 2])
        h_all[idx, :n, :n] = h
        mask[idx, :n] = True
        site_codes[idx, :n] = codes
        site_order[idx, :n] = order
        labels[idx] = allowed
        n_sites[idx] = n
        sample_id = f"peri_{idx:05d}"
        graph_rows.append(graph_record(reactants, sample_id))
        rows.append({
            "sample_id": sample_id,
            "reaction_family": template.name,
            "reactant_smiles": reactant_smiles(reactants),
            "parent_reactant_smiles": template.reactants,
            "reactant_graph_id": graph_id,
            "reactant_graph_hash": digest,
            "n_reactant_heavy_atoms": sum(reactants.nodes[i]["element"] != "H" for i in reactants),
            "substituents_by_template_site": json.dumps(substitutions),
            "candidate_approach": mode,
            "electron_sources": template.electron_sources,
            "n_active_orbitals": n,
            "n_active_electrons": n,
            "phase_inversions_mod2": parity,
            "gauge_invariant_cycle_phase": cycle_phase,
            "topology": "Huckel" if cycle_phase > 0 else "Mobius",
            "thermal_allowed": allowed,
            "homo_energy": homo,
            "lumo_energy": lumo,
            "homo_lumo_gap": lumo - homo,
            "orbital_order_from_template": json.dumps(order.tolist()),
            "geometry_checked": False,
            "activation_barrier_checked": False,
        })

    with (outdir / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (outdir / "reactant_graphs.jsonl").open("w", encoding="utf-8") as handle:
        for record in graph_rows:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    np.savez_compressed(outdir / "dataset.npz", H=h_all, y=labels,
                        site_mask=mask, n_sites=n_sites,
                        site_codes=site_codes, site_order=site_order)
    settings = {
        "count": count,
        "seed": seed,
        "label_meaning": "1 = formally thermally symmetry-allowed for the specified candidate approach; 0 = forbidden by ideal Woodward-Hoffmann/Huckel-Mobius rule",
        "matrix_meaning": "generalized Huckel-style active cyclic transition-state orbital interaction matrix; dimensionless beta_CC=-1",
        "padding": "H has shape (count,8,8); use n_sites or site_mask to remove zero padding",
        "site_code": {"0": "PAD", "1": "C active orbital", "2": "O active orbital", "3": "H transfer orbital"},
        "site_alpha": SITE_ALPHA,
        "site_coupling": SITE_COUPLING,
        "substituents": SUBSTITUENTS,
        "substituent_site_shifts": SITE_SHIFT,
        "reactant_graphs": "reactant_graphs.jsonl contains one exact valence-checked graph per matrix; all graphs are non-isomorphic",
        "role_magnitude_ranges": ROLE_RANGE,
        "rule": "phase +1 and 4k+2 electrons OR phase -1 and 4k electrons",
        "gauge": "random orbital sign gauge, cyclic rotation, and reversal; cycle sign is invariant",
        "template_count": len(TEMPLATES),
        "not_assessed": ["transition-state geometry", "reaction kinetics", "synthetic feasibility", "competing mechanisms"],
        "template_sources": {
            "general": "https://www.old.goldbook.iupac.org/html/P/P04491.html",
            "orbital_symmetry": "https://onlinelibrary.wiley.com/doi/10.1002/anie.196907811",
            "ene": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10087545/",
            "electrocyclic": "https://roaldhoffmann.com/sites/default/files/fromd6/18s_1.pdf",
        },
    }
    (outdir / "parameters.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    (outdir / "templates.json").write_text(json.dumps([
        {"name": t.name, "reactants": t.reactants, "sites": t.sites,
         "edge_roles_cyclic_order": t.edge_roles, "modes": t.modes,
         "electron_sources": t.electron_sources} for t in TEMPLATES
    ], indent=2) + "\n", encoding="utf-8")
    plot_overview(rows, outdir / "overview.png")
    plot_matrix_examples(outdir / "matrix_examples.png")
    print(f"Generated {count:,} candidates from {len(TEMPLATES)} reaction families")
    print(f"Thermally allowed: {int(labels.sum()):,}; forbidden: {count - int(labels.sum()):,}")
    print(f"Output: {outdir.resolve()}")


def plot_overview(rows: list[dict], path: Path) -> None:
    names = [t.name for t in TEMPLATES]
    allowed = [sum(r["thermal_allowed"] == 1 for r in rows if r["reaction_family"] == name) for name in names]
    forbidden = [sum(r["thermal_allowed"] == 0 for r in rows if r["reaction_family"] == name) for name in names]
    fig, ax = plt.subplots(figsize=(11, 5.3), layout="constrained")
    x = np.arange(len(names))
    ax.bar(x, forbidden, label="0  thermally forbidden", color="#dc6a66")
    ax.bar(x, allowed, bottom=forbidden, label="1  thermally allowed", color="#2b9b7b")
    ax.set_xticks(x, [name.replace("_", "\n") for name in names], fontsize=9)
    ax.set_ylabel("Candidate approaches")
    ax.set_title("Pericyclic thermal symmetry labels by reaction family")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_matrix_examples(path: Path) -> None:
    examples = (
        (TEMPLATES[0], "supra_supra", 0),
        (TEMPLATES[0], "supra_antara", 1),
        (TEMPLATES[4], "supra_supra", 0),
        (TEMPLATES[4], "supra_antara", 1),
        (TEMPLATES[2], "supra_supra", 0),
        (TEMPLATES[1], "all_suprafacial", 0),
    )
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.3), layout="constrained")
    cmap = LinearSegmentedColormap.from_list("signed", ["#2655a6", "#ffffff", "#d75968"])
    for ax, (template, mode, parity) in zip(axes.flat, examples):
        h, _, _, _ = build_candidate(template, parity, rng, randomize_gauge=False)
        allowed = thermal_allowed(len(template.sites), cycle_phase_from_matrix(h))
        ax.imshow(h, cmap=cmap, vmin=-1.1, vmax=1.1)
        ax.set_xticks(range(len(h)))
        ax.set_yticks(range(len(h)))
        ax.set_title(f"{template.name.replace('_', ' ')}\n{mode.replace('_', ' ')} · label {allowed}", fontsize=10)
        ax.set_xlabel("active orbital")
        ax.set_ylabel("active orbital")
        ax.set_xticks(np.arange(-0.5, len(h), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(h), 1), minor=True)
        ax.grid(which="minor", color="#d8dfe8", lw=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)
    fig.suptitle("Canonical phase examples · blue negative, red positive coupling", fontsize=15)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--outdir", type=Path, default=Path("pericyclic_thermal_dataset"))
    args = parser.parse_args()
    generate(args.count, args.seed, args.outdir)


if __name__ == "__main__":
    main()
