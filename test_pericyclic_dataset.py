"""Checks for reaction rules, matrix gauge, and generated data alignment."""

import csv
import json

import numpy as np

from pericyclic_dataset import (
    TEMPLATES, VALENCE, build_candidate, cycle_phase_from_matrix, generate, thermal_allowed,
)


def test_known_thermal_modes():
    expected = {
        "diels_alder_4plus2": {"supra_supra": 1, "supra_antara": 0},
        "alder_ene": {"all_suprafacial": 1, "one_antarafacial_component": 0},
        "claisen_3_3": {"supra_supra": 1, "supra_antara": 0},
        "cope_3_3": {"supra_supra": 1, "supra_antara": 0},
        "cycloaddition_2plus2": {"supra_supra": 0, "supra_antara": 1},
        "cycloaddition_4plus4": {"supra_supra": 0, "supra_antara": 1},
        "electrocyclic_4pi": {"disrotatory": 0, "conrotatory": 1},
        "electrocyclic_6pi": {"disrotatory": 1, "conrotatory": 0},
    }
    for template in TEMPLATES:
        for mode, parity in template.modes:
            phase = -1 if parity else 1
            assert thermal_allowed(len(template.sites), phase) == expected[template.name][mode]


def test_phase_is_invariant_to_gauge_and_cyclic_relabeling():
    rng = np.random.default_rng(11)
    for template in TEMPLATES:
        for mode, parity in template.modes:
            for _ in range(20):
                h, codes, order, _ = build_candidate(template, parity, rng)
                assert np.allclose(h, h.T)
                assert len(codes) == len(order) == len(template.sites)
                assert cycle_phase_from_matrix(h) == (-1 if parity else 1), mode


def test_small_saved_dataset_aligns_matrices_labels_and_metadata(tmp_path):
    generate(count=64, seed=123, outdir=tmp_path)
    data = np.load(tmp_path / "dataset.npz")
    with (tmp_path / "metadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    with (tmp_path / "reactant_graphs.jsonl").open() as f:
        graphs = [json.loads(line) for line in f]
    assert len(rows) == 64
    assert len(graphs) == 64
    assert len({r["reactant_graph_id"] for r in rows}) == 64
    assert data["H"].shape == (64, 8, 8)
    assert int(data["y"].sum()) == 32
    assert len({data["H"][i].tobytes() for i in range(64)}) == 64
    for i, (row, graph) in enumerate(zip(rows, graphs)):
        n = int(data["n_sites"][i])
        h = data["H"][i, :n, :n]
        phase = cycle_phase_from_matrix(h)
        assert phase == int(row["gauge_invariant_cycle_phase"])
        assert int(data["y"][i]) == int(row["thermal_allowed"])
        assert int(data["y"][i]) == thermal_allowed(n, phase)
        assert data["site_mask"][i, :n].all()
        assert not data["site_mask"][i, n:].any()
        assert graph["sample_id"] == row["sample_id"]
        used = [0] * len(graph["nodes"])
        for a, b, order in graph["bonds"]:
            used[a] += order
            used[b] += order
        for node, valence_used in zip(graph["nodes"], used):
            assert valence_used + node["implicit_h"] == VALENCE[node["element"]]
