#!/usr/bin/env python3
"""Independently sanitize/compare the exact bond graphs drawn in the figures."""
import json
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def make_graph(elements, orders):
    rw = Chem.RWMol()
    for element in elements:
        rw.AddAtom(Chem.Atom(element))
    for i, order in enumerate(orders):
        if order:
            rw.AddBond(i, (i + 1) % len(orders),
                       Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return Chem.RemoveHs(mol)


def main():
    manifest = json.loads((HERE / "orbital_manifest.json").read_text())
    templates = {t["name"]: t for t in json.loads(
        (ROOT / "pericyclic_thermal_dataset/templates.json").read_text())}
    records = []
    for reaction in manifest["reaction_bonds"]:
        name = reaction["family"]
        r = make_graph(reaction["elements"], reaction["reactant_cyclic_bond_orders"])
        p = make_graph(reaction["elements"], reaction["product_cyclic_bond_orders"])
        rs, ps = Chem.MolToSmiles(r), Chem.MolToSmiles(p)
        expected_r = Chem.MolToSmiles(Chem.MolFromSmiles(templates[name]["reactants"]))
        expected_p = Chem.MolToSmiles(Chem.MolFromSmiles(reaction["expected_product_smiles"]))
        assert rs == expected_r, (name, rs, expected_r)
        assert ps == expected_p, (name, ps, expected_p)
        rf, pf = rdMolDescriptors.CalcMolFormula(r), rdMolDescriptors.CalcMolFormula(p)
        assert rf == pf, (name, rf, pf)
        records.append({"family": name, "reactant_smiles": rs, "product_smiles": ps,
                        "conserved_formula": pf, "valence_sanitized": True})
        print(name, ps, pf)
    (HERE / "product_validation.json").write_text(json.dumps(
        {"rdkit_version": rdBase.rdkitVersion, "reactions": records}, indent=2) + "\n")
    print("Validated all 8 reactant/product graphs and conserved molecular formulas.")


if __name__ == "__main__":
    main()
