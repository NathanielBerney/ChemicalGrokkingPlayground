# Implementation provenance for the molecular representation probe

This file records what the local implementation establishes. It does **not** assign an upstream paper to code when the repository has no evidence for one.

| Component | Evidence in this repository | Recorded upstream implementation |
|---|---|---|
| Canonical SMILES strings | [`HeteroatomsScript.py`](../../HeteroatomsScript.py) calls RDKit `Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)` before writing the dataset. | RDKit generates the strings. |
| SMILES tokenization and IDs | [`prepare.py`](prepare.py) defines `TOKEN_RE`, splits each stored string, builds IDs by sorting the tokens found in this dataset, and reserves `PAD=0` and `CLS=1`. The resulting mapping is [`vocab.json`](vocab.json). | No external tokenizer, ChemBERTa vocabulary, or tokenizer checkpoint is loaded or attributed. The historical inspiration for the regex is not recorded. |
| SMILES prediction model | [`train.py`](train.py) defines `SmilesTransformer` using `torch.nn.Embedding` and `torch.nn.TransformerEncoder` (two layers, four heads), followed by a regression head. | No ChemBERTa model or pretrained weights are loaded. No paper or external codebase is cited as the source of this exact implementation. |
| π-site graph model | [`train.py`](train.py) defines `PiGraphLayer` and `PiGraph` using PyTorch linear layers, `torch.bmm`, normalization, and six message-passing rounds. Node inputs are `[H_ii/2, site_pi_electrons/2, nonzero_H_degree/4]`; off-diagonal Hückel entries are edge weights. | No PyG/DGL layer, pretrained model, external GNN code, or specific architecture paper is imported or cited. |
| Full atom graph model | [`prepare.py`](prepare.py) builds 18-channel atom features and bond/π-coupling arrays from the SDF and Hückel data. [`train.py`](train.py) defines `GraphLayer` and `MolecularGraph` using four custom message-passing rounds. | No external GNN implementation or paper is recorded as the source of this exact model. |

The three papers previously supplied in conversation (Weininger on SMILES, Gilmer et al. on message passing, and Vaswani et al. on transformers) are **background references**. The available files do not establish that their code or tokenizers were used to implement these experiments. They should not be presented as implementation citations.

This OPSandbox directory has no Git repository or commit history at this path, so source-file history cannot be reconstructed from Git here. Absence of a recorded attribution is not proof that a design had no external inspiration. For this experiment, the reproducible implementation references are the local files linked above.
