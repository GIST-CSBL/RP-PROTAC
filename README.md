# RP-PROTAC

Uncertainty-aware prediction of PROTAC-mediated protein degradation with a deep ensemble.

A PROTAC is split into three parts (POI ligand, linker, E3 ligand) and paired with the two
proteins it brings together (the POI and the E3 ligase). Protein and compound embeddings are
combined through cross-attention, and each ensemble member predicts both a degradation logit
and its aleatoric uncertainty.

## Environment

All training and inference were performed on an **NVIDIA GeForce RTX 4090** (CUDA 12.1).

Create the conda environment directly:

```bash
conda env create -f environment.yml
conda activate rpprotac
```

or create a Python 3.10 environment yourself and install the dependencies:

```bash
conda create -n rpprotac python=3.10 -y
conda activate rpprotac
pip install -r requirements.txt
```

Key dependencies: Python 3.10, PyTorch 2.1.2 (CUDA 12.1), PyTorch Geometric 2.4.0,
RDKit 2024.9.1, Transformers 4.51.3.

Both files pin the CUDA 12.1 builds of PyTorch and the matching PyTorch Geometric extensions,
and already point to the wheel servers that host them (`download.pytorch.org`, `data.pyg.org`).
On a GPU that CUDA 12.1 does not support, install the builds matching your own CUDA version
instead.

## Usage

Run the three steps below in order, or run them all at once with:

```bash
./run_rpprotac.sh
```

(To run the trained model on your own data instead, see
[Screening a custom dataset](#screening-a-custom-dataset).)

The script skips any step whose outputs already exist, so it goes straight to inference once
the embeddings and the trained weights are in place. It runs each step with its defaults; to
change anything, run the steps yourself with the options below. Every script documents its own
options under `--help`.

### 1. Generate embeddings

```bash
python data_processing.py
```

Encodes every protein and compound in `data/curated_data.csv`:

- compounds with the pretrained LEMON GNN (`weights/pretrained_modules/lemon_100_a1.0_b1.0.pt`)
  into atom-level embeddings (300-dim)
- proteins with ESM-2 `facebook/esm2_t33_650M_UR50D` into residue-level embeddings (1280-dim),
  sequences truncated to 1022 residues

Writes `data/lemon_compound_embedding.pt` and `data/esm2_protein_embedding.pt`.
The ESM-2 weights are downloaded from Hugging Face on the first run.

Main options: `--data`, `--compound_output`, `--protein_output`, `--lemon_weights`,
`--esm2_model`, `--device`.

### 2. Train

```bash
python train_ensemble.py
```

Trains 10 models on `data/train.csv` with different random initializations, each on the same
90% / 10% train-validation split. Training uses a positive class weight for the label
imbalance, label smoothing, and early stopping on validation AUROC (patience 20). The best
checkpoint of each member is saved to `weights/rpprotac_ensemble/model_{0..9}.pth`.

Main options: `--train_data`, `--save_dir`, `--num_seeds`, `--num_epochs`, `--batch_size`,
`--lr`, `--l2`, `--early_stop`, `--device`.

Use `--save_dir` to keep several ensembles side by side instead of overwriting the previous one:

```bash
python train_ensemble.py --num_seeds 5 --save_dir weights/rpprotac_5seeds
```

### 3. Evaluate

```bash
python test_ensemble.py
```

Evaluates the ensemble on the four held-out splits — **Unseen Pair**, **Unseen POI**,
**Unseen PROTAC** and **Unseen Both** — and prints AUPR / AUROC both per member
(mean ± std) and for the averaged ensemble prediction. Per-sample predictions, including the
aleatoric variance of each one, are written to `predictions/rpprotac_ensemble.csv`.

Point `--model_path` at the ensemble to evaluate and `--output` at where the predictions
should go:

```bash
python test_ensemble.py \
    --model_path weights/rpprotac_5seeds \
    --num_seeds 5 \
    --output predictions/rpprotac_5seeds.csv
```

Main options: `--model_path`, `--output`, `--unseen_{pair,poi,protac,both}`, `--num_seeds`,
`--batch_size`, `--device`.

Note that `--protein_dim`, `--compound_dim`, `--emb_dim` and `--num_heads` must match the
values the checkpoints were trained with, otherwise loading them fails.

## Screening a custom dataset

To run the trained ensemble over your own PROTACs, use the screening pipeline instead. It does
embedding generation and inference only — no training — and keeps everything it produces inside
its own output directory, so the published data, weights and predictions are never overwritten.

```bash
./run_screening.sh my_protacs.csv
```

Results land in `screening/my_protacs/`: the embeddings for your dataset, plus
`predictions.csv`. A second run on the same directory reuses the cached embeddings
(`--recompute` rebuilds them). Pass a second argument to choose the output directory, or call
`python screen.py --help` for the full set of options.

The input CSV needs these columns:

| column | contents |
| --- | --- |
| `poi`, `e3` | protein names, used as the embedding keys |
| `poi_sequence`, `e3_sequence` | amino acid sequences |
| `poi_ligand`, `linker`, `e3_ligand` | SMILES of the three PROTAC parts |
| `label` | **optional**, 0 or 1 |

`label` decides what you get back:

- **with `label`** — predictions plus an AUPR / AUROC report
- **without `label`** — predictions only

Either way `predictions.csv` holds every input column followed by:

| column | meaning |
| --- | --- |
| `Prob` | predicted degradation probability, averaged over the ensemble |
| `Aleatoric_Var` | uncertainty the members agree on (noise in the data) |
| `Epistemic_Var` | disagreement between members (uncertainty of the model itself) |

## Repository structure

```
rpprotac/
├── data/
│   ├── curated_data.csv                   # full curated dataset (used for embedding generation)
│   ├── train.csv                          # training set
│   ├── unseen_{pair,poi,protac,both}.csv  # test sets
│   ├── external_*.csv                     # external test sets
│   └── *.pt                               # precomputed embeddings (step 1 output)
├── weights/
│   ├── pretrained_modules/                # pretrained LEMON compound encoder
│   └── rpprotac_ensemble/                 # trained ensemble members (model_0.pth ~ model_9.pth)
├── predictions/                           # model outputs
├── src/
│   ├── lemon_model.py                     # LEMON GNN (graph + line graph) compound encoder
│   ├── protac_model.py                    # RP-PROTAC cross-attention predictor
│   └── utils.py                           # dataset, collate function, seeding
├── data_processing.py                     # step 1: embedding generation
├── train_ensemble.py                      # step 2: ensemble training
├── test_ensemble.py                       # step 3: evaluation
├── screen.py                              # inference on a custom dataset
├── run_rpprotac.sh                        # runs all three steps
└── run_screening.sh                       # runs screening on a custom dataset
```

## License
This project adopts a dual-licensing structure to distinguish source code from research data:
- **Source Code**: Licensed under the [GNU General Public License v3.0 (GPL-3.0)](LICENSE).
- **Dataset**: All data files located in the `data/` directory are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](data/LICENSE).
