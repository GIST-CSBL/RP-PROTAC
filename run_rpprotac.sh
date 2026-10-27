#!/usr/bin/env bash
# Run the full RP-PROTAC pipeline: embedding generation -> training -> evaluation.
# Steps whose outputs already exist are skipped.
set -euo pipefail

cd "$(dirname "$0")"

# 1. Protein / compound embeddings
if [[ -f data/esm2_protein_embedding.pt && -f data/lemon_compound_embedding.pt ]]; then
    echo "[1/3] Embeddings found in ./data. Skipping data_processing.py"
else
    echo "[1/3] Generating protein and compound embeddings..."
    python data_processing.py
fi

# 2. Ensemble training (10 members: model_0.pth ~ model_9.pth)
shopt -s nullglob
models=(weights/rpprotac_ensemble/model_*.pth)
num_models=${#models[@]}
shopt -u nullglob
if [[ "$num_models" -ge 10 ]]; then
    echo "[2/3] ${num_models} trained models found in ./weights/rpprotac_ensemble. Skipping train_ensemble.py"
else
    echo "[2/3] Training RP-PROTAC ensemble..."
    python train_ensemble.py
fi

# 3. Evaluation on the test sets
echo "[3/3] Evaluating RP-PROTAC ensemble..."
python test_ensemble.py
