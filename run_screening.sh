#!/usr/bin/env bash
# Screen a custom dataset with the trained RP-PROTAC ensemble: embeddings -> inference.
#
#   ./run_screening.sh <input.csv> [output_dir]
#
# The input CSV needs the columns poi, poi_sequence, e3, e3_sequence, poi_ligand, linker and
# e3_ligand. A "label" column is optional: with it the AUROC and AUPR are reported too,
# without it only the predictions are written.
#
# Everything this run produces (embeddings and predictions) stays inside <output_dir>,
# which defaults to ./screening/<input file name>. The published data, weights and
# predictions are never touched, so training is not needed and nothing is overwritten.
set -euo pipefail

cd "$(dirname "$0")"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 <input.csv> [output_dir]"
    echo "       Options beyond this are available with: python screen.py --help"
    exit 1
fi

input="$1"
outdir="${2:-./screening/$(basename "${input%.*}")}"

if [[ ! -f "$input" ]]; then
    echo "Input file not found: $input" >&2
    exit 1
fi

if [[ ! -f weights/rpprotac_ensemble/model_0.pth ]]; then
    echo "No trained ensemble in ./weights/rpprotac_ensemble." >&2
    echo "Train one first with ./run_rpprotac.sh, or pass --model_path to screen.py." >&2
    exit 1
fi

echo "Screening ${input} -> ${outdir}"
python screen.py --data "$input" --outdir "$outdir"
