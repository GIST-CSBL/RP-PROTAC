"""
Screen a custom dataset with a trained RP-PROTAC ensemble.

Generates the embeddings for the input CSV (cached in --outdir, so a second run reuses them)
and predicts a degradation probability for every row. Nothing under ./data, ./weights or
./predictions is written to, so a screening run never overwrites the published results.

The input CSV needs the columns poi, poi_sequence, e3, e3_sequence, poi_ligand, linker and
e3_ligand. A "label" column is optional: when it is present, AUROC and AUPR are reported as
well.
"""
import os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from torch.utils.data import DataLoader
from data_processing import build_compound_embeddings, build_protein_embeddings, save_embedding
from src import Predictor, PROTACDataset, protac_collate_fn, set_seed
import warnings
warnings.filterwarnings(action='ignore')

REQUIRED_COLUMNS = ["poi", "poi_sequence", "e3", "e3_sequence", "poi_ligand", "linker", "e3_ligand"]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Screen a custom dataset with a trained RP-PROTAC ensemble.")

    # data and paths
    parser.add_argument("--data", type=str, required=True,
                        help="CSV to screen. Must contain the columns poi, poi_sequence, e3, e3_sequence, "
                             "poi_ligand, linker and e3_ligand; a label column is optional and enables "
                             "the AUROC/AUPR report")
    parser.add_argument("--outdir", type=str, default="./screening",
                        help="directory for this run's embeddings and predictions (default: %(default)s)")
    parser.add_argument("--output", type=str, default=None,
                        help="where to write the predictions (default: <outdir>/predictions.csv)")
    parser.add_argument("--model_path", type=str, default="./weights/rpprotac_ensemble",
                        help="directory holding model_0.pth ... model_{num_seeds-1}.pth (default: %(default)s)")
    parser.add_argument("--lemon_weights", type=str, default="./weights/pretrained_modules/lemon_100_a1.0_b1.0.pt",
                        help="checkpoint of the pretrained LEMON compound encoder (default: %(default)s)")
    parser.add_argument("--esm2_model", type=str, default="facebook/esm2_t33_650M_UR50D",
                        help="ESM-2 model used to encode proteins; must be the one the ensemble was "
                             "trained with (default: %(default)s)")
    parser.add_argument("--recompute", action="store_true",
                        help="regenerate the embeddings even if they are already cached in --outdir")

    # model: must match the values the checkpoints were trained with
    parser.add_argument("--protein_dim", type=int, default=1280,
                        help="size of one protein embedding (default: %(default)s)")
    parser.add_argument("--compound_dim", type=int, default=300,
                        help="size of one compound embedding (default: %(default)s)")
    parser.add_argument("--emb_dim", type=int, default=512,
                        help="width every input is projected to before cross-attention (default: %(default)s)")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of cross-attention heads (default: %(default)s)")
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                        help="dropout probability; inference runs in eval mode (default: %(default)s)")

    # inference
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="number of ensemble members to load (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for the noise drawn during uncertainty sampling (default: %(default)s)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="inference batch size (default: %(default)s)")
    parser.add_argument("--sampling_T", type=int, default=30,
                        help="how many noisy logits to draw per sample and member (default: %(default)s)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes (default: %(default)s)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device to run on (default: %(default)s)")
    return parser.parse_args()

def load_embeddings(args, data):
    """Build the embeddings for this dataset, or reuse the ones cached in --outdir."""
    compound_path = os.path.join(args.outdir, "lemon_compound_embedding.pt")
    protein_path = os.path.join(args.outdir, "esm2_protein_embedding.pt")

    if not args.recompute and os.path.isfile(compound_path) and os.path.isfile(protein_path):
        print(f"Reusing cached embeddings in {args.outdir} (pass --recompute to rebuild them)")
        return torch.load(compound_path), torch.load(protein_path)

    compound_embedding = build_compound_embeddings(data, args.lemon_weights, args.device)
    save_embedding(compound_embedding, compound_path)
    protein_embedding = build_protein_embeddings(data, args.esm2_model, args.device)
    save_embedding(protein_embedding, protein_path)
    return compound_embedding, protein_embedding

def predict(args, data, compound_embedding, protein_embedding):
    """
    Run every ensemble member over the dataset.
    :return: probabilities and aleatoric variances, both shaped (num_seeds, num_samples)
    """
    dataset = PROTACDataset(data, protein_embedding, compound_embedding)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=protac_collate_fn)

    probs = []
    variances = []
    for seed in tqdm(range(args.num_seeds), desc="Screening"):
        model = Predictor(args.protein_dim, args.compound_dim, args.emb_dim, args.num_heads, args.dropout_rate).to(args.device)
        model.load_state_dict(torch.load(os.path.join(args.model_path, f"model_{seed}.pth"), map_location=args.device))
        model.eval()
        seed_probs = []
        seed_variances = []
        with torch.no_grad():
            for poi, poi_ligand, linker, e3_ligand, e3, _ in loader:
                output, std, _ = model(poi.to(args.device), poi_ligand.to(args.device), linker.to(args.device), e3_ligand.to(args.device), e3.to(args.device))
                # sample T noisy logits ~ N(output, std^2); mean -> probability, variance -> aleatoric uncertainty
                eps = torch.randn(size=(args.sampling_T,)+std.shape, device=std.device)
                output = torch.sigmoid(output.unsqueeze(0) + std.unsqueeze(0) * eps)
                seed_probs.extend(output.mean(0).squeeze(-1).cpu().tolist())
                seed_variances.extend(output.var(0).squeeze(-1).cpu().tolist())
        probs.append(seed_probs)
        variances.append(seed_variances)
    return np.array(probs), np.array(variances)

def main(args):
    data = pd.read_csv(args.data, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise SystemExit(f"{args.data} is missing the required column(s): {', '.join(missing)}")
    data = data.reset_index(drop=True)

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)
    compound_embedding, protein_embedding = load_embeddings(args, data)
    probs, variances = predict(args, data, compound_embedding, protein_embedding)

    # ensemble prediction, with uncertainty split into its two sources
    results = data.copy()
    results["Prob"] = probs.mean(axis=0)
    results["Aleatoric_Var"] = variances.mean(axis=0)   # noise the members agree on
    results["Epistemic_Var"] = probs.var(axis=0)        # disagreement between members

    output = args.output or os.path.join(args.outdir, "predictions.csv")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    results.to_csv(output, index=False)

    # labels are optional: report performance only when they are there
    if "label" in results.columns:
        labels = results["label"].astype(float)
        if labels.nunique() < 2:
            print("Only one class present in the label column, so AUROC and AUPR are undefined.")
        else:
            print(f"AUPR {average_precision_score(labels, results.Prob):.3f} | AUROC {roc_auc_score(labels, results.Prob):.3f}")
    else:
        print("No label column found, so predictions were written without a performance report.")
    print(f"Screened {len(results)} rows -> {output}")

if __name__ == "__main__":
    main(parse_args())
