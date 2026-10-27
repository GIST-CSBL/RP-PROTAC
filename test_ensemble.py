"""
Evaluate the RP-PROTAC deep ensemble on the four test splits
(Unseen Pair, Unseen POI, Unseen PROTAC, Unseen Both).

Loads `--model_path`/model_{i}.pth, prints AUPR / AUROC of each member
(mean ± std, "Deterministic") and of the averaged ensemble prediction ("Ensemble"),
and saves per-sample predictions to `--output`.
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from torch.utils.data import DataLoader
from src import Predictor, PROTACDataset, protac_collate_fn, set_seed
import warnings
warnings.filterwarnings(action='ignore')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the RP-PROTAC deep ensemble on the four held-out test splits.")

    # data and paths
    parser.add_argument("--unseen_pair", type=str, default="./data/unseen_pair.csv",
                        help="test set whose POI-PROTAC pairs are unseen (default: %(default)s)")
    parser.add_argument("--unseen_poi", type=str, default="./data/unseen_poi.csv",
                        help="test set whose POIs are unseen (default: %(default)s)")
    parser.add_argument("--unseen_protac", type=str, default="./data/unseen_protac.csv",
                        help="test set whose PROTACs are unseen (default: %(default)s)")
    parser.add_argument("--unseen_both", type=str, default="./data/unseen_both.csv",
                        help="test set whose POIs and PROTACs are both unseen (default: %(default)s)")
    parser.add_argument("--compound_embedding", type=str, default="./data/lemon_compound_embedding.pt",
                        help="compound embeddings produced by data_processing.py (default: %(default)s)")
    parser.add_argument("--protein_embedding", type=str, default="./data/esm2_protein_embedding.pt",
                        help="protein embeddings produced by data_processing.py (default: %(default)s)")
    parser.add_argument("--model_path", type=str, default="./weights/rpprotac_ensemble",
                        help="directory holding model_0.pth ... model_{num_seeds-1}.pth, i.e. the --save_dir "
                             "used for training (default: %(default)s)")
    parser.add_argument("--output", type=str, default="./predictions/rpprotac_ensemble.csv",
                        help="where to write the per-sample predictions (default: %(default)s)")

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
                        help="dropout probability; only affects model construction, as inference runs in "
                             "eval mode (default: %(default)s)")

    # inference
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="number of ensemble members to load (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for the noise drawn during uncertainty sampling (default: %(default)s)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="inference batch size (default: %(default)s)")
    parser.add_argument("--sampling_T", type=int, default=30,
                        help="how many noisy logits to draw per sample; their mean is the predicted "
                             "probability and their variance the aleatoric uncertainty (default: %(default)s)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes (default: %(default)s)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device to run inference on (default: %(default)s)")
    return parser.parse_args()

def main(args):
    unseen_pair = pd.read_csv(args.unseen_pair, keep_default_na=False)
    unseen_poi = pd.read_csv(args.unseen_poi, keep_default_na=False)
    unseen_protac = pd.read_csv(args.unseen_protac, keep_default_na=False)
    unseen_both = pd.read_csv(args.unseen_both, keep_default_na=False)

    compound_embedding = torch.load(args.compound_embedding)
    protein_embedding = torch.load(args.protein_embedding)
    dataset_list = [unseen_pair, unseen_poi, unseen_protac, unseen_both]
    dataset_name = ["Unseen Pair", "Unseen POI", "Unseen PROTAC", "Unseen Both"]
    set_seed(args.seed)
    outputs = pd.DataFrame(columns=["Dataset", "Seed", "Prob", "Var", "Label"])
    for dataset, name in zip(dataset_list, dataset_name):
        data_set = PROTACDataset(dataset, protein_embedding, compound_embedding)
        data_loader = DataLoader(data_set, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=protac_collate_fn)
        loop = tqdm(range(args.num_seeds))
        for seed in loop:
            loop.set_description(f"Dataset: {name} | Seed: {seed+1} proceeding...")
            model = Predictor(args.protein_dim, args.compound_dim, args.emb_dim, args.num_heads, args.dropout_rate).to(args.device)
            model.load_state_dict(torch.load(os.path.join(args.model_path, f"model_{seed}.pth"), map_location=args.device))
            model.eval()
            prob_list = []
            var_list = []
            label_list = []
            with torch.no_grad():
                for data in data_loader:
                    poi, poi_ligand, linker, e3_ligand, e3, label = data
                    output, std, _ = model(poi.to(args.device), poi_ligand.to(args.device), linker.to(args.device), e3_ligand.to(args.device), e3.to(args.device))
                    # sample T noisy logits ~ N(output, std^2); mean -> predicted probability, variance -> aleatoric uncertainty
                    eps = torch.randn(size=(args.sampling_T,)+std.shape, device=std.device)
                    output = torch.sigmoid(output.unsqueeze(0) + std.unsqueeze(0) * eps)
                    prob_list.extend(output.mean(0).squeeze(-1).tolist())
                    var_list.extend(output.var(0).squeeze(-1).tolist())
                    label_list += label.float()
            new_output = pd.DataFrame({"Dataset": name,
                                       "Seed": seed,
                                       "Prob": prob_list,
                                       "Var": var_list,
                                       "Label": torch.tensor(label_list).tolist()})
            outputs = pd.concat([outputs, new_output]).reset_index(drop=True)
    # sample index within each (Dataset, Seed) group, used to align predictions across ensemble members
    outputs["Index"] = outputs.groupby(["Dataset", "Seed"]).cumcount() + 1
    return outputs

if __name__ == "__main__":
    args = parse_args()
    ensemble_predictions = main(args)
    # performance of each ensemble member (mean ± std over members)
    print("Deterministic performance")
    for dataset in ["Unseen Pair", "Unseen POI", "Unseen PROTAC", "Unseen Both"]:
        aurocs = []
        auprs = []
        for seed in ensemble_predictions.Seed.unique():
            data = ensemble_predictions[(ensemble_predictions["Dataset"]==dataset)&(ensemble_predictions["Seed"]==seed)]
            auprs.append(average_precision_score(data.Label, data.Prob))
            aurocs.append(roc_auc_score(data.Label, data.Prob))
        print(f"{dataset:15s}: AUPR {np.mean(auprs):.3f} ± {np.std(auprs):.3f} | AUROC {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f}")
    print("====================================")
    # performance of the ensemble (probabilities averaged over members)
    print("Ensemble performance")
    for dataset in ["Unseen Pair", "Unseen POI", "Unseen PROTAC", "Unseen Both"]:
        data = ensemble_predictions[ensemble_predictions["Dataset"]==dataset]
        probs = data.groupby("Index").Prob.mean()
        labels = data.groupby("Index").Label.mean()
        print(f"{dataset:15s}: AUPR {average_precision_score(labels, probs):.3f} | AUROC {roc_auc_score(labels, probs):.3f}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ensemble_predictions.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")
