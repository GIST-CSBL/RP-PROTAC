"""
Train the RP-PROTAC deep ensemble.

Trains `--num_seeds` models with different random initializations on the training set
(90% train / 10% validation split) and saves the best checkpoint (by validation AUROC)
of each model to `--save_dir`/model_{i}.pth.
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import argparse
import copy
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src import Predictor, PROTACDataset, protac_collate_fn, set_seed
import warnings
warnings.filterwarnings(action='ignore')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the RP-PROTAC deep ensemble and save one checkpoint per member.")

    # data and paths
    parser.add_argument("--train_data", type=str, default="./data/train.csv",
                        help="training set CSV, split 90/10 into train and validation (default: %(default)s)")
    parser.add_argument("--compound_embedding", type=str, default="./data/lemon_compound_embedding.pt",
                        help="compound embeddings produced by data_processing.py (default: %(default)s)")
    parser.add_argument("--protein_embedding", type=str, default="./data/esm2_protein_embedding.pt",
                        help="protein embeddings produced by data_processing.py (default: %(default)s)")
    parser.add_argument("--save_dir", type=str, default="./weights/rpprotac_ensemble",
                        help="directory to write model_0.pth ... model_{num_seeds-1}.pth into. "
                             "Point this somewhere new to keep an earlier ensemble (default: %(default)s)")

    # model
    parser.add_argument("--protein_dim", type=int, default=1280,
                        help="size of one protein embedding, i.e. the width of the ESM-2 model used "
                             "in data_processing.py (default: %(default)s)")
    parser.add_argument("--compound_dim", type=int, default=300,
                        help="size of one compound embedding, i.e. the width of the LEMON encoder (default: %(default)s)")
    parser.add_argument("--emb_dim", type=int, default=512,
                        help="width every input is projected to before cross-attention; must be divisible "
                             "by --num_heads (default: %(default)s)")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of cross-attention heads (default: %(default)s)")
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                        help="dropout probability in the MLP header (default: %(default)s)")

    # training
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="number of ensemble members to train (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42,
                        help="master seed: fixes the train/validation split and draws the per-member seeds "
                             "(default: %(default)s)")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="maximum epochs per member; early stopping usually ends training sooner "
                             "(default: %(default)s)")
    parser.add_argument("--early_stop", type=int, default=20,
                        help="stop a member after this many epochs without a better validation AUROC "
                             "(default: %(default)s)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="training and validation batch size (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="initial Adam learning rate, multiplied by 0.1 when validation AUROC stops "
                             "improving for 5 epochs (default: %(default)s)")
    parser.add_argument("--l2", type=float, default=1e-4,
                        help="Adam weight decay (default: %(default)s)")
    parser.add_argument("--smoothing_factor", type=float, default=0.05,
                        help="label smoothing: labels 0 and 1 become s and 1-s (default: %(default)s)")
    parser.add_argument("--sampling_T", type=int, default=30,
                        help="how many noisy logits to draw per sample so the model learns its own "
                             "aleatoric uncertainty (default: %(default)s)")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader worker processes (default: %(default)s)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device to train on (default: %(default)s)")
    return parser.parse_args()

def main(args):
    print(f"Protein dim: {args.protein_dim} | Compound dim: {args.compound_dim} | Emb dim: {args.emb_dim} | Heads: {args.num_heads}")
    trainset = pd.read_csv(args.train_data, keep_default_na=False)
    compound_embedding = torch.load(args.compound_embedding)
    protein_embedding = torch.load(args.protein_embedding)

    models = []
    
    random.seed(args.seed)
    train_set, valid_set = train_test_split(trainset, test_size=0.1, random_state=args.seed, stratify=trainset["label"])
    train_dataset = PROTACDataset(train_set.reset_index(drop=True), protein_embedding, compound_embedding)
    valid_dataset = PROTACDataset(valid_set.reset_index(drop=True), protein_embedding, compound_embedding)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=protac_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=protac_collate_fn)
    seeds = random.sample(range(1000), args.num_seeds) # one random seed per ensemble member
    for seed_idx, seed in enumerate(seeds):
        seed_best_model = None
        seed_best_auc = 0
        set_seed(seed)
        model = Predictor(args.protein_dim, args.compound_dim, args.emb_dim, args.num_heads, args.dropout_rate).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.1)
        # weight positives by neg/pos ratio to handle class imbalance
        pos = trainset.label.sum()
        neg = len(trainset) - pos
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/pos))

        early_stop = args.early_stop
        
        loop = tqdm(range(args.num_epochs))
        for epoch in loop:
            loop.set_description(f"Seed: {seed_idx+1} / {len(seeds)} | Epoch: {epoch+1} / {args.num_epochs}")
            # train
            model.train()
            train_loss = 0.0
            for _, data in enumerate(train_loader):
                optimizer.zero_grad()
                poi, poi_ligand, linker, e3_ligand, e3, label = data
                output, std, _ = model(poi.to(args.device), poi_ligand.to(args.device), linker.to(args.device), e3_ligand.to(args.device), e3.to(args.device))
                # label smoothing: {0, 1} -> {s, 1-s}
                label = label * (1-args.smoothing_factor*2) + args.smoothing_factor
                # aleatoric uncertainty: sample T noisy logits ~ N(output, std^2) and average the BCE loss
                eps = torch.randn(size=(args.sampling_T,)+std.shape, device=std.device)
                output = output.unsqueeze(0) + std.unsqueeze(0) * eps
                loss = criterion(output.cpu(), label.view(-1, 1).unsqueeze(0).expand_as(output))
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss /= len(train_loader)

            # validation (uses the mean logit only)
            model.eval()
            pred_list = []
            label_list = []
            valid_loss = 0.0
            with torch.no_grad():
                for _, data in enumerate(valid_loader):
                    poi, poi_ligand, linker, e3_ligand, e3, label = data
                    output, std, _ = model(poi.to(args.device), poi_ligand.to(args.device), linker.to(args.device), e3_ligand.to(args.device), e3.to(args.device))
                    loss = criterion(output.cpu(), label.view(-1, 1))
                    valid_loss += loss.item()
                    pred_list += output.cpu().detach()
                    label_list += label
            valid_loss /= len(valid_loader)
            valid_auc = roc_auc_score(label_list, pred_list)
            
            scheduler.step(valid_auc)
            loop.set_postfix(Train_loss=np.round(train_loss, 4), Valid_loss=np.round(valid_loss, 4), Valid_AUC=np.round(valid_auc, 4), LR=np.round(optimizer.param_groups[0]["lr"], 6))

            # keep the best model by validation AUROC; stop if no improvement for `early_stop` epochs
            if valid_auc <= seed_best_auc:
                early_stop -= 1
            elif valid_auc > seed_best_auc:
                seed_best_auc = valid_auc
                early_stop = args.early_stop
                seed_best_model = copy.deepcopy(model)
                
            if early_stop <= 0:
                break
        models.append(seed_best_model)
    torch.cuda.empty_cache()
    return models

if __name__ == "__main__":
    args = parse_args()
    models = main(args)
    os.makedirs(args.save_dir, exist_ok=True)
    for i in range(len(models)):
        torch.save(models[i].state_dict(), os.path.join(args.save_dir, f"model_{i}.pth"))
    print(f"Saved {len(models)} models to {args.save_dir}")
