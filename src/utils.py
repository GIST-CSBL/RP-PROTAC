import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class PROTACDataset(Dataset):
    """
    Looks up precomputed embeddings (from data_processing.py) for each
    (POI, POI ligand, linker, E3 ligand, E3) row of a dataset.
    A dataset without a "label" column (unlabelled screening data) yields -1 as the label.
    """
    def __init__(self, dataset, protein_embedding, compound_embedding):
        self.dataset = dataset
        self.protein_embedding = protein_embedding
        self.compound_embedding = compound_embedding
        self.has_label = "label" in dataset.columns

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        poi = self.protein_embedding[self.dataset.loc[idx, "poi"]]
        poi_ligand = self.compound_embedding[self.dataset.loc[idx, "poi_ligand"]]
        linker = self.compound_embedding[self.dataset.loc[idx, "linker"]]
        e3_ligand = self.compound_embedding[self.dataset.loc[idx, "e3_ligand"]]
        e3 = self.protein_embedding[self.dataset.loc[idx, "e3"]]
        label = self.dataset.loc[idx, "label"] if self.has_label else -1
        return poi, poi_ligand, linker, e3_ligand, e3, label

def protac_collate_fn(batch):
    # zero-pad variable-length embeddings; padded positions are masked in the model
    poi_list, poi_ligand_list, linker_list, e3_ligand_list, e3_list, label_list = zip(*batch)
    poi_batch = pad_sequence(poi_list, batch_first=True, padding_value=0)
    poi_ligand_batch = pad_sequence(poi_ligand_list, batch_first=True, padding_value=0)
    linker_batch = pad_sequence(linker_list, batch_first=True, padding_value=0)
    e3_ligand_batch = pad_sequence(e3_ligand_list, batch_first=True, padding_value=0)
    e3_batch = pad_sequence(e3_list, batch_first=True, padding_value=0)
    label_batch = torch.tensor(label_list, dtype=torch.float32)
    return poi_batch, poi_ligand_batch, linker_batch, e3_ligand_batch, e3_batch, label_batch

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
