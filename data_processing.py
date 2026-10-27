"""
Generate input embeddings for RP-PROTAC.

- Compounds (PROTAC, POI ligand, linker, E3 ligand): atom-level embeddings from the
  pretrained LEMON GNN (./weights/pretrained_modules/lemon_100_a1.0_b1.0.pt)
- Proteins (POI, E3 ligase): residue-level embeddings from ESM-2 (facebook/esm2_t33_650M_UR50D)

Outputs (paths are configurable, see --help):
- ./data/lemon_compound_embedding.pt : {SMILES: Tensor(num_atoms, 300)}
- ./data/esm2_protein_embedding.pt   : {protein name: Tensor(num_tokens, 1280)} (including <cls>/<eos> tokens)
"""
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import OptTensor
from torch_sparse import coalesce
from torch_scatter import scatter_add
from transformers import AutoTokenizer, AutoModel
from src import GNN

# allowable node and edge features (indices are used as embedding ids in LEMON)
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def smiles_to_graph(smiles):
    """
    Converts a SMILES string to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param smiles: SMILES string
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    mol = Chem.MolFromSmiles(smiles)
    # atoms: atom type, chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

class LGData(Data):
    # Data object holding both the molecular graph and its line graph.
    # lg_edge_index is offset by lg_num_nodes (not num_nodes) when batching.
    def __init__(self, x: OptTensor=None, edge_index: OptTensor=None,
                 edge_attr: OptTensor=None, y: OptTensor=None,
                 pos: OptTensor=None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'lg_edge_index':
            return self.lg_num_nodes
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        return super().__cat_dim__(key, value, *args, **kwargs)
        
class LineGraph(BaseTransform):
    r"""Converts a graph to its corresponding line-graph:

    .. math::
        L(\mathcal{G}) &= (\mathcal{V}^{\prime}, \mathcal{E}^{\prime})

        \mathcal{V}^{\prime} &= \mathcal{E}

        \mathcal{E}^{\prime} &= \{ (e_1, e_2) : e_1 \cap e_2 \neq \emptyset \}

    Line-graph node indices are equal to indices in the original graph's
    coalesced :obj:`edge_index`.
    For undirected graphs, the maximum line-graph node index is
    :obj:`(data.edge_index.size(1) // 2) - 1`.

    New node features are given by old edge attributes.
    For undirected graphs, edge attributes for reciprocal edges
    :obj:`(row, col)` and :obj:`(col, row)` get summed together.

    Args:
        force_directed (bool, optional): If set to :obj:`True`, the graph will
            be always treated as a directed graph. (default: :obj:`False`)
    """
    def __init__(self, force_directed=False):
        self.force_directed = force_directed

    def __call__(self, data):
        lgd = LGData()
        # copy graph data
        for key in data.keys():
            lgd[key] = data[key]
        # no edge
        if lgd.edge_index.shape[1] == 0:
            lgd.lg_edge_index_map = torch.tensor([range(lgd.num_nodes), range(lgd.num_nodes)],
                                                 dtype=data.edge_index.dtype)  # lg_node = g_edge
            lgd.lg_edge_index_map2 = torch.tensor([], dtype=data.edge_index.dtype) # lg_edge == g_node
            lgd.lg_x = torch.zeros(lgd.num_nodes, 2, dtype=lgd.x.dtype)
            lgd.lg_edge_index = lgd.edge_index
            lgd.lg_num_nodes = lgd.num_nodes
            return lgd

        N = data.num_nodes
        edge_index, edge_attr = data.edge_index, data.edge_attr
        (row, col), edge_attr = coalesce(edge_index, edge_attr, N, N)

        # Compute node indices.
        mask = row < col
        row, col = row[mask], col[mask]
        i = torch.arange(row.size(0), dtype=torch.long, device=row.device)
        # store the map of edge_node in line graph, node of line graph == edge of ori graph
        lgd.lg_edge_index_map = torch.stack([row, col], dim=0)

        (row, col), i = coalesce(
            torch.stack([
                torch.cat([row, col], dim=0),
                torch.cat([col, row], dim=0)
            ], dim=0), torch.cat([i, i], dim=0), N, N)

        # Compute new edge indices according to `i`.
        count = scatter_add(torch.ones_like(row), row, dim=0,
                            dim_size=data.num_nodes)
        joints = torch.split(i, count.tolist())

        def generate_grid(x):
            row = x.view(-1, 1).repeat(1, x.numel()).view(-1)
            col = x.repeat(x.numel())
            return torch.stack([row, col], dim=0)

        joints = [generate_grid(joint) for joint in joints]
        joints = torch.cat(joints, dim=1)
        joints, _ = remove_self_loops(joints)
        N = row.size(0) // 2
        joints, _ = coalesce(joints, None, N, N)

        if edge_attr is not None:
            lgd.lg_x = scatter_add(edge_attr, i, dim=0, dim_size=N) / 2
            lgd.lg_x = lgd.lg_x.to(dtype=lgd.x.dtype)
        lgd.lg_edge_index = joints
        lgd.lg_num_nodes = edge_index.size(1) // 2
        # edge of line graph == node of ori graph
        # (find the atom shared by the two bonds connected by each line-graph edge)
        edge1 = lgd.lg_edge_index_map[:,joints[0]]
        edge2 = lgd.lg_edge_index_map[:,joints[1]]
        map2 = torch.zeros(lgd.lg_edge_index.size(1), dtype=lgd.edge_index.dtype)
        for i in range(2):
            for j in range(2):
                map2 += edge1[i] * (edge1[i] == edge2[j])
        lgd.lg_edge_index_map2 = map2
        return lgd

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

def smiles_to_embedding(encoder, smiles):
    # SMILES -> molecular graph + line graph -> atom-level embeddings (num_atoms, emb_dim)
    device = next(encoder.parameters()).device
    data = smiles_to_graph(smiles)
    lg_data = LineGraph()(data)
    embedding = encoder(lg_data.to(device))
    return embedding

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate LEMON compound embeddings and ESM-2 protein embeddings for RP-PROTAC.")
    parser.add_argument("--data", type=str, default="./data/curated_data.csv",
                        help="CSV holding the sequences and SMILES to encode. Must contain the columns "
                             "protac, poi_ligand, linker, e3_ligand, poi, poi_sequence, e3 and e3_sequence "
                             "(default: %(default)s)")
    parser.add_argument("--compound_output", type=str, default="./data/lemon_compound_embedding.pt",
                        help="where to save the compound embeddings, as a {SMILES: Tensor} dict (default: %(default)s)")
    parser.add_argument("--protein_output", type=str, default="./data/esm2_protein_embedding.pt",
                        help="where to save the protein embeddings, as a {protein name: Tensor} dict (default: %(default)s)")
    parser.add_argument("--lemon_weights", type=str, default="./weights/pretrained_modules/lemon_100_a1.0_b1.0.pt",
                        help="checkpoint of the pretrained LEMON compound encoder. The architecture below "
                             "(5 layers, 300 dims) is fixed to match it (default: %(default)s)")
    parser.add_argument("--esm2_model", type=str, default="facebook/esm2_t33_650M_UR50D",
                        help="ESM-2 model to encode proteins with, downloaded from Hugging Face on first use. "
                             "A different model changes the embedding size, so --protein_dim of the train and "
                             "test scripts has to be changed to match (default: %(default)s)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="device to run both encoders on (default: %(default)s)")
    return parser.parse_args()

def build_compound_embeddings(data, lemon_weights, device):
    """
    Encode every unique SMILES in the poi_ligand, linker and e3_ligand columns
    (and protac, when present) with the pretrained LEMON GNN.
    :return: {SMILES: Tensor(num_atoms, 300)}
    """
    # num_layer and emb_dim are fixed: they must match the pretrained checkpoint
    lemon = GNN(num_layer=5, emb_dim=300).to(device)
    lemon.load_state_dict(torch.load(lemon_weights, map_location=device))
    lemon.eval()

    columns = [c for c in ["protac", "poi_ligand", "linker", "e3_ligand"] if c in data.columns]
    smiles_list = np.unique(np.concatenate([data[c].unique() for c in columns]))

    compound_embedding = {}
    with torch.no_grad():
        for smiles in tqdm(smiles_list, desc="Processing compound embeddings"):
            compound_embedding[smiles] = smiles_to_embedding(lemon, smiles).detach().cpu()
    return compound_embedding

def build_protein_embeddings(data, esm2_model, device):
    """
    Encode every unique POI and E3 ligase sequence with ESM-2.
    :return: {protein name: Tensor(num_tokens, 1280)}
    """
    esm2_tokenizer = AutoTokenizer.from_pretrained(esm2_model)
    esm2 = AutoModel.from_pretrained(esm2_model).to(device)
    esm2.eval()

    poi_list = data[["poi", "poi_sequence"]].drop_duplicates().reset_index(drop=True).rename(columns={"poi": "name", "poi_sequence": "sequence"})
    e3_list = data[["e3", "e3_sequence"]].drop_duplicates().reset_index(drop=True).rename(columns={"e3": "name", "e3_sequence": "sequence"})

    protein_embedding = {}
    with torch.no_grad():
        for name, sequence in tqdm((pd.concat([poi_list, e3_list], ignore_index=True).drop_duplicates()).itertuples(index=False, name=None),
                                   desc="Processing protein embeddings"):
            # 1022 is the longest sequence ESM-2 accepts once <cls> and <eos> are added
            inputs = esm2_tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1022)
            output = esm2(**inputs.to(device))
            protein_embedding[name] = output.last_hidden_state.cpu().squeeze()
    return protein_embedding

def save_embedding(embedding, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(embedding, path)

def main(args):
    data = pd.read_csv(args.data, keep_default_na=False)
    save_embedding(build_compound_embeddings(data, args.lemon_weights, args.device), args.compound_output)
    save_embedding(build_protein_embeddings(data, args.esm2_model, args.device), args.protein_output)

if __name__ == "__main__":
    main(parse_args())
