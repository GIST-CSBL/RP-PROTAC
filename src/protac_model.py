import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention where `query` attends to `keyvalue`.
    Zero-padded positions (std == 0) are masked; a residual connection is added to the projected query.
    """
    def __init__(self, emb_dim, num_heads=1):
        super().__init__()
        self.query_proj = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=False)
        self.key_proj = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=False)
        self.value_proj = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=False)
        self.out_proj = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.scale = nn.Parameter(torch.tensor(1/2))
        
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        
    def attention(self, query, key, value, scale=None, query_mask=None, key_mask=None):
        if key_mask == None:
            query_mask = query.std(dim=-1)==0
            key_mask = key.std(dim=-1)==0
        if scale == None:
            scale = self.scale
        Q = query.view(-1, query.shape[-2], self.num_heads, self.emb_dim//self.num_heads).transpose(-2, -3) # (B, Lq, D) -> (B, Lq, H, D//H) -> (B, H, Lq, D//H)
        K = key.view(-1, key.shape[-2], self.num_heads, self.emb_dim//self.num_heads).transpose(-2, -3) # (B, Lk, D) -> (B, Lk, H, D//H) -> (B, H, Lk, D//H)
        V = value.view(-1, value.shape[-2], self.num_heads, self.emb_dim//self.num_heads).transpose(-2, -3) # (B, Lv, D) -> (B, Lv, H, D//H) -> (B, H, Lv, D//H)
        weight_matrix = (Q @ K.transpose(-2, -1)).masked_fill(key_mask.view(-1, 1, 1, key_mask.shape[-1]), -1e9) # (B, H, Lq, Lk)
        weight_matrix = F.softmax(weight_matrix * scale, dim=-1).masked_fill(query_mask.view(-1, 1, query_mask.shape[-1], 1), 0) # (B, H, Lq, Lk)
        output = weight_matrix @ V # (B, H, Lq, D//H)
        output = output.transpose(-2, -3).contiguous().view(-1, query.shape[-2], self.emb_dim) # (B, H, Lq, D//H) -> (B, Lq, H, D//H) -> (B, Lq, D)
        output = self.out_proj(output)
        return output, weight_matrix
        
    def forward(self, query, keyvalue, query_mask=None, keyvalue_mask=None):
        query = self.query_proj(query)
        key = self.key_proj(keyvalue)
        value = self.value_proj(keyvalue)
        output, weight_matrix = self.attention(query, key, value, self.scale, query_mask, keyvalue_mask)
        output = output + query # residual connection
        return output, weight_matrix    

class Predictor(nn.Module):    
    """
    RP-PROTAC predictor (one member of the deep ensemble).
    Returns the degradation logit, its standard deviation (aleatoric uncertainty)
    and the cross-attention weights.
    """
    def __init__(self, protein_dim=1280, compound_dim=300, emb_dim=512, num_heads=2, dropout_rate=0.1, pool="mean"):
        super().__init__()
        assert emb_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.poi_proj = nn.Sequential(nn.LayerNorm(normalized_shape=protein_dim),
                                      nn.Linear(in_features=protein_dim, out_features=emb_dim),
                                      nn.LeakyReLU())
        self.poi_ligand_proj = nn.Sequential(nn.LayerNorm(normalized_shape=compound_dim),
                                             nn.Linear(in_features=compound_dim, out_features=emb_dim),
                                             nn.LeakyReLU())
        self.linker_proj = nn.Sequential(nn.LayerNorm(normalized_shape=compound_dim),
                                         nn.Linear(in_features=compound_dim, out_features=emb_dim),
                                         nn.LeakyReLU())
        self.e3_ligand_proj = nn.Sequential(nn.LayerNorm(normalized_shape=compound_dim),
                                            nn.Linear(in_features=compound_dim, out_features=emb_dim),
                                            nn.LeakyReLU())
        self.e3_proj = nn.Sequential(nn.LayerNorm(normalized_shape=protein_dim),
                                     nn.Linear(in_features=protein_dim, out_features=emb_dim),
                                     nn.LeakyReLU())

        self.poi_poi_ligand_attention = MultiHeadCrossAttention(emb_dim, num_heads)
        self.e3_e3_ligand_attention = MultiHeadCrossAttention(emb_dim, num_heads)
        self.poi_linker_attention = MultiHeadCrossAttention(emb_dim, num_heads)
        self.e3_linker_attention = MultiHeadCrossAttention(emb_dim, num_heads)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.pool = pool

        self.header = nn.Sequential(nn.Linear(emb_dim*2, emb_dim//4),
                                    nn.LayerNorm(normalized_shape=emb_dim//4),
                                    nn.LeakyReLU(),
                                    nn.Dropout(p=dropout_rate))
        self.prob_header = nn.Linear(emb_dim//4, 1)
        self.std_header = nn.Linear(emb_dim//4, 1)
        
    def forward(self, poi, poi_ligand, linker, e3_ligand, e3):
        # linear projection to align dimensions of all input features
        poi_out = self.poi_proj(poi)
        poi_ligand_out = self.poi_ligand_proj(poi_ligand)
        linker_out = self.linker_proj(linker)
        e3_ligand_out = self.e3_ligand_proj(e3_ligand)
        e3_out = self.e3_proj(e3)

        # extract the padded region
        poi_mask = poi.std(dim=-1)==0
        poi_ligand_mask = poi_ligand.std(dim=-1)==0
        linker_mask = linker.std(dim=-1)==0
        e3_ligand_mask = e3_ligand.std(dim=-1)==0
        e3_mask = e3.std(dim=-1)==0
        
        # first cross-attention: ligand attends to its target protein
        poi_poi_ligand_out, poi_poi_ligand_weight = self.poi_poi_ligand_attention(poi_ligand_out, poi_out, poi_ligand_mask, poi_mask)
        e3_e3_ligand_out, e3_e3_ligand_weight = self.e3_e3_ligand_attention(e3_ligand_out, e3_out, e3_ligand_mask, e3_mask)
        
        # second cross-attention: linker attends to each protein-ligand complex
        poi_linker_out, poi_linker_weight = self.poi_linker_attention(linker_out, poi_poi_ligand_out, linker_mask, poi_ligand_mask)
        e3_linker_out, e3_linker_weight = self.e3_linker_attention(linker_out, e3_e3_ligand_out, linker_mask, e3_ligand_mask)

        # max pooling
        if self.pool == "max":
            poi_linker_out = torch.max(poi_linker_out, dim=-2).values
            e3_linker_out = torch.max(e3_linker_out, dim=-2).values
            
        # mean pooling
        elif self.pool == "mean":
            poi_linker_out = poi_linker_out.masked_fill(linker_mask.view(-1, linker_mask.shape[-1], 1), 0).sum(dim=-2) / (linker_mask.shape[1] - linker_mask.sum(dim=-1)).unsqueeze(dim=-1)
            e3_linker_out = e3_linker_out.masked_fill(linker_mask.view(-1, linker_mask.shape[-1], 1), 0).sum(dim=-2) / (linker_mask.shape[1] - linker_mask.sum(dim=-1)).unsqueeze(dim=-1)

        elif self.pool == "sum":
            poi_linker_out = torch.sum(poi_linker_out, dim=-2)
            e3_linker_out = torch.sum(e3_linker_out, dim=-2)
            
        # MLP header: logit (prob) and log-std -> std
        out = torch.cat([poi_linker_out, e3_linker_out], dim=-1)
        out = self.header(out)
        prob = self.prob_header(out)
        std = torch.exp(self.std_header(out))
        return prob, std, (poi_poi_ligand_weight, e3_e3_ligand_weight, poi_linker_weight, e3_linker_weight)