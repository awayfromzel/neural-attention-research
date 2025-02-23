from torch import nn
from Models.TransformerBlock import TransformerBlock
from PositionalEncoding import PositionalEncoding
import torch
from PatchEmbedding import PatchEmbedding_CNN

class SimpleTransformer(nn.Module):
    def __init__(self, dim, num_unique_tokens=10, num_layers=6, heads=8, dim_head=None, max_seq_len=1024, use_neural_attention=True):
        super().__init__()
        self.max_seq_len = max_seq_len
 
        self.token_emb = PatchEmbedding_CNN(emb_size=dim)  # Vision token embedding
        
        self.pos_enc = PositionalEncoding(dim, max_seq_length=max_seq_len)
 
        # Create the layers, use neural attention only in the first layer
        self.block_list = [TransformerBlock(dim=dim, heads=heads, dim_head=dim_head, use_neural_attention=(i == 0)) for i in range(num_layers)]
        self.layers = nn.ModuleList(self.block_list)
        
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_unique_tokens)
        )
        
    def forward(self, x, mask=None):
        pos = torch.arange(0, x.shape[-1], dtype=torch.long, device=x.device).unsqueeze(0)  # shape (1, t)
        x = self.token_emb(x)
        x = x + self.pos_enc(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.to_logits(x)