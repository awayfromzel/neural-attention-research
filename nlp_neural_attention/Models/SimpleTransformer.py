import torch
from torch import nn
from TransformerBlock import TransformerBlock
from PositionalEncoding import PositionalEncoding

class SimpleTransformer(nn.Module):
    def __init__(self, dim, num_layers, num_unique_tokens, heads, max_seq_len, dim_head=None, dim_linear_block=1024, dropout=0.1, causal=True):
        super().__init__()
        self.max_seq_len = max_seq_len  # Add this line to store max_seq_len as an attribute
        self.causal=causal
        self.token_emb = nn.Embedding(num_unique_tokens, dim)
        self.pos_enc = PositionalEncoding(dim, max_seq_length=max_seq_len)
        self.layers = nn.ModuleList([])

        for i in range(num_layers):
            self.layers.append(TransformerBlock(
                dim=dim, 
                heads=heads, 
                dim_head=dim_head,
                causal=causal,
                dim_linear_block=dim_linear_block, 
                dropout=dropout, 
                use_neural_attention=(i == 0)  # Use neural attention only in the first layer              
            )
        )
        
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_unique_tokens)
        )

    def forward(self, x, mask=None):
        b, n = x.shape
        x = self.token_emb(x)
        x += x + self.pos_enc(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.to_logits(x)