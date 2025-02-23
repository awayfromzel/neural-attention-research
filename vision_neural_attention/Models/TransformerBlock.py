from torch import nn
from Models.MHSelfAttention import MHSelfAttention
from Models.NeuralAttention import NeuralAttention  # Import your neural attention class

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, causal=False, pos_embed=None, dim_linear_block=1024, dropout=0.1, use_neural_attention=False):
        super().__init__()
        # Use neural attention for the first block, otherwise use standard attention
        if use_neural_attention:
            self.attention = NeuralAttention(dim=dim, heads=heads, dim_head=dim_head, causal=causal)
        else:
            self.attention = MHSelfAttention(dim=dim, heads=heads, dim_head=dim_head)
        
        self.drop = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        
        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

    def set_causal(self, causal):
        self.attention.set_causal(causal)
        
    def forward(self, x, mask=None):  # For vision, mask can default to None
        y = self.norm_1(self.drop(self.attention(x, mask)) + x)
        return self.norm_2(self.linear(y) + y)