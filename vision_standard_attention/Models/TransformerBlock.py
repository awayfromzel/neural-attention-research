from torch import nn

from Models.MHSelfAttention import MHSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, pos_embed=None, dim_linear_block=1024, dropout=0.1):
        super().__init__()
        self.mhsa = MHSelfAttention(dim=dim, heads=heads, dim_head=dim_head)
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
        
    def forward(self, x):
        y = self.norm_1(self.drop(self.mhsa(x)) + x)
        return self.norm_2(self.linear(y) + y)
