import torch
import numpy as np
from torch import nn
from einops import rearrange

class NeuralAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, causal=True, projection_dim=None):
        super().__init__()
        self.dim_head = (int(dim/heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads 
        self.causal = causal 
        self.to_qkv = nn.Linear(dim, _dim * 3, bias=False) 
        self.W_out = nn.Linear(_dim, dim, bias=False) 
        self.scale_factor = self.dim_head ** -0.5 

        # Projection dimension for scaling down q and k
        self.projection_dim = projection_dim if projection_dim is not None else self.dim_head // 32
        self.q_proj = nn.Linear(self.dim_head, self.projection_dim)
        self.k_proj = nn.Linear(self.dim_head, self.projection_dim)
        
        # Define the sequence of linear layers for custom attention
        self.linear1 = nn.Linear(self.projection_dim * 2, self.projection_dim)
        self.linear2 = nn.Linear(self.projection_dim, self.projection_dim // 2)
        self.linear3 = nn.Linear(self.projection_dim // 2, 1)
    
    def set_causal(self, causal):
        self.causal = causal
        
    def forward(self, x, mask=None):
        assert x.dim() == 3
        qkv = self.to_qkv(x)  # [b, n, dim*3]
        q, k, v = tuple(rearrange(qkv, 'b n (d k h) -> k b h n d', k=3, h=self.heads))  # [3, b, heads, seq_length, dim_head]

        # Apply linear projection to scale down q and k
        q = self.q_proj(q)  # [b, heads, seq_length, projection_dim]
        k = self.k_proj(k)  # [b, heads, seq_length, projection_dim]

        # Compute attention scores using vectorized operations
        q = q.unsqueeze(3).repeat(1, 1, 1, q.shape[2], 1)  # [b, heads, seq_length, seq_length, projection_dim]
        k = k.unsqueeze(2).repeat(1, 1, k.shape[2], 1, 1)  # [b, heads, seq_length, seq_length, projection_dim]

        qk_cat = torch.cat((q, k), dim=-1)  # [b, heads, seq_length, seq_length, projection_dim * 2]
        score = self.linear1(qk_cat)
        score = torch.relu(score)
        score = self.linear2(score)
        score = torch.relu(score)
        score = self.linear3(score).squeeze(-1)  # [b, heads, seq_length, seq_length]

        # Apply mask if necessary
        if self.causal:
            mask = torch.ones_like(score).triu_(1).bool()
            score = score.masked_fill(mask, float('-inf'))
        
        if mask is not None:
            score = score.masked_fill(mask, float('-inf'))
        
        # Compute attention weights
        attention = torch.softmax(score, dim=-1)  # [b, heads, seq_length, seq_length]
        
        # Compute the output for each head
        out = torch.einsum('b h i j, b h j d -> b h i d', attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # Combine heads back together
        
        return self.W_out(out)  # Final linear transformation