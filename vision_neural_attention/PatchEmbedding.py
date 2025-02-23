from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange
import torch

class PatchEmbedding_Linear(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 512):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
 
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x
    
class PatchEmbedding_CNN(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 8, emb_size: int = 512):
        self.patch_size = patch_size
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size).cuda())
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, 
            stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
 
    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0] # batch_size
        cls_tokens = self.cls_token.expand(B, -1, -1) # cls token
        x = self.projection(x)
        x = torch.cat((cls_tokens, x), dim=1)
        return x
