import torch
from torch import nn
import torch.nn.functional as F

class TrainValidateWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.model = net
        self.max_seq_len = net.max_seq_len
        
    @torch.no_grad()    
    def validate(self, x, y):
        self.model.eval()
        out = self.model(x) # output based on cls token
        logits_reorg = out[:,0,:]
        actual_recog = logits_reorg.argmax(dim=1)
        count_correct = (actual_recog == y).sum().item()
        return count_correct
    
    def forward(self, x, y):
        out = self.model(x)
        logits_reorg = out[:,0,:] # output based on cls token 
        loss = F.cross_entropy(logits_reorg,y)
        return loss
