import torch
from torch.utils.data import Dataset

class MyNLPDataSet(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
       
    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,)) #the (1,) at the end here is to define the dimensions of the output tensor
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long() #the full_seq tensor contains the indicies of the embeddings in the sequence in our dictionary, not the embeddings themselves
        return full_seq.cuda()
    
    def __len__(self):
        return self.data.size(0) #self.seq_len