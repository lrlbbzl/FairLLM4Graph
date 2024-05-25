import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

NUM_NEG_PER_SAMPLE = 100

class LPDataset(Dataset):
    def __init__(self, input_ids, attention_mask, pos_edge, neg_edge):
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.pos_edge = pos_edge
        self.neg_edge = neg_edge
        self.all_edge = torch.cat([self.pos_edge, self.neg_edge], dim=-1)
        self.labels = torch.FloatTensor([0] * self.pos_edge.size(1) + [1] * self.neg_edge.size(1))

    def __len__(self):
        return self.all_edge.size(1)
    
    def __getitem__(self, idx):
        src, dst, l = self.all_edge[0][idx], self.all_edge[1][idx], self.labels[idx]
        node_idx = torch.LongTensor([src, dst])
        return {
            'input_ids' : self.input_ids[node_idx],
            'attention_mask' : self.attention_mask[node_idx], 
            'label' : l
        }

class EncodeDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self, ):
        return self.input_ids.size(0)
    
    def __getitem__(self, index):
        return {
            'input_ids' : self.input_ids[index],
            'attention_mask' : self.attention_mask[index]
        }