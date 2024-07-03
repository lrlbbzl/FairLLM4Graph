import torch
import numpy as np
import random

x = torch.load('results.pt')
scores = x['scores'].cpu()
labels = x['labels']
is_heter = x['is_heter']
node_idx = x['node_idx']

def merge(a, b):
    return list(set(a.numpy()) & set(b.numpy()))

heter_idx, homo_idx = torch.nonzero(is_heter == 1).squeeze(-1), torch.nonzero(is_heter == 0).squeeze(-1)
true_l, false_l = torch.nonzero(labels == 1).squeeze(-1), torch.nonzero(labels == 0).squeeze(-1)

heter_scores, homo_scores = scores[heter_idx], scores[homo_idx]
heter_true_scores = scores[merge(true_l, heter_idx)]
heter_false_scores = scores[merge(false_l, heter_idx)]
homo_true_scores = scores[merge(true_l, homo_idx)]
homo_false_scores = scores[merge(false_l, homo_idx)]

print(len(heter_true_scores), torch.mean(heter_true_scores))
print(len(homo_true_scores), torch.mean(homo_true_scores))
print(len(heter_false_scores), torch.mean(heter_false_scores))
print(len(homo_false_scores), torch.mean(homo_false_scores))

gap = torch.abs(labels - scores)
prob = (gap - torch.min(gap)) / (torch.max(gap) - torch.min(gap))
beta = 0.3
prob = prob + (1 - prob) * beta
idx = []
for i, p in enumerate(prob):
    t = random.random()
    if t <= p:
        idx.append(i)
new_heter, new_homo = [i for i in idx if is_heter[i] == 1], [i for i in idx if is_heter[i] == 0]
new_true_heter, new_false_heter = [i for i in new_heter if labels[i] == 1], [i for i in new_heter if labels[i] == 0]
new_true_homo, new_false_homo = [i for i in new_homo if labels[i] == 1], [i for i in new_homo if labels[i] == 0]

print(len(new_true_heter))
print(len(new_false_heter))
print(len(new_true_homo))
print(len(new_false_homo))

## prepare data
pos_idx, neg_idx = [i for i in idx if labels[i] == 1], [i for i in idx if labels[i] == 0]

mp = {
    'pos_edge' : node_idx[pos_idx].transpose(0, 1),
    'neg_edge' : node_idx[neg_idx].transpose(0, 1)
}

torch.save(mp, 'filter_data.pt')