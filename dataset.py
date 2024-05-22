import json
import torch
import os.path as osp

def get_dataset(args):
    if args.dataset == 'cora':
        p = osp.join('./dataset', args.dataset)
        g, text = torch.load(osp.join(p, 'g.pt')), json.load(open(osp.join(p, 'text.json'), 'r'))
        return g, text