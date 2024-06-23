import gc
import logging
import os
import os.path as osp

import evaluate
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import subgraph
from torchmetrics.functional import retrieval_reciprocal_rank as mrr
from transformers import EarlyStoppingCallback
from transformers import Trainer as HugTrainer
from transformers import TrainingArguments


class InnerTrainer(HugTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        input_ids, attention_mask = inputs.pop("input_ids"), inputs.pop("attention_mask")
        pred = model(input_ids, attention_mask)
        pred = pred.squeeze(-1)
        loss = F.binary_cross_entropy(pred, labels)
        if return_outputs:
            pred = {'pred' : pred}
        return (loss, pred) if return_outputs else loss