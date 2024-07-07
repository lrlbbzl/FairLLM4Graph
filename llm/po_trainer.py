import gc
import logging
import os
import os.path as osp

import evaluate
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import KLDivLoss
import torch.nn.functional as F
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import subgraph
from torchmetrics.functional import retrieval_reciprocal_rank as mrr
from transformers import EarlyStoppingCallback
from transformers import Trainer as HugTrainer
from transformers import TrainingArguments
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from .lm_modeling import load_model

from config import args

if args.mode == 'po':
    ref_model = load_model(args, type='ref')

class PoTrainer(HugTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensitive_weight = 2.0
        self.non_sensitive_weight = 1.0
        self.gamma = 2.0
        self.compute_steps = 0.0

    def compute_loss(self, model, inputs, return_outputs=False):

        heter_input_ids, heter_attention_mask = inputs.pop('heter_input_ids'), inputs.pop('heter_attention_mask')
        homo_input_ids, homo_attention_mask = inputs.pop('homo_input_ids'), inputs.pop('homo_attention_mask')

        win_pred = model(heter_input_ids, heter_attention_mask)
        lose_pred = model(homo_input_ids, homo_attention_mask)

        with torch.no_grad():
            win_ref = ref_model(heter_input_ids, heter_attention_mask)
            lose_ref = ref_model(homo_input_ids, homo_attention_mask)

        # import pdb; pdb.set_trace()
        theta_logratio = torch.log(win_pred) - torch.log(lose_pred)
        ref_logratio = torch.log(win_ref) - torch.log(lose_ref)

        beta = args.po_beta

        loss = - F.logsigmoid(beta * (theta_logratio - ref_logratio)).mean()
        if return_outputs:
            pred = {'pred' : win_pred.squeeze(-1) }
        return (loss, pred) if return_outputs else loss