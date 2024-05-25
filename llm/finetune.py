import gc
import logging
import os
import os.path as osp
import json

import evaluate
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import subgraph, negative_sampling
from torchmetrics.functional import retrieval_reciprocal_rank as mrr
from transformers import EarlyStoppingCallback
from transformers import Trainer as HugTrainer
from transformers import TrainingArguments
from transformers import AutoModel, AutoTokenizer
from .dataset import LPDataset, EncodeDataset
from .lm_modeling import LP_model, lp_compute_metrics
from transformers import Trainer
from .lm_trainer import InnerTrainer
from peft import PeftConfig, PeftModel
from torch.utils.data import Dataset
from .pooling import MeanPooling, MaxPooling
from tqdm import tqdm

def finetune_lm(args, data, text):
    neg_edges_train = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1) // 2
        )
    ## Encode text
    t = [b for a, b in text.items()]
    tokenizer = AutoTokenizer.from_pretrained(args.plm_path)
    inputs = tokenizer(t,
                       truncation=True, 
                       add_special_tokens=True,
                       max_length=256,
                       padding='max_length',
                       return_tensors='pt')
    train_dataset = LPDataset(inputs['input_ids'],
                              inputs['attention_mask'],
                              data.train_pos_edge_index,
                              neg_edges_train)
    valid_dataset = LPDataset(inputs['input_ids'],
                              inputs['attention_mask'],
                              data.val_pos_edge_index,
                              data.val_neg_edge_index)
    test_dataset = LPDataset(inputs['input_ids'],
                              inputs['attention_mask'],
                              data.test_pos_edge_index,
                              data.test_neg_edge_index)
    train_dataloder = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
    model = LP_model(args)
    # for da in train_dataloder:
    #     output = model(input_ids=da['input_ids'], attention_mask=da['attention_mask'])
    #     import pdb; pdb.set_trace()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.ft_lr,
        per_device_train_batch_size=args.lm_batch_size,
        per_device_eval_batch_size=args.lm_batch_size,
        num_train_epochs=args.lm_epochs,
        weight_decay=0.01,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        logging_dir='./train.log',
        logging_steps=50,
    )
    trainer = InnerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=lp_compute_metrics
    )
    trainer.train()
    import pdb; pdb.set_trace()
    lm = model.model
    lm.save_pretrained(osp(args.output_dir, 'save_model'))




def merge_modeling(args, g, text):
    lm = AutoModel.from_pretrained(args.plm_path).cuda()
    peft_model = PeftModel.from_pretrained(lm, osp.join(args.output_dir))
    if args.use_peft:
        model = peft_model.model
    else:
        model = lm
    t = [b for a, b in text.items()]
    tokenizer = AutoTokenizer.from_pretrained(args.plm_path)
    inputs = tokenizer(t,
                    truncation=True, 
                    add_special_tokens=True,
                    max_length=512,
                    padding='max_length',
                    return_tensors='pt')
    encode_data = EncodeDataset(inputs['input_ids'], inputs['attention_mask'])
    data_loader = DataLoader(encode_data, batch_size=64, shuffle=False, num_workers=4)
    if args.pooling == 'mean':
        pooler = MeanPooling()
    elif args.pooling == 'max':
        pooler = MaxPooling()
    res = []
    for step, data in tqdm(enumerate(data_loader)):
        input_ids, attention_mask = data['input_ids'].to(lm.device), data['attention_mask'].to(lm.device)
        outputs = model(input_ids, attention_mask)
        outputs = pooler(outputs.last_hidden_state, attention_mask)
        res.append(outputs)
    return torch.cat(res, dim=0)