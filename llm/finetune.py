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

from config import logger

def finetune_lm(args, data, text):
    neg_edges_train = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1) // 2
        )
    
    ## Encode text
    t = [b for a, b in text.items()]
    if args.plm_name == 'bert-base-uncased':
        tokenizer = AutoTokenizer.from_pretrained(args.plm_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    logger.info('Use model : {}'.format(args.plm_name))
    inputs = tokenizer(t,
                    truncation=True, 
                    add_special_tokens=True,
                    max_length=256,
                    padding='max_length',
                    return_tensors='pt',)
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
        per_device_train_batch_size=args.sm_batch_size,
        gradient_accumulation_steps=args.lm_batch_size // args.sm_batch_size,
        output_dir=args.model_path,
        learning_rate=args.ft_lr,
        per_device_eval_batch_size=args.lm_batch_size,
        num_train_epochs=args.lm_epochs,
        weight_decay=0.01,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
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
    lm = model.model
    lm.save_pretrained(osp.join(args.model_path, 'save_model'))


def merge_modeling(args, g, text):
    if args.plm_name == 'bert-base-uncased':
        lm = AutoModel.from_pretrained(args.plm_path)
        tokenizer = AutoTokenizer.from_pretrained(args.plm_path)
    else:
        lm = AutoModel.from_pretrained(args.plm_path, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(args.plm_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    peft_model = PeftModel.from_pretrained(lm, osp.join(args.model_path, 'save_model'))
    if args.use_peft:
        model = peft_model.model
    else:
        model = lm
    t = [b for a, b in text.items()]
    inputs = tokenizer(t,
                    truncation=True, 
                    add_special_tokens=True,
                    max_length=256,
                    padding='max_length',
                    return_tensors='pt')
    encode_data = EncodeDataset(inputs['input_ids'], inputs['attention_mask'])
    data_loader = DataLoader(encode_data, batch_size=32, shuffle=False, num_workers=4)
    if args.pooling == 'mean':
        pooler = MeanPooling()
    elif args.pooling == 'max':
        pooler = MaxPooling()
    res = []
    logger.info("Get Embedding. Total time: {}".format(len(encode_data) / 32))
    for step, data in tqdm(enumerate(data_loader)):
        input_ids, attention_mask = data['input_ids'].to(lm.device), data['attention_mask'].to(lm.device)
        outputs = model(input_ids, attention_mask)
        outputs = pooler(outputs.last_hidden_state, attention_mask)
        res.append(outputs)
    return torch.cat(res, dim=0)
