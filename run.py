import os
import os.path as osp
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch_geometric as tg
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit
import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from dataset import get_dataset
from torch_geometric.utils import train_test_split_edges
from gnn_model.model import GNN
from sklearn.metrics import classification_report
from torch.optim import Adam, AdamW
from utils import get_link_labels

from config import logger
from llm.finetune import finetune_lm, merge_modeling
from transformers import AutoModel, AutoTokenizer
from llm.lm_trainer import InnerTrainer
from llm.lm_modeling import LP_model

from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"

def run(args):
    ## prepare dataset
    g, text = get_dataset(args)
    args.in_dim = g.num_features
    protected_attribute = g.y
    num_calsses = len(np.unique(protected_attribute))
    g.to(device)
    data = train_test_split_edges(g, val_ratio=0.1, test_ratio=0.2)
    data = data.to(device)
    N = data.num_nodes

    if args.mode == 'ft_lm':
        ## Start fine-tune LLM for Graph context
        model_path = args.output_dir
        embeds_path = osp.join(args.output_dir, 'text_embeddings_{}.pt'.format(args.use_peft))
        if not any('checkpoint' in d for d in os.listdir(model_path)):
            finetune_lm(args, g, text)
        if not osp.exists(embeds_path):
            text_embeddings = merge_modeling(args, g, text)
            torch.save(text_embeddings, embeds_path)
        text_embeddings = torch.load(embeds_path)
        args.in_dim = text_embeddings.size(1)
        data.x = text_embeddings.detach()
    model = GNN(args.in_dim, args.out_dim, args.n_heads, args.n_layers, args.dropout, args.conv_name).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)


    best_val_perf = test_perf = 0
    for epoch in range(1, args.epoch + 1):
        neg_edges_train = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=N,
            num_neg_samples=data.train_pos_edge_index.size(1)
        ).to(device)

        model.train()
        optimizer.zero_grad()
        h = model(data.x, data.train_pos_edge_index)
        link_logits, _ = model.test(h, data.train_pos_edge_index, neg_edges_train)
        train_labels = get_link_labels(data.train_pos_edge_index, neg_edges_train).to(device)
        loss = F.binary_cross_entropy_with_logits(link_logits, train_labels)
        loss.backward()
        optimizer.step()

        # eval
        model.eval()
        res = []
        for cls in ['val', 'test']:
            pos_edge_index = data['{}_pos_edge_index'.format(cls)]
            neg_edge_index = data['{}_neg_edge_index'.format(cls)]
            with torch.no_grad():
                h = model(data.x, data.train_pos_edge_index)
                link_logits, edge_index = model.test(h, pos_edge_index, neg_edge_index)
            link_probs = link_logits.sigmoid()
            link_labels = get_link_labels(pos_edge_index, neg_edge_index)
            metric = roc_auc_score(link_labels.cpu(), link_probs.cpu())
            res.append(metric)

        val_res, tmp_test_res = res
        if val_res > best_val_perf:
            best_val_perf = val_res
            test_perf = tmp_test_res
        if epoch % 20 == 0:
            logger.info('Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(epoch, loss, best_val_perf, test_perf))

        

        




