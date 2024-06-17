import argparse
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Fairness TAG')
parser.add_argument('--input-dir', type=str, default='./data/')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--in-dim', type=int, default=200)
parser.add_argument('--out-dim', type=int, default=512)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--hidden-size', type=int, default=200)
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--n-layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--conv-norm', type=bool, default=True)
parser.add_argument('--conv-name', type=str, default='gcn')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--mode', type=str, default='ft_lm', choices=['gnn', 'ft_lm'])
parser.add_argument('--output-dir', type=str, default='./checkpoints/bert')

# lm config
parser.add_argument('--plm-path', type=str, default='/apdcephfs_cq8/share_300043402/rileyrlluo/models/models--google-bert--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594')
parser.add_argument('--plm-finetune', action='store_true')
parser.add_argument('--pooling', type=str, default='mean', choices=['max', 'mean'])
parser.add_argument('--lm-batch-size', type=int, default=4)
parser.add_argument('--lm-epochs', type=int, default=2)
parser.add_argument('--sm-batch-size', type=int, default=4)
parser.add_argument('--ft-lr', type=float, default=1e-4)
# peft config
parser.add_argument('--peft-r', type=int, default=8)
parser.add_argument('--peft-lora-alpha', type=int, default=8)
parser.add_argument('--peft-lora-dropout', type=float, default=0.2)
parser.add_argument('--use-peft', type=bool, default=True)

args = parser.parse_args()
args.lm_name = args.output_dir[args.output_dir.rfind('/') + 1:]
 