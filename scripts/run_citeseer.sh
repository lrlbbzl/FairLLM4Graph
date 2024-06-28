export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
wandb offline
python main.py --conv-name gcn \
    --mode ft_lm \
    --dataset citeseer \
    --eval-steps 500 \
    --epoch 500 \