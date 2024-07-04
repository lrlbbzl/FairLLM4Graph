export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
wandb offline
python main.py --conv-name gcn \
    --mode ft_lm \
    --dataset pubmed \
    --epoch 500 \
    --use-peft \
    --lm-epochs 5 \
    --lm-batch-size 48 \
    --sm-batch-size 48 \
    --logging-steps 10 \
    --eval-steps 500 \
    --ft-lr 1e-3 \
    # --filter
    
