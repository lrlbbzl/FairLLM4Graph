export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
wandb offline
python main.py --conv-name gcn \
    --mode po \
    --dataset cora \
    --po-epoch 500 \
    --po-sm-batch-size 8 \
    --po-batch-size 8 \
    --logging-steps 10 \
    --eval-steps 200 \
    --po-lr 1e-4 \
    --use-peft
    # --filter \
    # --add-kl \
    # --use-peft
    
