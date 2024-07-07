export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
wandb offline
python main.py --conv-name gcn \
    --mode po \
    --dataset cora \
    --po-epoch 0 \
    --po-sm-batch-size 16 \
    --po-batch-size 16 \
    --logging-steps 10 \
    --eval-steps 200 \
    --po-lr 1e-5 \
    --use-peft
    # --filter \
    # --add-kl \
    
