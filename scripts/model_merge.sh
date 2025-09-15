NAME=math_unlikely_seed_43_kl_0.0
CHECKPOINT_PATH=/scratch/gpfs/KARTHIKN/jtuyls/checkpoints/llm-exploration-rl-training/${NAME}/best_pass@1/actor

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $CHECKPOINT_PATH \
    --target_dir $CHECKPOINT_PATH/hf