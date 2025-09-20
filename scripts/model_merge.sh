NAME=dapo-with-aime2425_grpo_seed_41_kl_0.0_ppo_epochs_1
CHECKPOINT_PATH=/scratch/gpfs/jtuyls/llm-rl-exploration/checkpoints/llm-exploration-rl-training/${NAME}/global_step_230/actor

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $CHECKPOINT_PATH \
    --target_dir $CHECKPOINT_PATH/hf