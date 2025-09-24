NAME=countdown-4_elliptical_seed_44_kl_0.0_ppo_epochs_1_beta_0.01_turn_off_elliptical_if_none_correct_True_sparse_dim_128
CHECKPOINT_PATH=/scratch/gpfs/jtuyls/llm-rl-exploration/checkpoints/llm-exploration-rl-training/${NAME}/global_step_200/actor

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $CHECKPOINT_PATH \
    --target_dir $CHECKPOINT_PATH/hf