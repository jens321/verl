BETA=0.01
for TASK in math gsm8k dapo-with-aime24; do
    if [ ${TASK} == "dapo-with-aime24" ]; then
        SPARSE_DIM=128
    else
        SPARSE_DIM=32
    fi
    
    for SEED in 41 42 43; do
        sbatch --job-name=${TASK}_elliptical_seed_${SEED} train_elliptical.slurm $TASK $SPARSE_DIM $BETA $SEED
    done
done