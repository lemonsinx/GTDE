#!/bin/bash
env="StarCraft2v2"
map="10gen_protoss"
algo="GTDE"
units="20v20"

exp="protoss"
seed_max=5

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python ../train/train_smacv2.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --units ${units} --n_training_threads 16 --n_rollout_threads 16 --num_mini_batch 1 --episode_length 400 \
    --num_env_steps 10000000 --ppo_epoch 5 --use_value_active_masks --use_eval --eval_episodes 32 --use_wandb
done

