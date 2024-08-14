env_name="gather_v3"
algo="iac"
exp="gather"
seed_max=1
for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python ../train.py --env_name ${env_name} --algo ${algo} --exp ${exp} \
  --value_coef 1. --ent_coef 0.01 --gamma 0.99 --learning_rate 1e-4 --n_round 1200
done