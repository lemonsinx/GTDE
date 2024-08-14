env_name="battle_v3"
algo="gac"
exp="battles"
seed_max=5
for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python ../train.py --env_name ${env_name} --algo ${algo} --exp ${exp} \
  --value_coef 0.1 --ent_coef 0.08 --gamma 0.95 --learning_rate 1e-4
done