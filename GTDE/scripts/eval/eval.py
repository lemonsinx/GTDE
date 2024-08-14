#!/usr/bin/env python
import os
import sys

sys.path.append("/home/ictzx/GTGE")
import setproctitle
import numpy as np
import torch
from GTDE.config import get_config
from GTDE.scripts.train.train_smacv2 import parse_smacv2_distribution, make_eval_env, parse_args
from GTDE.algorithms.mappo.mappo import MAPPO as TrainAlgo
from GTDE.algorithms.mappo.algorithm.MAPPOPolicy import MAPPOPolicy as Policy

"""eval script for SMACV2."""


def _t2n(x):
    return x.detach().cpu().numpy()


@torch.no_grad()
def eval_mode(config, index, path):
    args = config['all_args']
    eval_envs = config['eval_envs']
    device = config['device']
    num_agents = config['num_agents']

    share_observation_space = eval_envs.share_observation_space[0] if args.use_centralized_V else \
        eval_envs.observation_space[0]

    policy = Policy(args,
                    eval_envs.observation_space[0],
                    share_observation_space,
                    eval_envs.action_space[0],
                    device=device)
    policy.actor.load_state_dict(torch.load(path))

    trainer = TrainAlgo(args, policy, device=device)

    eval_battles_won = 0
    eval_episode = 0
    eval_obs, eval_share_obs, eval_available_actions = eval_envs.reset()
    eval_rnn_states = np.zeros((args.n_eval_rollout_threads, num_agents, args.recurrent_N, args.hidden_size),
                               dtype=np.float32)
    eval_masks = np.ones((args.n_eval_rollout_threads, num_agents, 1), dtype=np.float32)
    while True:
        eval_actions, eval_rnn_states = \
            trainer.policy.act(np.concatenate(eval_obs),
                               np.concatenate(eval_rnn_states),
                               np.concatenate(eval_masks),
                               np.concatenate(eval_available_actions),
                               deterministic=True)
        eval_actions = np.array(np.split(_t2n(eval_actions), args.n_eval_rollout_threads))
        eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), args.n_eval_rollout_threads))

        # Obser reward and next obs
        eval_obs, eval_share_obs, _, eval_dones, eval_infos, eval_available_actions = eval_envs.step(
            eval_actions)
        eval_dones_env = np.all(eval_dones, axis=1)

        eval_rnn_states[eval_dones_env == True] = np.zeros(
            ((eval_dones_env == True).sum(), num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)

        eval_masks = np.ones((args.n_eval_rollout_threads, num_agents, 1), dtype=np.float32)
        eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), num_agents, 1),
                                                      dtype=np.float32)

        for eval_i in range(args.n_eval_rollout_threads):
            if eval_dones_env[eval_i]:
                eval_episode += 1
                if eval_infos[eval_i][0]['won']:
                    eval_battles_won += 1

        if eval_episode >= args.eval_episodes:
            eval_win_rate = eval_battles_won / eval_episode
            print("eval win rate is {}.".format(eval_win_rate))
            np.save(args.algorithm_name + "_eval_win_rate" + f"_{index}_{args.seed}" + ".npy", eval_win_rate)
            break


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # set process name
    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    eval_envs = make_eval_env(all_args)
    num_agents = parse_smacv2_distribution(all_args)['n_units']

    if all_args.algorithm_name in ["mappo", "ippo", "GTGE"]:
        if all_args.algorithm_name == "mappo":
            all_args.use_GTGE = False
            all_args.use_centralized_V = True
            print("u are choosing to use mappo")
        elif all_args.algorithm_name == "ippo":
            all_args.use_GTGE = False
            all_args.use_centralized_V = False
            print("u are choosing to use ippo")
        elif all_args.algorithm_name == "GTGE":
            all_args.use_GTGE = True
            all_args.use_centralized_V = False
            print("u are choosing to use GTGE")
    else:
        raise NotImplementedError

    config = {
        "all_args": all_args,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device
    }
    all_args.model_dir = "../../../result/10v10/zerg/ippo"
    path = [all_args.model_dir + "/" + i + "/models/actor.pt" for i in os.listdir(all_args.model_dir)]
    for i, j in enumerate(path):
        eval_mode(config, i, j)


if __name__ == "__main__":
    main(sys.argv[1:])
