#!/usr/bin/env python
import sys
sys.path.append("/home/user/GTDE")
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from GTDE.config import get_config
from GTDE.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
"""Train script for SMACV2."""


def parse_smacv2_distribution(args):
    units = args.units.split('v')
    distribution_config = {
        "n_units": int(units[0]),
        "n_enemies": int(units[1]),
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "map_x": 32,
            "map_y": 32,
        }
    }
    if 'protoss' in args.map_name:
        distribution_config['team_gen'] = {
            "dist_type": "weighted_teams",
            "unit_types": ["stalker", "zealot", "colossus"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        }
    elif 'zerg' in args.map_name:
        distribution_config['team_gen'] = {
            "dist_type": "weighted_teams",
            "unit_types": ["zergling", "baneling", "hydralisk"],
            "weights": [0.45, 0.1, 0.45],
            "observe": True,
        }
    elif 'terran' in args.map_name:
        distribution_config['team_gen'] = {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        }
    return distribution_config


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2v2":
                from GTDE.envs.starcraft2.SMACv2 import SMACv2
                env = SMACv2(capability_config=parse_smacv2_distribution(all_args), map_name=all_args.map_name)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2v2":
                from GTDE.envs.starcraft2.SMACv2 import SMACv2
                env = SMACv2(capability_config=parse_smacv2_distribution(all_args), map_name=all_args.map_name)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='3m',
                        help="Which smac map to run on")
    parser.add_argument('--units', type=str, default='10v10')  # for smac v2
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_false', default=True)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_false', default=True)

    all_args = parser.parse_known_args(args)[0]

    return all_args


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

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) + "_" +
                              str(all_args.units) +
                              "_seed" + str(all_args.seed),
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
        all_args = wandb.config
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # set process name
    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    if all_args.algorithm_name in ["mappo", "ippo", "GTGE"]:
        from GTDE.runner.onpolicy.smac_runner import SMACRunner as Runner
        if all_args.algorithm_name == "mappo":
            all_args.use_GTGE = False
            all_args.use_centralized_V = False
            all_args.use_mappo = True
            print("u are choosing to use mappo")
        elif all_args.algorithm_name == "ippo":
            all_args.use_GTGE = False
            all_args.use_centralized_V = False
            all_args.use_mappojo = False
            print("u are choosing to use ippo")
        elif all_args.algorithm_name == "GTGE":
            all_args.use_GTGE = True
            all_args.use_centralized_V = False
            all_args.use_mappojo = False
            print("u are choosing to use GTGE")
    else:
        raise NotImplementedError

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    if all_args.env_name == 'StarCraft2v2':
        num_agents = parse_smacv2_distribution(all_args)['n_units']
    else:
        raise ValueError

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)

    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
