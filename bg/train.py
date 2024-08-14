import argparse
import os
from algo import spawn_ai
from algo import tools
from senarios.senario_battle import play
from tensorboardX import SummaryWriter
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def linear_decay(epoch, x, y):
    min_v, max_v = y[0], y[-1]
    start, end = x[0], x[-1]
    if epoch == start:
        return min_v
    eps = min_v
    for i, x_i in enumerate(x):
        if epoch <= x_i:
            interval = (y[i] - y[i - 1]) / (x_i - x[i - 1])
            eps = interval * (epoch - x[i - 1]) + y[i - 1]
            break
    return eps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, choices={'battle_v3', 'gather_v3'}, required=True)
    parser.add_argument('--algo', type=str, choices={'iac', 'mfac', "gac", "maac"}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--save_every', type=int, default=20, help='decide the self-play update interval')
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--n_round', type=int, default=2000, help='set the trainning round')
    parser.add_argument('--render', default=False, action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--render_time', type=int, default=20)
    parser.add_argument('--map_size', type=int, default=40, help='set the size of map')  # then the amount of agents is 64
    parser.add_argument('--max_steps', type=int, default=400, help='set the max steps')
    parser.add_argument('--cuda', type=bool, default=True, help='use the cuda')
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--train_mode', default=True, action='store_false')
    parser.add_argument('--value_coef', type=float, default=0.1)
    parser.add_argument('--ent_coef', type=float, default=0.08)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--exp', type=str, default="tune")
    parser.add_argument('--use_u', default=False, action='store_true')
    parser.add_argument('--use_g', default=False, action='store_true')
    parser.add_argument('--use_a', default=False, action='store_true')
    args = parser.parse_args()

    if args.env_name == 'battle_v3':
        from pettingzoo.magent import battle_v3
        # Initialize the environment
        env = battle_v3.env(
            map_size=args.map_size,
            minimap_mode=False,
            step_reward=-0.005,
            dead_penalty=-0.1,
            attack_penalty=-0.1,
            attack_opponent_reward=0.2,
            max_cycles=args.max_steps,
            extra_features=False
        )
        handles = env.unwrapped.env.get_handles()
        models = [spawn_ai(args, env, handles[0], args.algo + '-me'),
                  spawn_ai(args, env, handles[1], args.algo + '-opponent')]
    elif args.env_name == 'gather_v3':
        from pettingzoo.magent import gather_v3
        # Initialize the environment
        env = gather_v3.env(
            minimap_mode=False,
            step_reward=-0.01,
            attack_penalty=-0.1,
            dead_penalty=-1,
            attack_food_reward=0.5,
            max_cycles=args.max_steps,
            extra_features=False
        )
        handles = env.unwrapped.env.get_handles()
        handles = [handles[1]]
        models = [spawn_ai(args, env, handles[0], args.algo + '-me')]
    else:
        raise ValueError

    run_dir = Path(os.path.join(BASE_DIR, 'data/{}'.format(args.algo))) / args.exp
    if not run_dir.exists():
        os.makedirs(str(run_dir))
        curr_run = "run1"
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

    log_dir = f"{run_dir}/tmp"
    model_dir = f"{run_dir}/models"
    render_dir = f"{run_dir}/render"

    start_from = 0

    writter = SummaryWriter(log_dir)

    runner = tools.Runner(args, env, handles, models, play, model_dir=model_dir, render_dir=render_dir, writter=writter)
    if not args.train_mode and len(models) == 2:
        models[0].load(args.load_dir)
        models[1].load(args.load_dir)
    elif not args.train_mode and len(models) == 1:
        models[0].load(args.load_dir)

    for k in range(start_from, start_from + args.n_round):
        eps = linear_decay(k, [0, int(args.n_round * 0.8), args.n_round], [1, 0.2, 0.1])
        runner.run(eps, k, args.n_round)

    runner.models[0].save(runner.model_dir + '-0', 2000)
    if len(models) == 2:
        runner.models[1].save(runner.model_dir + '-1', 2000)
    runner.writter.export_scalars_to_json(str(log_dir + '/summary.json'))
    runner.writter.close()
