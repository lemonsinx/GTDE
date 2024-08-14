from . import q_learning
from . import ac

IQL = q_learning.DQN
MFQ = q_learning.MFQ
AC = ac.ActorCritic
MFAC = ac.MFAC
GAC = ac.groupac
MAAC = ac.MAAC


def spawn_ai(args, env, handle, human_name):
    algo_name = args.algo
    if algo_name == 'mfq':
        model = MFQ(env, human_name, handle, args.max_steps, memory_size=80000)
    elif algo_name == 'iql':
        model = IQL(env, human_name, handle, args.max_steps, memory_size=80000)
    elif algo_name == 'iac':
        model = AC(args, env, human_name, handle)
    elif algo_name == 'mfac':
        model = MFAC(args, env, human_name, handle)
    elif algo_name == 'gac':
        model = GAC(args, env, human_name, handle)
    elif algo_name == 'maac':
        model = MAAC(args, env, human_name, handle)
    else:
        raise ValueError
    if args.cuda:
        model = model.cuda()
    return model