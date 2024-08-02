#!/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
import random
from pathlib import Path
import torch
sys.path.append("../../")
from mappo.config import get_config
from mappo.envs.case_study.case_study_env import CaseStudyEnv
from mappo.runner.shared.case_study_runner import CaseStudyRunner as Runner


def parse_args(args, parser):
    parser.add_argument('--model_name', type=str, default='', help="Which model to uese")
    parser.add_argument('--max_new_tokens', type=int, default=10, help="max_new_tokens")
    parser.add_argument('--vacab_size', type=int, default=32000)
    parser.add_argument('--gradient_cp_steps', type=int, default=1)
    all_args = parser.parse_known_args(args)[0]

    return all_args

def build_run_dir(all_args):
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/scripts/results") / all_args.experiment_name / "case_study" / all_args.algorithm_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
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
    return run_dir

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    all_args.episode_length = 32
    all_args.n_rollout_threads = 4
    all_args.log_interval = 1
    all_args.critic_lr = 5e-5
        
    run_dir = build_run_dir(all_args)

    # seed
    random.seed(all_args.seed)
    np.random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)  # If you're using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    envs = CaseStudyEnv(all_args.n_rollout_threads)

    config = {
        "all_args": all_args,
        "envs": envs,
        "num_agents": envs.num_agents,
        "run_dir": run_dir
    }

    runner = Runner(config)
    runner.run()

    # post process
    if envs is not None:
        envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
