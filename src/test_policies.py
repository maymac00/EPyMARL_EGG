import os

from utils.logging import get_logger, Logger

import random
import matplotlib
matplotlib.use('TkAgg')
import argparse
import yaml
from types import SimpleNamespace

from main import recursive_dict_update
from controllers import REGISTRY as mac_REGISTRY
from runners import REGISTRY as r_REGISTRY
from components.transforms import OneHot
import torch as th
from run import evaluate_sequential, run_sequential
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model_path", type=str)
    parser.add_argument("--config", type=str, default="ippo_egg_mo_ns")
    parser.add_argument("--env-config", type=str, default="egg_large_mo")
    parser.add_argument("--model", type=str, default="beegfs/EPyMARL/models/reference_policy_eff40_seed4_MultiAgentEthicalGathering-large-mo-v1_12_05_20_30_371714")
    test_args = parser.parse_args()

    # load env config as a name space
    config = yaml.safe_load(open("config/default.yaml"))
    recursive_dict_update(config, yaml.safe_load(open(f"config/envs/{test_args.env_config}.yaml")))
    recursive_dict_update(config, yaml.safe_load(open(f"config/algs/{test_args.config}.yaml")))

    args = SimpleNamespace(**config)
    logger = Logger(get_logger())
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    # TODO:parametrise this
    args.env_args["efficiency"] = 0.6
    args.evaluate = True
    args.render = False
    args.use_cuda = False
    # Get current directory
    args.checkpoint_path = os.path.join(os.path.dirname(os.getcwd()), test_args.model)
    #last_ckpt = max([int(d) for d in os.listdir(args.checkpoint_path) if os.path.isdir(os.path.join(args.checkpoint_path, d))])
    #args.checkpoint_path = os.path.join(args.checkpoint_path, str(last_ckpt))

    args.env_args["seed"] = random.randint(0, 1000000)
    args.batch_size_run = 1
    args.device = "cpu"
    args.test_nepisode = 500
    args.runner = "episode"

    runner, mac, learner = run_sequential(args, logger)

    env = runner.env._env.unwrapped
    print(env.agents[0].acc_r_vec)
    env.print_results()
    env.plot_results("median", save_path=args.checkpoint_path+"/results.png")


