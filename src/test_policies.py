import os

from utils.logging import get_logger, Logger

import random
import matplotlib
matplotlib.use('TkAgg')
import argparse
import yaml
from types import SimpleNamespace

from main import recursive_dict_update
from run import run_sequential

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model_path", type=str)
    parser.add_argument("--config", type=str, default="mappo_egg")
    parser.add_argument("--env-config", type=str, default="egg_large")
    parser.add_argument("--model", type=str, default="beegfs/EPyMARL/models/mappo_seed2_MultiAgentEthicalGathering-large-v1_12_04_00_28_135987")
    test_args = parser.parse_args()

    # load env config as a name space
    config = yaml.safe_load(open("config/default.yaml"))
    recursive_dict_update(config, yaml.safe_load(open(f"config/envs/{test_args.env_config}.yaml")))
    recursive_dict_update(config, yaml.safe_load(open(f"config/algs/{test_args.config}.yaml")))

    args = SimpleNamespace(**config)
    logger = Logger(get_logger())

    args.evaluate = True
    args.render = False
    args.use_cuda = False
    # Get current directory
    args.checkpoint_path = os.path.join(os.path.dirname(os.getcwd()), test_args.model)
    args.env_args["seed"] = random.randint(0, 1000000)
    args.batch_size_run = 1
    args.device = "cpu"
    args.test_nepisode = 20
    args.runner = "episode"

    run_sequential(args, logger)




