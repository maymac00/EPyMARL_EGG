import time

import numpy as np
from EthicalGatheringGame import NormalizeReward, StatTracker
from EthicalGatheringGame.presets import large

from src.modules.agents import RNNNSEGGAgent

import matplotlib
import os
matplotlib.use('TkAgg')
import gymnasium as gym
import torch as th
import argparse
import yaml
from types import SimpleNamespace
from envs.gymma import GymmaWrapper

parser = argparse.ArgumentParser()
#parser.add_argument("--model_path", type=str)
parser.add_argument("--config", type=str, default="ippo_egg")
parser.add_argument("--env-config", type=str, default="egg_large")
args = parser.parse_args()

# load env config as a name space
env_config = SimpleNamespace(**yaml.safe_load(open(f"config/envs/{args.env_config}.yaml")))
alg_config = SimpleNamespace(**yaml.safe_load(open(f"config/algs/{args.config}.yaml")))
# combine namespaces
config = SimpleNamespace(**vars(env_config), **vars(alg_config))

env = GymmaWrapper(**env_config.env_args)

model_path = "epymarl/results/models/ippo_egg_seed2_MultiAgentEthicalGathering-very-large-demo/40005000"



pass