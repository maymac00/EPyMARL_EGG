import time

import numpy as np
from EthicalGatheringGame import NormalizeReward, StatTracker
from EthicalGatheringGame.presets import lcarge

from src.modules.agents import RNNNSEGGAgent

import matplotlib
import os
matplotlib.use('TkAgg')
import gymnasium as gym
import torch as th
import argparse
import yaml
from types import SimpleNamespace


env = gym.make("MultiAgentEthicalGathering-large-v1",
         donation_capacity=10,
         we=[1, 2.6],
         efficiency=[0.2, 0.2, 0.67, 0.2, 0.67],)

parser = argparse.ArgumentParser()
#parser.add_argument("--model_path", type=str)
parser.add_argument("--config", type=str, default="ippo_egg")
parser.add_argument("--env-config", type=str, default="egg_large")
args = parser.parse_args()

# load env config as a name space
env_config = SimpleNamespace(**yaml.safe_load(open(f"config/envs/{args.env_config}.yaml")))
env_args = env.unwrapped.__dict__
alg_config = SimpleNamespace(**yaml.safe_load(open(f"config/algs/{args.config}.yaml")))
# combine namespaces
config = SimpleNamespace(**vars(env_config), **vars(alg_config))


model_path = "epymarl/results/models/ippo_egg_seed2_MultiAgentEthicalGathering-very-large-demo/40005000"

agents = RNNNSEGGAgent(env.unwrapped.observation_space[0].shape, config)
pass