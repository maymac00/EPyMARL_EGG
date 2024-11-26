from collections.abc import Iterable
import warnings

import gymnasium as gym
from gymnasium.spaces import flatdim
from gymnasium.wrappers import TimeLimit
import numpy as np

from .multiagentenv import MultiAgentEnv
from .wrappers import FlattenObservation
from .pz_wrapper import PettingZooWrapper  # noqa
import envs.pretrained as pretrained
from .gymma import GymmaWrapper

class GymmomaWrapper(GymmaWrapper):

    def __init__(
            self,
            key,
            time_limit,
            pretrained_wrapper,
            seed,
            common_reward,
            reward_scalarisation,
            **kwargs,
    ):
        super().__init__(key, time_limit, pretrained_wrapper, seed, common_reward, reward_scalarisation, **kwargs)
        self.common_reward = False
        self.reward_scalarisation = ""

    def step(self, actions):
        actions = [int(a) for a in actions]
        obs, reward, done, truncated, self._info = self._env.step(actions)
        self._obs = self._pad_observation(obs)

        if not isinstance(reward[0], Iterable):
            warnings.warn("The environment does not return multiple rewards for each agent.")
        if isinstance(done, Iterable):
            done = all(done)
        return self._obs, reward, done, truncated, self._info