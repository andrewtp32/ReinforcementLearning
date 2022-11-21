'''
This script is based upon the youtube tutorial:
https://www.youtube.com/watch?v=Mut_u40Sqz4&list=PLgNJO2hghbmid6heTu2Ar68CzJ344nUHy&index=1
'''

# ---------- 1. Import Dependencies ----------
# import gym stuff
import gym
from gym import Env  # this is a superclass that will allow us to build our own environment
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete  # These are all the types of spaces

# import helpers
import numpy as np
import random
import os

# import stable baselines stuff
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# ---------- 2. Types of Spaces ----------
# discrete
example = Discrete(3).sample()
print(example)
# box
example = Box(0, 1, shape=(3, 3)).sample()
print(example)
example = Box(2, 5, shape=(6, 2)).sample()
print(example)
example = Box(2, 5, shape=(4, 1)).sample()
print(example)
example = Box(2, 5, shape=(4,)).sample()
print(example)
# tuple
example = Tuple((Discrete(3), Box(0, 1, shape=(3, 3)))).sample()
print(example)
# dict
example = Dict({'height': Discrete(5), "speed": Box(0, 10, shape=(3, 2))})
print(example)
example = Dict({'height': Discrete(5), "speed": Box(0, 10, shape=(3,))}).sample()
print(example)
# multi binary
example = MultiBinary(4).sample()
print(example)
# multi discrete
example = MultiDiscrete([5, 2, 2]).sample()
print(example)

# ---------- 3. Building Custom Environment ----------
'''
Goals:
- build an agent to give us the best shower possible
- THe temperature of the shower is going to be changing at random
- the agent will regulate the temperature
- maintain a water temp of 37-39 degrees 
'''


class ShowerEnv(Env):
    def __init__(self):
        pass

    def step(self, action):
        pass

    def render(self, mode="human"):
        pass

    def reset(self):
        pass
