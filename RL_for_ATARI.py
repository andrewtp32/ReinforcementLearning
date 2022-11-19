# This is a project that uses RL with the Atari game, Breakout.

""" ---------- 1. IMPORT DEPENDENCIES ---------- """
import gym
from stable_baselines3 import A2C
# This allows you to vectorize your environments. Basically, vectorise allows you to run multiple environments at once.
# We plan to run four environments at once.
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import  evaluate_policy
# This allows us to work with atari environments
from stable_baselines3.common.env_util import make_atari_env
import os

''' ---------- 2. TEST ENVIRONMENT ---------- '''
# You need to download the rar files to be able to work with the atari environments. It doesnt just come with the
# import anymore. You must paste the .rar files into your project folder.
# Link is: http://www.atarimania.com/roms/Roms.rar

# Then, you must run this command into you terminal (it will point to the
# ROMS directories): python -m atari_py.import_roms ./ROMS/ROMS

environment_name = 'Breakout-v0'
# make environment
env = gym.make(environment_name)
# observe environment
env.reset()


''' ---------- 3. VECTORISE ENVIRONMENT AND TRAIN MODEL ---------- '''
''' ---------- 4. SAVE AND RELOAD MODEL ---------- '''
''' ---------- 5. EVALUATE AND TEST ---------- '''