# This is a project that uses RL with the Atari game, Breakout.

""" ---------- 1. IMPORT DEPENDENCIES ---------- """
import gym
import ale_py
from stable_baselines3 import A2C
# This allows you to vectorise your environments. Basically, vectorise allows you to run multiple environments at once.
# We plan to run four environments at once.
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
# This allows us to work with atari environments
from stable_baselines3.common.env_util import make_atari_env
import os

''' ---------- 2. TEST ENVIRONMENT ---------- '''
# You need to download the rar files to be able to work with the atari environments. It doesnt just come with the
# import anymore. You must paste the .rar files into your project folder.
# Link is: http://www.atarimania.com/roms/Roms.rar

# Then, you must run this command into you terminal to install the environments (it will point to the
# ROMS directories): python -m atari_py.import_roms ./ROMS/ROMS

environment_name = "Breakout-v0"
# make environment
env = gym.make("ALE/Adventure-v5", render_mode="human")
# observe environment
env.reset()

# Here, we will test out our environment
episodes = 5
# loop through our episodes
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    # take random actions on the environment so the agent can see what the environment looks like.
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

''' ---------- 3. VECTORISE ENVIRONMENT AND TRAIN MODEL ---------- '''
''' ---------- 4. SAVE AND RELOAD MODEL ---------- '''
''' ---------- 5. EVALUATE AND TEST ---------- '''
