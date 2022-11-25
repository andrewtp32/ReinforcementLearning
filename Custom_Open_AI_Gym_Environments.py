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
'''# discrete
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
print(example)'''

# ---------- 3. Building Custom Environment ----------
'''
Goals:
- build an agent to give us the best shower possible
- THe temperature of the shower is going to be changing at random
- the agent will regulate the temperature
- maintain a water temp of 37-39 degrees 
'''


# We make our class here. Notice how we pass through "Env". Env is a class within the gym library.
class ShowerEnv(Env):
    def __init__(self):
        # We have three choices of action: Turn the heat up, turn the heat down, or remain held. Hence, Discrete(3).
        self.action_space = Discrete(3)
        # Observation space is just a scalar value (or 1x1 matrix) that represents the water temperature. The value
        # of the observation ranges from 0 to 100.
        self.observation_space = Box(low=0, high=100, shape=(1,))
        # This defines our initial state.
        self.state = 38 + random.randint(-3, 3)
        # This value represents how long the agent showers for.
        self.shower_length = 60

    def step(self, action):
        # Apply temp adj. This applies the impact of the agent's action onto the state. An action_space of "0" will
        # represent decreasing heat by one degree, a "1" will represent no chnge, and a "2" represents increasing
        # heat by one degree. Each action space value is decreased by 1 (this makes updating the state easier):
        # 0 -1 = -1 temperature
        # 1 -1 = 0
        # 2 -1 = 1 temperature
        self.state += action - 1

        # Decrease shower time. The remaining time to take a show is reduced by 1 for each time step taken.
        self.shower_length -= 1

        # Calculate reward.
        if 37 <= self.state <= 39:
            reward = 1
        else:
            reward = -1

        # See if shower time is over
        if self.shower_length <= 0:
            done = True
        else:
            done = False

        # If we want to pass additional info, we would put it in this dictionary.
        info = {}

        return self.state, reward, done, info

    def render(self, mode="human"):
        # Implement viz
        pass

    def reset(self):
        # Reset everything back to its original state
        self.state = np.array([38 + random.randint(-3, 3)]).astype(float)
        self.shower_length = 60
        return self.state


# Declare environment
env = ShowerEnv()

# ---------- 4. Test Environment ----------
episodes = 5
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

# ---------- 5. Train Model ----------
log_path = os.path.join('Training', 'Logs')
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
# learn our model
model.learn(total_timesteps=100_000)

# ---------- 6. Save Model ----------
shower_path = os.path.join('Training', 'Saved_Models', 'Shower_Model_PPO')
model.save(path=shower_path)
# delete model and then load it in
del model
model = PPO.load(shower_path, env)
evaluation = evaluate_policy(model, env, n_eval_episodes=10, render=False)
print(evaluation)