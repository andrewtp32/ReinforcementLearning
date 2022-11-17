'''

Followed this github project:
https://github.com/nicknochnack/ReinforcementLearningCourse
https://www.youtube.com/watch?v=Mut_u40Sqz4&list=PLgNJO2hghbmid6heTu2Ar68CzJ344nUHy&index=1

OpenAI environments using GYM:
https://www.gymlibrary.dev
Environments are represented by "spaces". Many different types of spaces. Some include:
1. Box - n dimensional tensor, range of values ~ use for a continuous value ~ e.g. Box(0,1,shape=(3,3))
2. Discrete - set of items ~ a discrete set with value used to describe an action~ e.g. Discrete(3)
3. Tuple - tuple together different spaces (Bos or Discrete) ~ e.g. Tuple((Discrete(2), Box(0,100,shape=(1,))))
4. Dict - dictionary of spaces (Bos or Discrete) ~ e.g. Dict({'height':Discrete(2), "speed":Box(0,100,shape=(1,))})
5. MultiBinary - one hot encoded set of binary values ~ e.g. Multibinary(4)
6. MultiDiscrete - multiple discrete values ~ e.g. MultiDiscrete([5,2,2])

'''

''' ---------- IMPORT DEPENDENCIES ---------- '''
import gym
from stable_baselines3 import PPO
# DummyVecEnv allows you to vectorize environments. Allows for training multiple environments at the same time.
from stable_baselines3.common.vec_env import DummyVecEnv
# evaluate_policy makes it easier to see how well model is running.
from stable_baselines3.common.evaluation import evaluate_policy

''' ---------- LOAD ENVIRONMENT ---------- '''
# This is case-sensitive. This is a mapping to the pre-installed gym environment.
environment_name = 'CartPole-v0'
# Here, we are making our environment.
env = gym.make(environment_name)
'''
the main 'env.' (aka, environment) functions:
1. env.reset() - reset the environment and obtain initial observations
2. env.render() - visualize the environment
3. env.step() - apply an action to the environemtn
4. env.close( - close down the render frame
'''
# We are going to test the cart-pole environment 5 times. We have 5 "episodes". Think of an episode as one full game
# within the environment. Some environments have a fixed episode length (meaning a fixed number of frames,
# like a movie), e.g. CartPole, which is 200 frames. Others are continuous, e.g. Breakout or PacMan, where you play
# until you run out of lives.
episodes = 5
# loop through each episode
for episode in range(1, episodes + 1):
    # Resetting our environment. Obtain an initial set of observations. Later on, we will pass along these
    # observations along to our agent, at whichpoint our agent will decide on an action.
    state = env.reset()
    # Set up some temporary variables.
    done = False
    score = 0

    # run until done
    while not done:
        # Render allows us to view the graphical representation of the environment.
        env.render()
        # Here, we will generate a random action; in contrast to the agent "considering" its observations and then
        # generating a useful action. In this case, our action space is a Discrete(2), meaning that there are only
        # two actions available (i.e. the action is only capable of being a "1" or "0").
        action = env.action_space.sample()
        # unpack our env.step function
        n_state, reward, done, info = env.step(action)
        # accumulating our reward
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
# close the rendering
env.close()

''' ---------- UNDERSTNADING THE ENVIRONMENT ---------- '''
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

# Look at this link to understand the anatomy of the environment more. The code for all the other environments can be
# found here.


