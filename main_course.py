"""
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
"""
import os

''' ---------- 1. IMPORT DEPENDENCIES ---------- '''
import gym
from stable_baselines3 import PPO
# DummyVecEnv allows you to vectorize environments. Allows for training multiple environments at the same time.
from stable_baselines3.common.vec_env import DummyVecEnv
# evaluate_policy makes it easier to see how well model is running.
from stable_baselines3.common.evaluation import evaluate_policy

''' ---------- 2. LOAD ENVIRONMENT ---------- '''
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
        # two actions available (i.e. the action is only capable of being a "1" or "0"). This is just where the
        # action is GENERATED; the action is actually TAKEN in the next line.
        action = env.action_space.sample()
        # unpack our env.step function. This is where the action is ACTUALLY TAKEN.
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

''' ---------- 3. TRAINING ---------- '''
'''
~ Choosing Algorithms ~
There are a number of algorithms available through Stable Baselines. We can easily switch between these.
Certain algorithms will perform better for certain environments. OFten it helps to review literature in order to determine the best approach.
For a given action space it is best to choose:
1. Discrete Single Process: DQN
2. Discrete Multi Processed: PPO or A2C
3. Continuous Single Process: SAC or TD3
4. Continuous Multi Processed: PPO or A2C

~ Understanding Training Metrics ~ 
1. Evaluation Metrics: ep_len_mean - episode length mean, ep_rew_mean - episode 
reward means 
2. Time Metrics: fps - frames per second, iterations - how many times you've gone through, time_elapsed 
- how long its been running in total time, total_timesteps - how many steps youve taken in an episode 
3. Loss 
Metrics: entropy_loss, policy_loss, value_loss 
4. Other Metrics: explained_variance - how much environment variance the agent is able to explain, 
learning_rate - how fast policies are updating, n_updates - total amount of updates agent has made 
'''

''' ---------- TRAIN AN RL MODEL ---------- '''
log_path = os.path.join('Training', 'Logs')

# recreate the environment
env = gym.make(environment_name)
# wrap our environment into the DummyVecEnv wrapper. Allows us to work with our environment while its wrapped in a
# dummy vector environment. Think of it as a wrapper for a non-vectorized environment.
env = DummyVecEnv([lambda: env])
# Defining our model. Think of it as defining our agent, or even, think of it as defining the rules for how the agent
# behaves within the environment. The first argument defines the policy that we want to use (MLP is
# multi-layer-processing which basically means its a regular neural network). Stable Baselines 3 has three policy
# types: MlpPolicy, CnnPolicy, and MultiInputPolicy. The second argument is the definition of the environment.
# verbose=1 because we want to log the results of that model. then we specify our log path.
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=20000)

''' ---------- SAVE AND RELOAD MODEL ---------- '''
# define a path to save our model to
PPO_Path = os.path.join('Training', 'Saved_Models', 'PPO_Model_Cartpole')
# save madel into path
model.save(PPO_Path)
# Delete model and then reload it. This technique kind of simulated "deployment". You're gunna be reloading from your
# model each time you test.
del model
# reload model back into memory
model = PPO.load(PPO_Path, env=env)

''' ---------- 4. EVALUATE MODEL ---------- '''
'''
~ Evaluation Metrics ~ 
There are a number of metrics available from the models when trained. There are two core 
values that you should pay attention to: 
1. ep_len_mean - the average time (in frames) a particular episode lasted before "done".
2. ep_rew_mean - the average reward that the agent accumulated per episode.

~ Monitoring in Tensorboard ~ 
You are able to review evaluation, time, and training metrics using Tensorboard. To use 
Tensorboard, you must specify a logging directory when you initialise your model:
in code -> model = A2C("CnnPolicy", env, verbose=1, tensorboard_log=log_path)
in terminal -> !tensorboard --logdir=<your_log_path_here>
'''
# This line evaluates/tests our policy. we pass through our model, the environment, the number of episodes we wish to
# do testing with, and whether or not we wish to render the environment.
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(f"Average reward = {mean_reward}, Standard deviation of reward = {std_reward}")
# now we close out our environment
env.close()

''' ---------- 5. TEST MODEL ---------- '''

# look at section 2 for more comments on this block of code
episodes = 5
for episode in range(1, episodes + 1):
    # Changed "state" to "obs".
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        # Here, we make a key change from section 2. "model.predict()" will output our "action" and our "states".
        # Previously, we generated random actions to train the model. Now, we will generate actions based upon
        # observations made by our model. We will pass our observations through the function as arguments. WE ARE
        # USING ARE MODEL HERE!!!!!!!!
        action, _ = model.predict(observation=obs)
        # Changed "n_state" to "obs".
        obs, reward, done, info = env.step(action)
        # With a successful model, your total score will be 200. Your total score will be 200 because your score is
        # the total sumation of your rewards after each step. Our environment rewards the agent with 1 reward-point
        # after every time step (or "frame") that the pole remains upright. Since there are 200 total frames in the
        # environment, a perfect score is 200 reward-points. If you do not understand the way the reward system
        # works, consult the environment code.
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

''' ---------- 6. VIEW LOGS IN TENSORBOARD ---------- '''
'''
If you are training a much larger or much more sophisticated environment, you may want to view the training logs 
with Tensorport. 
'''
# Specify a path to your log file.
training_log_path = os.path.join(log_path, "PPO_10")

