import os
# Avoid reinstalling packages that are available on edstem
if not os.getenv("ED_COURSE_ID"):
    !pip install tensorflow stable_baselines3 torch collections gym box2d-py --user

# Import gym libraries
import gym 
from gym import Env # the supperclass to build our own environment
# All different types of spaces available in Gym
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 

# Import helpers
import numpy as np
import random

#Import stable bbaselines libraries
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Define a discrete space
disc = Discrete(3)

# Sample the discrete space for a value (between 0 and 2)
disc.sample()

# Define a box space
box = Box(0,1,shape=(3,3))

#TODO: Sample the box space for a value
box.sample()

# Define a tuple space and combine a discrete and box spaces
tup = Tuple((Discrete(2), Box(0,100, shape=(1,))))

#TODO: Sample the tuple space for a value
tup.sample()

# Define a dict space
dic = Dict({'height':Discrete(2), "speed":Box(0,100, shape=(1,))}).sample()

# Define a multibinary space
multibi = MultiBinary(4)

#TODO: Sample the multibinary space for a value
multibi.sample()

# Define a multidiscrete space
multidi = MultiDiscrete([5,2,2])

#TODO: Sample the multidiscrete space for a value
multidi.sample()

# Define a shower environment class with four key functions
class ShowerEnv(Env):
    # Define a function to initialize the environment
    def __init__(self):
        # Define the discrete action space: 
        # Actions we can take, down, hold, up
        self.action_space = Discrete(3)
        # Define a temperature range from 0 to 100
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        # Set initial state: starting temp is 38 +- 3
        self.state = 38 + random.randint(-3,3)
        # Set shower length: set to 60 seconds for testing
        self.shower_length = 60

    # Define the step function for what to do in one action step    
    def step(self, action):
        # Apply impact of the action on current state
        # 0 -1 = -1 temperature
        # 1 -1 = 0 
        # 2 -1 = 1 temperature 
        self.state += action -1 
        # Reduce shower length by 1 second at each action
        self.shower_length -= 1 
        
        # Calculate reward
        # If the temperature is within preferred range, the reward is positive
        if self.state >= 37 and self.state <= 39: 
            reward = 1 
        # If the reward is outside of preferred range, the reward is negative 
        else: 
            reward = -1 
        
        # Check if shower is done
        if self.shower_length <= 0: 
            done = True
        else:
            done = False
        
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    # For this lab, we will not implement a visualization of the environment
    def render(self):
        # Implement viz
        pass
    
    # Define function to reset the environment for the next run
    def reset(self):
        # Reset shower temperature to a random value between 35 and 41
        self.state = np.array([38 + random.randint(-3,3)]).astype(float)
        # Reset shower time
        self.shower_length = 60 
        return self.state
 
# Initialize the environment
env=ShowerEnv()

#TODO: Write code to sample the environment's observation space
env.observation_space.sample()

#TODO: Write code to sample the environment's action space
env.action_space.sample()

# Reset the environment
env.reset()

# Test five episodes of taking random Actions
# in the environment
episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
    
env.close()

# Define a path for where to output the training log files
log_path = os.path.join('ReinforcementLearning/ShowerEnvironment/Training', 'Logs')

# Set up model, pass in 'MlpPolicy' as the policy we use, 
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

# Start to train the model
model.learn(total_timesteps=500000)

#Save the model
model.save('PPO')

# Evaluate the model (with render set to false)
evaluate_policy(model, env, n_eval_episodes=10, render=False)
