import numpy as np
import pandas as pd
import gym
from gym import spaces, logger
from gym.utils import seeding
from or_gym.utils import assign_env_config
import copy
from stable_baselines3 import PPO
import numpy as np
from train import NormalizingWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Type
import numpy as np
import or_gym
import gym
from gym import ObservationWrapper, spaces, logger
from or_gym.envs.classic_or.knapsack import BoundedKnapsackEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from torch import nn
from gym.utils import seeding
from or_gym.utils import assign_env_config
import copy

class KnapsackEnv(gym.Env):
    
    # Internal list of placed items for better rendering
    collected_items = []
    
    def __init__(self, *args, **kwargs):
        # Generate data with consistent random seed to ensure reproducibility
        self.N = 5
        self.max_weight = 10
        self.current_weight = 0
        self._max_reward = 10000
        self.mask = True
        self.seed = 0
        self.item_numbers = np.arange(self.N)
        self.item_weights = np.array([1,2,3,4,5])
        self.item_values = np.array([1,2,3,4,5])
        self.over_packed_penalty = 0
        self.randomize_params_on_reset = False
        self.collected_items.clear()
        # Add env_config, if any
        assign_env_config(self, kwargs)
        self.set_seed()

        obs_space = spaces.Box(
            0, self.max_weight, shape=(self.N + 1,), dtype=np.int32)
        self.action_space = spaces.Discrete(self.N)
        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(self.N,), dtype=np.uint8),
                "avail_actions": spaces.Box(0, 1, shape=(self.N,), dtype=np.uint8),
                "state": obs_space
                })
        else:
            self.observation_space = spaces.Box(
                0, self.max_weight, shape=(2, self.N + 1), dtype=np.int32)
        
        self.reset()

    def sample_action(self):
        return np.random.choice(self.item_numbers)

    def set_seed(self, seed=None):
        if seed == None:
            seed = np.random.randint(0, np.iinfo(np.int32).max)        
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        return self._RESET()

    def step(self, action):
        return self._STEP(action)
        
    def render(self):
        total_value = 0
        total_weight = 0
        for i in range(self.N) :
            if i in self.collected_items :
                total_value += self.item_values[i]
                total_weight += self.item_weights[i]
        print(self.collected_items, total_value, total_weight)
        
        # RlLib requirement: Make sure you either return a uint8/w x h x 3 (RGB) image or handle rendering in a window and then return `True`.
        return True
    

class BoundedKnapsackEnv(KnapsackEnv):
    
    def __init__(self, *args, **kwargs):
        self.N = 5
        self.item_limits_init = np.random.randint(1, 10, size=self.N, dtype=np.int32)
        self.item_limits = self.item_limits_init.copy()
        super().__init__()
        self.item_weights = np.array([1,2,3,4,5],dtype=np.int32)
        self.item_values =np.array([1,2,3,4,5],dtype=np.int32)

        assign_env_config(self, kwargs)

        obs_space = spaces.Box(
            0, self.max_weight, shape=(3, self.N + 1), dtype=np.int32)
        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(len(self.item_limits),), dtype=np.uint8),
                "avail_actions": spaces.Box(0, 1, shape=(len(self.item_limits),), dtype=np.uint8),
                "state": obs_space
            })
        else:
            self.observation_space = obs_space
        
    def _STEP(self, item):
        # Check item limit
        if self.item_limits[item] > 0:
            # Check that item will fit
            if self.item_weights[item] + self.current_weight <= self.max_weight:
                self.current_weight += self.item_weights[item]
                reward = self.item_values[item]
                if self.current_weight == self.max_weight:
                    done = True
                else:
                    done = False
                self._update_state(item)
            else:
                # End if over weight
                reward = 0
                done = True
        else:
            # End if item is unavailable
            reward = 0
            done = True
            
        return self.state, reward, done, {}

    def _update_state(self, item=None):
        if item is not None:
            self.item_limits[item] -= 1
        state_items = np.vstack([
            self.item_weights,
            self.item_values,
            self.item_limits
        ], dtype=np.int32)
        state = np.hstack([
            state_items, 
            np.array([[self.max_weight],
                      [self.current_weight], 
                      [0] # Serves as place holder
                ], dtype=np.int32)
        ])
        if self.mask:
            mask = np.where(self.current_weight + self.item_weights > self.max_weight, 0, 1).astype(np.uint8)
            mask = np.where(self.item_limits > 0, mask, 0)
            self.state = {
                "action_mask": mask,
                "avail_actions": np.ones(self.N, dtype=np.uint8),
                "state": state
            }
        else:
            self.state = state.copy()
        
    def sample_action(self):
        return np.random.choice(
            self.item_numbers[np.where(self.item_limits!=0)])
    
    def _RESET(self):
        if self.randomize_params_on_reset:
            self.item_weights = np.random.randint(1, 100, size=self.N, dtype=np.int32)
            self.item_values = np.random.randint(0, 100, size=self.N, dtype=np.int32)
            self.item_limits = np.random.randint(1, 10, size=self.N, dtype=np.int32)
        else:
            self.item_limits = self.item_limits_init.copy()

        self.current_weight = 0
        self._update_state()
        return self.state

def update_state(env, item=None):
    if item is not None:
        env.item_limits[item] -= 1
        env.collected_items.append(item)
    state_items = np.vstack([
        env.item_weights,
        env.item_values,
        env.item_limits
    ], dtype=np.int32)
    state = np.hstack([
        state_items, 
        np.array([[env.max_weight],
                    [env.current_weight], 
                    [0] # Serves as place holder
            ], dtype=np.int32)
    ])
    if env.mask:
        mask = np.where(env.current_weight + env.item_weights > env.max_weight, 0, 1).astype(np.uint8)
        mask = np.where(env.item_limits > 0, mask, 0)
        env.state = {
            "action_mask": mask,
            "avail_actions": np.ones(env.N, dtype=np.uint8),
            "state": state
        }
    else:
        env.state = state.copy()

def STEP(env,item):
    if env.item_limits[item] > 0:
            # Check that item will fit
            if env.item_weights[item] + env.current_weight <= env.max_weight:
                env.current_weight += env.item_weights[item]
                reward = env.item_values[item]
                if env.current_weight == env.max_weight:
                    done = True
                else:
                    done = False
                update_state(env,item)
            else:
                # End if over weight
                reward = 0
                done = True
    else:
        # End if item is unavailable
        reward = 0
        done = True
        
    return env.state, reward, done, {}


def render(env):
    total_value = 0
    total_weight = 0
    for i in env.collected_items :
        total_value += env.item_values[i]
        total_weight += env.item_weights[i]
    print(env.collected_items, total_value, total_weight)
    pass

def RESET(env):
    if env.randomize_params_on_reset:
        env.item_weights = np.random.randint(1, 100, size=env.N, dtype=np.int32)
        env.item_values = np.random.randint(0, 100, size=env.N, dtype=np.int32)
        env.item_limits = np.random.randint(1, 10, size=env.N, dtype=np.int32)
    else:
        env.item_limits = env.item_limits_init.copy()

    env.current_weight = 0
    update_state(env)
    return env.state

# Set seed for reproducibility
np.random.seed(42)

# Specify knapsack maximum weight
MAX_WEIGHT = 300


class NormalizingWrapper(ObservationWrapper):
    """Environment wrapper to divide observations by their maximum value"""

    def __init__(self, env: BoundedKnapsackEnv):
        
        # Perform default wrapper initialization
        super().__init__(env)
        self.N=env.N
        self.max_weight=env.max_weight
        
        
        self.observation_space = spaces.Box(0, 1, shape=(3+3*env.N,), dtype=np.float32)

    def observation(self, observation: np.array):
        
        # Convert observation to float to allow division and float output type
        observation = observation.astype(np.float32)

        # Normalize item weights
        observation[0, :-1] = observation[0, :-1] / np.max(observation[0, :-1])
        # Fix max weight input to 0
        observation[0, -1] = 0.0
        # Normalize item values
        observation[1, :-1] = observation[1, :-1] / np.max(observation[1, :-1])
        # Normalize current weight
        observation[1, -1] = observation[1, -1] / self.max_weight
        # Normalize item limits
        observation[2, :] = observation[2, :] / np.max(observation[2, :])

        # Concatenate three vectors emitted by default bounded knapsack environment
        observation = np.reshape(observation, (3+3*self.N,))

        return observation


def train_model(
    algorithm_class: Type[OnPolicyAlgorithm] = PPO,
    gamma: float = 0.99,
    learning_rate: float = 0.0003,
    normalize_env: bool = True,
    activation_fn: Type[nn.Module] = nn.ReLU,
    net_arch=[256, 256],
    total_timesteps: int = 100000,
    verbose: int = 1,
) -> OnPolicyAlgorithm:
    
    # Make environment and apply normalization wrapper if specified
    env = BoundedKnapsackEnv( mask=False)
    if normalize_env:
        env = NormalizingWrapper(env)

    # Initialize environment by resetting
    env.reset()
   
    # Model definition
    model = algorithm_class(
        policy="MlpPolicy",
        env=env,
        gamma=gamma,
        learning_rate=learning_rate,
        policy_kwargs=dict(
            activation_fn=activation_fn,
            net_arch=net_arch,
        ),
       
        verbose=verbose,
    )

    # Model training
    model.learn(
        total_timesteps=total_timesteps,
        
    )

    # Stop environment
    env.close()

    return model



model = train_model()


env = BoundedKnapsackEnv( mask=False)

combined = np.vstack((env.item_values, env.item_weights,env.item_limits))
env = NormalizingWrapper(env)
## convert your array into a dataframe
df = pd.DataFrame (combined)

## save to xlsx file

filepath = 'my_excel_file.xlsx'

df.to_excel(filepath, index=False)


mean_reward, std_reward = evaluate_policy(
        model=model, 
        env=env, 
        n_eval_episodes=100, 
        deterministic=False
    )
print(f"Mean reward over 100 episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

mean_reward, std_reward = evaluate_policy(
        model=model, 
        env=env, 
        n_eval_episodes=1000, 
        deterministic=False
    )
print(f"Mean reward over 1000 episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

render(env)

obs = RESET(env)
obs = np.reshape(obs, (18,))   
  
states=None
i=0
while i<10:
    action, states = model.predict(obs,states)
    obs, rewards, dones, info = STEP(env,action)
    print(rewards)
    obs = np.reshape(obs, (18,)) 
    i+=1
   

render(env)
