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


# Set seed for reproducibility
np.random.seed(42)

# Specify knapsack maximum weight
MAX_WEIGHT = 300

# Custom wrapper to normalize environment observations
##########################################################################################
class NormalizingWrapper(ObservationWrapper):
    """Environment wrapper to divide observations by their maximum value"""

    def __init__(self, env: BoundedKnapsackEnv):
        """Change observation space of wrapped environment

        Args:
            env (BoundedKnapsackEnv): environment to wrap (tailed towards or_gym)
        """
        # Perform default wrapper initialization
        super().__init__(env)
        self.N=env.N
        self.max_weight=env.max_weight
        # Change default observation space to concatenate the three vectors from the
        # default or_gym implementation, and allow for floating point values in the
        # range [0, 1].
        
        self.observation_space = spaces.Box(0, 1, shape=(3+3*env.N,), dtype=np.float32)

    def observation(self, observation: np.array):
        """Perform postprocessing on observations emitted by wrapped environment

        Args:
            observation (np.array): observation emitted by knapsack environment

        Returns:
            np.array: transformed observation
        """
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
    

# Training function
##########################################################################################
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
    """Train model with logging and checkpointing

    Args:
        algorithm_class (Type[OnPolicyAlgorithm], optional): algorithm class to use.
            Defaults to PPO.
        gamma (float, optional): discount factor to use.
            Defaults to 0.99.
        learning_rate (float, optional): learning rate to use.
            Defaults to 0.0003.
        normalize_env (bool, optional): whether to normalize the observation space.
            Defaults to True.
        activation_fn (Type[nn.Module], optional): activation function to use.
            Defaults to nn.ReLU.
        net_arch (list, optional): shared layer sizes for MLPPolicy.
            Defaults to [256, 256].
        total_timesteps (int, optional): total timesteps to train for.
            Defaults to 150000.
        verbose (int, optional): whether to do extensive logging.
            Defaults to 1.

    Returns:
        OnPolicyAlgorithm: trained model
    """
    # Make environment and apply normalization wrapper if specified
    env: BoundedKnapsackEnv = or_gym.make(
        "Knapsack-v2", max_weight=MAX_WEIGHT, mask=False
    )
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
model.save("ppo_model")