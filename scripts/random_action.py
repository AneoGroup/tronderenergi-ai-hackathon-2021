"""
Example of an  agent interacting with the environment for one episode,
where the episode starts with a random initial state.

The agent is simple (and naive) and selects random actions.

"""

import gym
import numpy as np
import pandas as pd

from rye_flex_env.env import RyeFlexEnv
from rye_flex_env.plotter import RyeFlexEnvEpisodePlotter


class RandomActionAgent:
    def __init__(self, action_space: gym.spaces.Box):
        self._action_space = action_space

    def get_action(self) -> np.ndarray:
        """
        Normally one would take state as input, and select action based on this.
        Since we are taking random action here, knowing the stat is not necessary.
        """

        return self._action_space.sample()


if __name__ == "__main__":

    data = pd.read_csv("data/train.csv", index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)

    agent = RandomActionAgent(action_space=env.action_space)
    plotter = RyeFlexEnvEpisodePlotter()
    info = None
    done = False

    while not done:
        action = agent.get_action()
        state, reward, done, info = env.step(action)
        plotter.update(info)

    print(f"Your score is: {info['cumulative_reward']}")
    plotter.plot_episode()
