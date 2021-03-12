"""
The test script that will be used for evaluation of the "agents" performance
"""
from datetime import datetime
from os.path import abspath, dirname, join

import gym
import numpy as np
import pandas as pd

from rye_flex_env.env import RyeFlexEnv
from rye_flex_env.plotter import RyeFlexEnvEpisodePlotter


class RandomActionAgent:
    def __init__(self, action_space: gym.spaces.Box):
        self._action_space = action_space

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Normally one would take state as input, and select action based on this.
        Since we are taking random action here, knowing the stat is not necessary.
        """

        action: np.ndarray = self._action_space.sample()
        return action


def main() -> None:
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)

    env = RyeFlexEnv(data=data)
    env.reset(start_time=datetime(2021, 2, 1, 0, 0))
    plotter = RyeFlexEnvEpisodePlotter()

    # INSERT YOUR OWN ALGORITHM HERE
    agent = RandomActionAgent(env.action_space)

    # Example with random initial state
    info = {}
    done = False
    # Initial state
    state = env._state

    while not done:

        # INSERT YOUR OWN ALGORITHM HERE
        action = agent.get_action(state)

        state, reward, done, info = env.step(action)

        plotter.update(info)

    print(f"Your test score is: {info['cumulative_reward']} NOK")

    plotter.plot_episode()


if __name__ == "__main__":
    main()
