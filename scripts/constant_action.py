"""
Example of an  agent interacting with the environment for two episodes,
where for the first episode the environment have a random initial state,
and in the second episode, the initial state is partially defined.

The agent is a simple (and naive agent) and only selects constant actions.

"""

from datetime import datetime
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd

from rye_flex_env.env import RyeFlexEnv
from rye_flex_env.plotter import RyeFlexEnvEpisodePlotter


class ConstantActionAgent:
    """
    An agent which always returns a constant action
    """

    def get_action(self) -> np.ndarray:
        """
        Normally one would take the state as input, and select action based on this.
        Since we are taking random action here, knowing the stat is not necessary.
        """

        """
        First entry = battery charge/discharge
        and second entry = hydrogen charge/discharge
        """
        return np.array([1, 0])


def main() -> None:
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)

    env = RyeFlexEnv(data=data)
    plotter = RyeFlexEnvEpisodePlotter()
    agent = ConstantActionAgent()

    # Example with random initial state
    info = {}
    done = False
    # Initial state
    state = env._state

    while not done:

        action = agent.get_action()

        state, reward, done, info = env.step(action)

        plotter.update(info)

    print(f"Your score is: {info['cumulative_reward']} NOK")
    plotter.plot_episode()

    # Example where environment are reset
    env.reset(start_time=datetime(2020, 2, 3), battery_storage=1)

    done = False
    while not done:
        action = agent.get_action()

        state, reward, done, info = env.step(action)

        plotter.update(info)

    print(f"Your score is: {info['cumulative_reward']} NOK")
    plotter.plot_episode()


if __name__ == "__main__":
    main()
