from datetime import datetime

import pandas as pd

from rye_flex_env.env import RyeFlexEnv
from rye_flex_env.plotter import RyeFlexEnvEpisodePlotter


def test_episodes():
    """
    Test to check that length of episode,
    cumulative reward and done signal are sent correctly
    """

    data = pd.read_csv("data/train.csv", index_col=0, parse_dates=True)

    env = RyeFlexEnv(data=data)
    plotter = RyeFlexEnvEpisodePlotter()
    length = int(env._episode_length.days * 24)

    # Example with random initial state
    done = False
    cumulative_reward = env._cumulative_reward

    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        new_cumulative_reward = info["cumulative_reward"]

        assert round(new_cumulative_reward - cumulative_reward, 5) == round(reward, 5)

        cumulative_reward = new_cumulative_reward
        plotter.update(info)

    assert len(plotter._states) == length

    plotter.plot_episode(show=False)

    # Example where environment are set to partial known state
    env.reset(start_time=datetime(2020, 2, 3), battery_storage=1)

    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        plotter.update(info)

    assert len(plotter._states) == length
    plotter.plot_episode(show=False)


if __name__ == "__main__":
    test_episodes()
