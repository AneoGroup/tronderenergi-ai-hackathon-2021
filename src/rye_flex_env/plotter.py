from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Union, cast

import matplotlib.pyplot as plt
import pandas as pd

from .states import Action, State


@dataclass
class RyeFlexEnvEpisodePlotter:
    """A tool for plotting the states, actions and rewards for an episode.

    Args:
        states: List of states from one episode.
        actions: List of actions from one episode.
        times: List of time-steps from one episode.
        rewards: List of rewards from one episode (both reward and cumulative reward).
    """

    _states: List[Dict[str, float]]
    _actions: List[Dict[str, float]]
    _times: List[datetime]
    _rewards: List[Dict[str, float]]

    def __init__(self) -> None:
        self.reset()

    def update(self, info: Dict[str, Union[State, Action, float, datetime]]) -> None:
        """Update list of states, actions, times and rewards.

        Args:
            info: Info dictionary from the output from env.step(action).
        """
        self._states.append(info["state"].__dict__)
        self._actions.append(info["action"].__dict__)
        self._times.append(cast(datetime, info["time"]))
        self._rewards.append(
            {
                "reward": cast(float, info["reward"]),
                "cumulative_reward": cast(float, info["cumulative_reward"]),
            }
        )

    def plot_episode(self, show: bool = True) -> None:
        """Plot states, rewards and actions from the episode, and there prepare for the
        next episode (reset).

        Args:
            show: Boolean if the plot should be shown.
        """
        _states = pd.DataFrame(self._states, index=self._times)
        _actions = pd.DataFrame(self._actions, index=self._times)
        _reward = pd.DataFrame(self._rewards, index=self._times)

        _states.plot(subplots=True, title="States")

        _actions.plot(subplots=True, title="Actions")

        _reward.plot(subplots=True, title="Rewards")

        if show:
            plt.show()
        self.reset()

    def reset(self) -> None:
        """Reset the list of states, actions, times and rewards."""
        self._states = []
        self._actions = []
        self._times = []
        self._rewards = []
