"""
Example of an  agent interacting with the environment for two episodes,
where for the first episode the environment have a random initial state,
and in the second episode, the initial state is partially defined.

The agent is a simple (and naive agent) selecting constant
 actions based on the total production.

"""

from datetime import datetime
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd

from rye_flex_env.env import RyeFlexEnv
from rye_flex_env.plotter import RyeFlexEnvEpisodePlotter
from rye_flex_env.states import State, Action


class SimpleStateBasedAgent:
    """
    An agent which always returns a constant action
    """

    def get_action(self, state_vector: np.ndarray) -> np.ndarray:
        """
        Normally one would take the state as input, and select action based on this.
        Since we are taking random action here, knowing the stat is not necessary.
        """

        # Convert from numpy array to State:
        state = State.from_vector(state_vector)

        # Create a state for total production:
        total_production = state.pv_production + state.wind_production

        if total_production > 30:
            # Charging battery with 10 kWh/h and hydrogen with 0 kWh/h
            action = Action(charge_battery=10, charge_hydrogen=0)
            return action.vector
        else:
            # Charging battery with 0 kWh/h and hydrogen with 10 kWh/h
            return np.array([0, 10])


def main() -> None:
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)

    env = RyeFlexEnv(data=data)
    plotter = RyeFlexEnvEpisodePlotter()
    agent = SimpleStateBasedAgent()

    # Get initial state
    state = env.get_state_vector()
    info = {}
    done = False

    while not done:
        action = agent.get_action(state)

        state, reward, done, info = env.step(action)

        plotter.update(info)

    print(f"Your score is: {info['cumulative_reward']} NOK")
    plotter.plot_episode()


if __name__ == "__main__":
    main()
