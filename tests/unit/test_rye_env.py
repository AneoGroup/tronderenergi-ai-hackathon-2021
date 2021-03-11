from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from rye_flex_env.env import RyeFlexEnv
from rye_flex_env.states import Action, State


def test_attributes_time():
    """
    Test that all time attributes are handled correctly
    """

    data = pd.read_csv("data/train.csv", index_col=0, parse_dates=True)

    input_episode_length = timedelta(days=30)
    env = RyeFlexEnv(episode_length=input_episode_length, data=data)

    res_start_time = env._time
    res_end_time = env._episode_end_time

    assert input_episode_length == env._episode_length
    assert res_start_time + input_episode_length == res_end_time
    assert env._start_time_data <= res_start_time
    assert res_start_time <= env._end_time_data

    input_start_time = datetime(2020, 10, 1)
    env.reset(start_time=input_start_time)
    res_end_time = input_episode_length + env._time
    ans_end_time = datetime(2020, 10, 31)

    assert ans_end_time == res_end_time

    _, _, _, res_info = env.step(action=np.array([1, 1]))
    res_time = env._time

    assert res_time == env._time_resolution + input_start_time
    assert res_info["time"] == res_time


def test_loss_and_reward_attributes():
    """
    Test that all loss and reward constants are set correctly
    """
    input_charge_loss_battery = 0.8
    input_charge_loss_hydrogen = 0.6
    input_energy_grid_tariff = 0.051
    input_peak_grid_tariff = 49.0

    data = pd.DataFrame(
        data={
            "consumption": [1, 2, 1, 3],
            "pv_production": [1, 2, 3, 4],
            "wind_production": [1, 2, 3, 4],
            "spot_market_price": [1, 2, 3, 4],
        },
        index=pd.date_range("2020-1-1T12:00", periods=4, freq="H"),
    )

    env = RyeFlexEnv(
        charge_loss_battery=input_charge_loss_battery,
        charge_loss_hydrogen=input_charge_loss_hydrogen,
        grid_tariff=input_energy_grid_tariff,
        peak_grid_tariff=input_peak_grid_tariff,
        data=data,
        episode_length=timedelta(hours=2),
    )

    assert env._charge_loss_battery_storage == input_charge_loss_battery
    assert env._charge_loss_hydrogen_storage == input_charge_loss_hydrogen
    assert env._grid_tariff == input_energy_grid_tariff
    assert env._peak_grid_tariff == input_peak_grid_tariff


def test_state_space_and_action_space():
    """
    Test that state space are set correctly
    """

    data = pd.DataFrame(
        data={
            "consumption": [1, 2, 1, 3],
            "pv_production": [1, 2, 3, 4],
            "wind_production": [1, 2, 3, 4],
            "spot_market_price": [1, 2, 3, 4],
        },
        index=pd.date_range("2020-1-1T12:00", periods=4, freq="H"),
    )

    env = RyeFlexEnv(data=data, episode_length=timedelta(hours=2))

    assert (env.observation_space.low == env._state_space_min.vector).all()
    assert (env.observation_space.high == env._state_space_max.vector).all()

    assert (env.action_space.low == env._action_space_min.vector).all()
    assert (env.action_space.high == env._action_space_max.vector).all()


def test_reset():
    """
    Test that the reset function works (time resetting is tested in another test above)
    """

    data = pd.DataFrame(
        data={
            "consumption": [1, 2, 1, 3],
            "pv_production": [1, 2, 3, 4],
            "wind_production": [1, 2, 3, 4],
            "spot_market_price": [1, 2, 3, 4],
        },
        index=pd.date_range("2020-1-1T12:00", periods=4, freq="H"),
    )

    env = RyeFlexEnv(data=data, episode_length=timedelta(hours=2))

    env._cumulative_reward = 1000

    battery_storage = 0.1
    hydrogen_storage = 0.2
    grid_import = 0.3

    res_new_state_vector = env.reset(
        battery_storage=battery_storage,
        hydrogen_storage=hydrogen_storage,
        grid_import=grid_import,
    )
    res_new_state = env._state

    assert env._cumulative_reward == 0

    assert (res_new_state_vector == res_new_state.vector).all()
    assert (res_new_state_vector <= env.observation_space.high).all()
    assert (res_new_state_vector >= env.observation_space.low).all()

    assert res_new_state.battery_storage == battery_storage
    assert res_new_state.hydrogen_storage == hydrogen_storage
    assert res_new_state.grid_import == grid_import


def test_step_random_state():
    """
    Test that:
        - The state changes (also when not using when not using reset)
        - The state are in the desired state-space/observation-space
    """
    data = pd.DataFrame(
        data={
            "consumption": [1, 2, 1, 3],
            "pv_production": [2, 2, 5, 4],
            "wind_production": [3, 2, 1, 4],
            "spot_market_price": [0.0, 0.4, 0.2, 0.1],
        },
        index=pd.date_range("2020-1-1T12:00", periods=4, freq="H"),
    )
    env = RyeFlexEnv(
        data=data, episode_length=timedelta(hours=2), charge_loss_hydrogen=0.5
    )

    input_action = Action(charge_battery=-1.1, charge_hydrogen=-1.2).vector

    old_state_vector = env._state.vector

    res_state_vector, res_reward, res_done, res_info = env.step(input_action)

    res_state = env._state

    assert (res_state_vector != old_state_vector).any()

    assert (res_state_vector == res_state.vector).all()
    assert (res_state_vector <= env.observation_space.high).all()
    assert (res_state_vector >= env.observation_space.low).all()
    assert (res_info["state"].vector == res_state.vector).all()


def test_step_not_import_from_grid():
    """
    Test where we charge hydrogen and battery, but do not import from grid
    due to producing enough power.
    """
    data = pd.DataFrame(
        data={
            "consumption": [1, 2, 1, 3],
            "pv_production": [2, 20, 5, 4],
            "wind_production": [3, 2, 1, 4],
            "spot_market_price": [0.0, 0.4, 0.2, 0.1],
        },
        index=pd.date_range("2020-1-1T12:00", periods=4, freq="H"),
    )
    env = RyeFlexEnv(
        data=data, episode_length=timedelta(hours=2), charge_loss_hydrogen=0.5
    )

    env.reset(start_time=datetime(2020, 1, 1, 12), hydrogen_storage=1668)

    input_action = Action(charge_hydrogen=10, charge_battery=10).vector

    old_state = env._state
    print(old_state)

    # Check that we set correct states from data
    assert old_state.consumption == 1
    assert old_state.pv_production == 2
    assert old_state.wind_production == 3
    assert old_state.spot_market_price == 0

    res_new_state_vector, res_reward, res_done, res_info = env.step(input_action)

    res_new_state = env._state
    res_action = res_info["action"]

    # Check that the state-vectors have correct value
    # and are within state space
    assert (res_new_state_vector != old_state.vector).any()
    assert (res_new_state_vector == res_new_state.vector).all()
    assert (res_new_state_vector <= env.observation_space.high).all()
    assert (res_new_state_vector >= env.observation_space.low).all()
    assert (res_new_state.vector == res_new_state.vector).all()
    assert (res_info["state"].vector == res_new_state.vector).all()

    """
    Explanation of states:
    - Battery storage is set to 8.5, since we charged by 10, and have transformation
        losses of 85%, and had an initial state of 0.
    - Hydrogen storage is set to 1670, since we have transformation loss of 50%,
        , and had initial state of 1668 (max = 1670).
    - Grid import is set to 0, since we have:
        load = 2(consumption) + 10 (hydrogen) + 10.0 (battery) = 22
        production= 20 (solar) + 2(wind) = 22,
        grid_import = load - consumption = 0,
        meaning we do not need to import from the grid.
    """

    ans_new_state = State(
        consumption=2,
        wind_production=2,
        pv_production=20,
        spot_market_price=0.4,
        battery_storage=8.5,
        hydrogen_storage=1670,
        grid_import=0,
        grid_import_peak=0,
    )
    print(res_new_state)
    assert (ans_new_state.vector == res_new_state.vector).all()

    # Check that actions are calculated correctly.
    # Since all actions where charging, the actions are the same
    ans_action_vector = Action(charge_battery=10, charge_hydrogen=10).vector
    print(res_action)
    assert (res_action.vector == ans_action_vector).all()

    # Check that the reward are the correct value
    assert (
        res_reward
        == (ans_new_state.spot_market_price + env._grid_tariff)
        * ans_new_state.grid_import
    )


def test_step_import_from_grid():
    """
    Test where we charge and discharge, but
     it does not meet the consumption demand, and need to import from grid.
    """
    data = pd.DataFrame(
        data={
            "consumption": [1, 2, 1, 3],
            "pv_production": [2, 12, 5, 4],
            "wind_production": [3, 2, 1, 4],
            "spot_market_price": [0.0, 0.4, 0.2, 0.1],
        },
        index=pd.date_range("2020-1-1T12:00", periods=4, freq="H"),
    )
    env = RyeFlexEnv(
        data=data, episode_length=timedelta(hours=2), charge_loss_hydrogen=0.5
    )

    env.reset(start_time=datetime(2020, 1, 1, 13), battery_storage=8)

    input_action = Action(charge_hydrogen=-2, charge_battery=10).vector

    old_state = env._state

    res_new_state_vector, res_reward, res_done, res_info = env.step(input_action)

    res_new_state = env._state
    res_action = res_info["action"]

    # Check that state-vectors have correct value and are within state space
    assert (res_new_state_vector != old_state.vector).any()
    assert (res_new_state_vector == res_new_state.vector).all()
    assert (res_new_state_vector <= env.observation_space.high).all()
    assert (res_new_state_vector >= env.observation_space.low).all()
    assert (res_info["state"].vector == res_new_state.vector).all()

    """
    Explanation of states:
    - We tried to discharge hydrogen, but since hydrogen_storage was 0,
        we could not discharge.
    - The battery storage  is increased by 8.5, compared to initial state of 8.
    - We need to import 3.5 from the grid since we have a load of
        1 (consumption) + 10 (battery) = 11  and production of 1 (wind) + 5 (pv) = 6
        , leading to grid_import = 11 - 6 = 5.
    - Grid_import_peak = 5,since initial peak = 0.
    """
    ans_new_state = State(
        consumption=1,
        wind_production=1,
        pv_production=5,
        spot_market_price=0.2,
        battery_storage=16.5,
        hydrogen_storage=0.0,
        grid_import=5,
        grid_import_peak=5,
    )
    print(res_new_state)
    assert (ans_new_state.vector == res_new_state.vector).all()

    # Check that actions are calculated correctly
    # charge_hydrogen = 0, since we could not discharge hydrogen due to being empty
    ans_action = Action(charge_hydrogen=0, charge_battery=10)
    print(res_action)
    assert (res_action.vector == ans_action.vector).all()

    assert (
        res_reward
        == (ans_new_state.spot_market_price + env._grid_tariff)
        * ans_new_state.grid_import
    )


def test_step_saturation():
    """
    Test where we only look at the saturation of the actions
    """
    data = pd.read_csv("data/train.csv", index_col=0, parse_dates=True)
    env = RyeFlexEnv(data, charge_loss_hydrogen=0.5)

    env.reset(start_time=datetime(2020, 1, 3), battery_storage=400)

    input_action = Action(
        charge_hydrogen=1000000000, charge_battery=-20000000000000000
    ).vector

    res_new_state_vector, res_reward, res_done, res_info = env.step(input_action)

    ans_action = Action(charge_battery=-400, charge_hydrogen=55)
    res_action = res_info["action"]
    print(res_action)

    assert (res_action.vector == ans_action.vector).all()

    assert (res_action.vector >= env.action_space.low).all()
    assert (res_action.vector <= env.action_space.high).all()


def test_step_new_grid_import_peak():
    """
    Test where we get a new grid import peak
    """
    data = pd.DataFrame(
        data={
            "consumption": [1, 2, 1, 3],
            "pv_production": [2, 12, 5, 4],
            "wind_production": [3, 2, 1, 4],
            "spot_market_price": [0.0, 0.4, 0.2, 0.1],
        },
        index=pd.date_range("2020-1-1T12:00", periods=4, freq="H"),
    )
    env = RyeFlexEnv(
        data=data, episode_length=timedelta(hours=2), charge_loss_hydrogen=0.5
    )
    input_grid_import = 1.0

    env.reset(
        start_time=datetime(2020, 1, 1, 13),
        battery_storage=8,
        grid_import=input_grid_import,
    )

    # Check that the correct grid peak was set
    input_action = np.array([-2, 10])
    assert env._state.grid_import_peak == input_grid_import

    res_new_state_vector, res_reward, res_done, res_info = env.step(input_action)

    res_new_state = res_info["state"]
    print(res_new_state)

    # Check that the new peak was set
    assert res_new_state.grid_import_peak == 3
    assert res_new_state.grid_import == 3

    assert (
        res_reward
        == (res_new_state.spot_market_price + env._grid_tariff)
        * res_new_state.grid_import
    )


def test_step_new_grid_import_peak_end_of_month():
    """
    Test where we get a new grid import peak, and since it is the
    "end of the month" = episode length,
    we also get reward for grid tariff
    """
    data = pd.DataFrame(
        data={
            "consumption": [1, 2, 2, 3],
            "pv_production": [2, 12, 5, 4],
            "wind_production": [3, 2, 1, 4],
            "spot_market_price": [0.0, 0.4, 0.2, 0.1],
        },
        index=pd.date_range("2020-1-1T12:00", periods=4, freq="H"),
    )
    env = RyeFlexEnv(
        data=data, episode_length=timedelta(hours=2), charge_loss_hydrogen=0.5
    )
    input_grid_import = 1.0

    env.reset(
        start_time=datetime(2020, 1, 1, 13),
        battery_storage=8,
        grid_import=input_grid_import,
    )

    # Check that the correct grid peak was set
    input_action = np.array([-2, 10])
    assert env._state.grid_import_peak == input_grid_import

    _, _, res_done, res_info = env.step(input_action)

    # Check that the grid import and peak
    assert res_info["state"].grid_import == 4
    assert res_info["state"].grid_import_peak == 4
    assert not res_done

    res_new_state_vector, res_reward, res_done, res_info = env.step(input_action)

    res_new_state = res_info["state"]
    print(res_new_state)

    # Check grid import and peak
    assert res_done
    assert res_new_state.grid_import_peak == 4
    assert res_new_state.grid_import == 3

    assert res_reward == (
        (res_new_state.spot_market_price + env._grid_tariff) * res_new_state.grid_import
        + env._peak_grid_tariff * res_new_state.grid_import_peak
    )


def test_get_possible_start_times():
    data = pd.DataFrame(
        data={
            "consumption": [1, 2, 2, 3],
            "pv_production": [2, 12, 5, 4],
            "wind_production": [3, 2, 1, 4],
            "spot_market_price": [0.0, 0.4, 0.2, 0.1],
        },
        index=pd.date_range("2020-1-1T12:00", periods=4, freq="H"),
    )
    env = RyeFlexEnv(
        data=data, episode_length=timedelta(hours=2), charge_loss_hydrogen=0.5
    )

    ans = [datetime(2020, 1, 1, 12), datetime(2020, 1, 1, 13)]

    res = env.get_possible_start_times()

    print(res)

    assert ans == res
