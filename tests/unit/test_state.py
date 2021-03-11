from rye_flex_env.states import Action, State


def test_action():
    """
    Test that action are init correctly
    """
    input_charge_battery = 400
    input_charge_hydrogen = 55
    res_action = Action(
        charge_battery=input_charge_battery,
        charge_hydrogen=input_charge_hydrogen,
    )

    assert res_action.charge_battery == input_charge_battery
    assert res_action.charge_hydrogen == input_charge_hydrogen

    assert (Action.from_vector(res_action.vector).vector == res_action.vector).all()


def test_states():
    """
    Test that state are init correctly
    """

    consumption = 11
    wind_production = 6
    pv_production = 6
    battery_storage = 500
    hydrogen_storage = 1670
    grid_import = 1000000
    grid_import_peak = 100000
    spot_market_price = 100000

    res_state = State(
        consumption=consumption,
        wind_production=wind_production,
        pv_production=pv_production,
        battery_storage=battery_storage,
        hydrogen_storage=hydrogen_storage,
        grid_import=grid_import,
        grid_import_peak=grid_import_peak,
        spot_market_price=spot_market_price,
    )

    assert res_state.consumption == consumption
    assert res_state.pv_production == pv_production
    assert res_state.wind_production == wind_production
    assert res_state.battery_storage == battery_storage
    assert res_state.hydrogen_storage == hydrogen_storage
    assert res_state.grid_import == grid_import
    assert res_state.grid_import_peak == grid_import_peak
    assert res_state.spot_market_price == spot_market_price

    assert (State.from_vector(res_state.vector).vector == res_state.vector).all()
