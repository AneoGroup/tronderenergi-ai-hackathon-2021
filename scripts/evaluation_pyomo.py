"""
The test script that will be used for evaluation of the "agents" performance
"""
from datetime import datetime, timedelta
from os.path import abspath, dirname, join

import gym
import numpy as np
import pandas as pd

from rye_flex_env.env import RyeFlexEnv
from rye_flex_env.plotter import RyeFlexEnvEpisodePlotter
from rye_flex_env.states import State, Action

from pyomo.environ import *

import matplotlib.pyplot as plt


class PyomoAgent:
    """
    An example agent which always returns a constant action
    """
    def __init__(self):

        model = AbstractModel()

        model.periods = Set()

        # Constants

        ########### Cost

        model.PeakCost = Param(default=49)

        ########### Battery

        model.MaxCapacityBattery = Param(default=500)
        model.MaxDischargeBattery = Param(default=400)
        model.MaxChargeBattery = Param(default=400)
        model.RoundEfficiencyBattery = Param(default=0.85)
        model.SoCBatteryStart = Param(default=0)

        ########### Hydrogen

        model.MaxCapacityHydrogen = Param(default=1670)
        model.MaxDischargeHydrogen = Param(default=100)
        model.MaxChargeHydrogen = Param(default=55)
        model.RoundEfficiencyHydrogen = Param(default=0.325)
        model.SoCHydrogenStart = Param(default=0)

        # Parameter

        model.Cost = Param(model.periods)
        model.Demand = Param(model.periods)
        model.Solar = Param(model.periods)
        model.Wind = Param(model.periods)

        # Variables

        ############# Grid

        model.gridImport = Var(model.periods, within=NonNegativeReals)
        model.maxEl = Var(within=NonNegativeReals)

        ############# Battery

        model.soCBattery = Var(model.periods, within=NonNegativeReals)
        model.dischargeBattery = Var(model.periods, within=NonNegativeReals)
        model.chargeBattery = Var(model.periods, within=NonNegativeReals)

        ############# Hydrogen

        model.soCHydrogen = Var(model.periods, within=NonNegativeReals)
        model.dischargeHydrogen = Var(model.periods, within=NonNegativeReals)
        model.chargeHydrogen = Var(model.periods, within=NonNegativeReals)

        ###### Obj function #############

        def obj_function(model):
            return sum(
                model.gridImport[t] * (model.Cost[t] + 0.05) for t in model.periods) + (
                               model.PeakCost * model.maxEl)

        model.obj_function = Objective(rule=obj_function, sense=minimize)

        ###### Peak #############

        def maxEle(model, t):
            return model.gridImport[t] <= model.maxEl

        model.MaxEl = Constraint(model.periods, rule=maxEle)

        ###### Energy balanse #############

        def EnergyBalance(model, t):
            return model.gridImport[t] + model.dischargeBattery[t] + model.Solar[t] + \
                   model.Wind[t] + model.dischargeHydrogen[t] >= model.Demand[t] + \
                   model.chargeBattery[t] + model.chargeHydrogen[t]

        model.EnergyBalance_Con = Constraint(model.periods, rule=EnergyBalance)

        ###### Battery #############

        def BatterySoC(model, t):
            if t == 1:
                return model.soCBattery[t] == model.SoCBatteryStart + \
                       model.chargeBattery[t] * model.RoundEfficiencyBattery - \
                       model.dischargeBattery[t]
            else:
                return model.soCBattery[t] == model.soCBattery[t - 1] + \
                       model.chargeBattery[t] * model.RoundEfficiencyBattery - \
                       model.dischargeBattery[t]

        model.BatterySoC_Con = Constraint(model.periods, rule=BatterySoC)

        def BatteryCharge(model, t):
            return model.chargeBattery[t] <= model.MaxChargeBattery

        model.BatteryCharge_Con = Constraint(model.periods, rule=BatteryCharge)

        def BatteryDischarge(model, t):
            return model.dischargeBattery[t] <= model.MaxDischargeBattery

        model.BatteryDischarge_Con = Constraint(model.periods, rule=BatteryDischarge)

        ###### Hydrogen Tank #############

        def HydrogenSoC(model, t):
            if t == 1:
                return model.soCHydrogen[t] == model.SoCHydrogenStart + \
                       model.chargeHydrogen[t] * model.RoundEfficiencyHydrogen - \
                       model.dischargeHydrogen[t]
            else:
                return model.soCHydrogen[t] == model.soCHydrogen[t - 1] + \
                       model.chargeHydrogen[t] * model.RoundEfficiencyHydrogen - \
                       model.dischargeHydrogen[t]

        model.HydrogenSoC_Con = Constraint(model.periods, rule=HydrogenSoC)

        def HydrogenCharge(model, t):
            return model.chargeHydrogen[t] <= model.MaxChargeHydrogen

        model.Hydrogencharge_Con = Constraint(model.periods, rule=HydrogenCharge)

        def HydrogenDischarge(model, t):
            return model.dischargeHydrogen[t] <= model.MaxDischargeHydrogen

        model.Hydrogendischarge_con = Constraint(model.periods, rule=HydrogenDischarge)

        def SoCBatMax(model, t):
            return model.soCBattery[t] <= model.MaxCapacityBattery

        model.MaxBattSoC_Con = Constraint(model.periods, rule=SoCBatMax)

        def SoCHydrogenMax(model, t):
            return model.soCHydrogen[t] <= model.MaxCapacityHydrogen

        model.MaxHydrogenSoC_Con = Constraint(model.periods, rule=SoCHydrogenMax)



        data = DataPortal()
        data.load(filename='data/pyomo_test.csv', select=(
            'time', "pv_production", 'wind_production', 'consumption', 'spot_market_price'),
            param=(model.Solar, model.Wind, model.Demand, model.Cost),
            index=model.periods
        )

        instance = model.create_instance(data)  # load parameters
        solver = SolverFactory("glpk")  # Free version
        results = solver.solve(instance, tee=True)
        instance.solutions.load_from(results)  # Loading solution into instance

        time = pd.read_csv("data/test.csv", index_col=0, parse_dates=True).index
        self.states = pd.DataFrame()
        self.states["consumption"]= pd.Series(instance.Demand.extract_values(),index = time)
        self.states["wind_production"] = pd.Series(instance.Wind.extract_values(),index = time)
        self.states["pv_production"] = pd.Series(instance.Solar.extract_values(),index = time)
        self.states["battery_storage"] = pd.Series(instance.soCBattery.get_values(),index = time)
        self.states["hydrogen_storage"] = pd.Series(instance.soCHydrogen.get_values(),index = time)
        self.states["grid_import"] = pd.Series(instance.gridImport.get_values(),index = time)
        self.states["spot_market_price"] = pd.Series(instance.Cost.extract_values(),index = time)

        discharge_battery = pd.DataFrame.from_dict(instance.dischargeBattery.get_values(), orient='index')[0]
        charge_battery = pd.DataFrame.from_dict(instance.chargeBattery.get_values(), orient='index')[0]

        self.charge_battery = pd.DataFrame( (charge_battery - discharge_battery).values, index = time)[0]

        discharge_hydrogen = pd.DataFrame.from_dict(instance.dischargeHydrogen.get_values(), orient='index')[0]
        charge_hydrogen = pd.DataFrame.from_dict(instance.chargeHydrogen.get_values(), orient='index')[0]

        self.charge_hydrogen = pd.DataFrame((charge_hydrogen - discharge_hydrogen).values, index=time)[0]

        self.actions = pd.DataFrame()

        self.actions["charge_battery"] = self.charge_battery
        self.actions["charge_hydrogen"] = self.charge_hydrogen





    def get_action(self, time:datetime) -> np.ndarray:
        """
        Normally one would take the state as input, and select action based on this.
        Since we are taking random action here, knowing the stat is not necessary.
        """

        action = Action(
            charge_battery=self.charge_battery.loc[time + timedelta(hours = 1)],
            charge_hydrogen=self.charge_hydrogen.loc[time + timedelta(hours = 1)]
        )
        return action.vector



def main() -> None:
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/test.csv"), index_col=0, parse_dates=True)

    env = RyeFlexEnv(data=data)
    plotter = RyeFlexEnvEpisodePlotter()

    # Reset episode to feb 2021, and get initial state
    state = env.reset(start_time=datetime(2021, 2, 1, 0, 0))

    # INSERT YOUR OWN ALGORITHM HERE
    agent = PyomoAgent()

    info = {}
    done = False

    while not done:

        # INSERT YOUR OWN ALGORITHM HERE
        action = agent.get_action(env._time)
        print(action)

        state, reward, done, info = env.step(action)

        plotter.update(info)

    print(f"Your test score is: {info['cumulative_reward']} NOK")
    plotter.plot_episode( )


if __name__ == "__main__":
    main()
