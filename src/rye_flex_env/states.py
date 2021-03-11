from dataclasses import dataclass

import numpy as np


@dataclass
class Action:
    """The action vector.

    Args:
        charge_battery [kWh/h]: If value is:
            - Positive -> store energy in battery.
            - Negative -> discharged energy from battery and use in microgrid.
        charge_hydrogen [kWh/h]: If value is:
            - Positive -> store power in hydrogen.
            - Negative -> discharged from battery and use in microgrid.
    """

    charge_battery: float
    charge_hydrogen: float

    @property
    def vector(self) -> np.ndarray:
        return np.array(
            [
                self.charge_battery,
                self.charge_hydrogen,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(cls, action: np.ndarray) -> "Action":
        return cls(
            charge_battery=action.item(0),
            charge_hydrogen=action.item(1),
        )


@dataclass
class State:
    """The state vector.

    Args:
        consumption [kWh/h]: Consumption in the microgrid.
        wind_production [kWh/h]: Power production from solar used in the microgrid.
        pv_production [kWh/h]: Power production from solar wind used in the microgrid.
        battery_storage [kWh]: Potential energy stored in the battery.
        hydrogen_storage [kWh]: Potential energy stored in the hydrogen.
        grid_import [kWh/h]: Power imported from the grid to the microgrid.
        grid_import_peak [kWh/h]: Peak power imported from the grid to the microgrid.
        spot_market_price [NOK/kWh]: The spot market price for Trondheim.
    """

    consumption: float
    pv_production: float
    wind_production: float
    battery_storage: float
    hydrogen_storage: float
    grid_import: float
    grid_import_peak: float
    spot_market_price: float

    @property
    def vector(self) -> np.ndarray:
        return np.array(
            [
                self.consumption,
                self.pv_production,
                self.wind_production,
                self.battery_storage,
                self.hydrogen_storage,
                self.grid_import,
                self.grid_import_peak,
                self.spot_market_price,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(cls, state: np.ndarray) -> "State":
        return cls(
            consumption=state.item(0),
            pv_production=state.item(1),
            wind_production=state.item(2),
            battery_storage=state.item(3),
            hydrogen_storage=state.item(4),
            grid_import=state.item(5),
            grid_import_peak=state.item(6),
            spot_market_price=state.item(7),
        )
