"""
Day-Ahead Market MILP Optimisation Script (Single Market Case)

This script:
- Loads EXAA (forecast) and EPEX Spot (realised) day-ahead price data.
- Solves a daily MILP optimisation problem for battery dispatch.
- Aggregates results for all days in the time range defined in `src/shared/config.py`.
- Saves results to a CSV file.

Requires:
- Data in `DATA_PATH_DA`
- Configuration constants from `src/shared/config.py`
- MILP solver backend available via Pyomo
"""

import os
import pandas as pd
from dotenv import load_dotenv
import pyomo.environ as pyo
from src.shared.calculate_cost_of_use import get_optimal_cou
from src.single_market.day_ahead_market_optimizer import DayAheadMarketOptimizationModel
from src.shared.config import (
    C_RATE,
    MAX_CYCLES_LIFETIME,
    START,
    END,
    OUTPUT_DIR_DA,
    FILENAME_DA,
    DATA_PATH_DA,
    BATTERY_CAPACITY,
    EFFICIENCY,
    START_END_SOC,

)

load_dotenv()

# Output configuration
OUTPUT_DIR = OUTPUT_DIR_DA
FILENAME = FILENAME_DA

# Battery parameters from config.py
BATTERY_CAPACITY = BATTERY_CAPACITY
CHARGE_RATE = C_RATE * BATTERY_CAPACITY  # in MW
DISCHARGE_RATE = C_RATE * BATTERY_CAPACITY  # in MW
EFFICIENCY = EFFICIENCY
MAX_CYCLES = MAX_CYCLES_LIFETIME
START_END_SOC = START_END_SOC

cost_of_use = get_optimal_cou() # in EUR/FEC


def load_data():
    """
    Load EXAA and EPEX Spot day-ahead auction prices from CSV.

    Returns:
        dict: Dictionary with keys:
            - "da_prices_forecast": EXAA 15-minute prices (forecast)
            - "da_prices": EPEX Spot 60-minute prices (realised)
    """
    data_path = DATA_PATH_DA
    da_auction_prices = pd.read_csv(data_path, index_col=0, parse_dates=True)[
        ["epex_spot_60min_de_lu_eur_per_mwh", "exaa_15min_de_lu_eur_per_mwh"]
    ]
    da_auction_prices.index = da_auction_prices.index.tz_convert("Europe/Berlin")

    return {
        "da_prices_forecast": da_auction_prices["exaa_15min_de_lu_eur_per_mwh"],
        "da_prices": da_auction_prices["epex_spot_60min_de_lu_eur_per_mwh"],
    }


def main():
    """
    Solve daily MILP battery optimisation problems for all days in the simulation range.
    Results are saved to disk as a single CSV file.
    """

    # Get daily timestamps
    days = [
        t.date().isoformat()
        for t in pd.date_range(START, END - pd.Timedelta(days=1), freq="1d")
    ]
    
    # limit days to START and END from config.py
    days = [day for day in days if (pd.Timestamp(day).date() >= START.date()) and (pd.Timestamp(day).date() < END.date())]

    data = load_data()

    results = []
    cycles_used = 0

    for day in days:
        model = DayAheadMarketOptimizationModel(
            time_index=data["da_prices"].loc[day].index,
            da_prices_forecast=data["da_prices_forecast"].loc[day].values,
            da_prices=data["da_prices"].loc[day].values,
            battery_capacity=BATTERY_CAPACITY,
            charge_rate=CHARGE_RATE,
            discharge_rate=DISCHARGE_RATE,
            efficiency=EFFICIENCY,
            max_cycles=MAX_CYCLES,
            start_end_soc=START_END_SOC,
            cost_of_use=cost_of_use,
            cycles_used_init=cycles_used,
        )
        temp_results = model.solve()
        cycles_used += pyo.value(model.model.delta_cycles)
        results.append(temp_results)

    final_df = pd.concat(results).sort_index()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    final_df.to_csv(os.path.join(OUTPUT_DIR, FILENAME))

    final_df["profit"] = final_df["discharge_revenues"] + final_df["charge_costs"]

    cumulative_profit = final_df["profit"].sum()
    print(f"Total Revenue:{cumulative_profit}")


if __name__ == "__main__":
    main()
