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

from src.single_market.day_ahead_market_optimizer import DayAheadMarketOptimizationModel
from src.shared.config import (
    C_RATE,
    MAX_CYCLES_LIFETIME,
    RTE,
    START,
    END,
    OUTPUT_DIR_DA,
    FILENAME_DA,
    DATA_PATH_DA,
)

load_dotenv()

# Output configuration
OUTPUT_DIR = OUTPUT_DIR_DA
FILENAME = FILENAME_DA

# Battery parameters
cost_of_use = 0 # in EUR/FEC
BATTERY_CAPACITY = 1  # in MWh
CHARGE_RATE = C_RATE * BATTERY_CAPACITY  # in MW
DISCHARGE_RATE = C_RATE * BATTERY_CAPACITY  # in MW
EFFICIENCY = RTE**0.5
MAX_CYCLES = MAX_CYCLES_LIFETIME
START_END_SOC = 0.0


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

    data = load_data()

    results = []

    remaining_cycles = float(MAX_CYCLES)

    for day in days:
        model = DayAheadMarketOptimizationModel(
            time_index=data["da_prices"].loc[day].index,
            da_prices_forecast=data["da_prices_forecast"].loc[day].values,
            da_prices=data["da_prices"].loc[day].values,
            battery_capacity=BATTERY_CAPACITY,
            charge_rate=CHARGE_RATE,
            discharge_rate=DISCHARGE_RATE,
            efficiency=EFFICIENCY,
            max_cycles=max(0.0, remaining_cycles),
            start_end_soc=START_END_SOC,
            cost_of_use=cost_of_use,
        )
        temp_results = model.solve()

        # --- update remaining lifetime budget ---
        if ("volume_charge" in temp_results.columns) and ("volume_discharge" in temp_results.columns):
            used_throughput = float(temp_results["volume_charge"].sum() + temp_results["volume_discharge"].sum())
            used_cycles = used_throughput / (2 * BATTERY_CAPACITY)
            remaining_cycles -= used_cycles
        else:
            print("WARNING: volume_charge/volume_discharge not found for budget update. Columns:", list(temp_results.columns))

        results.append(temp_results)

        # early stop if lifetime budget is depleted
        if remaining_cycles <= 1e-9:
            break
    print(f"remaining lifetime cycles after simulation: {remaining_cycles:.4f}")

    final_df = pd.concat(results).sort_index()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    final_df.to_csv(os.path.join(OUTPUT_DIR, FILENAME))


if __name__ == "__main__":
    main()
