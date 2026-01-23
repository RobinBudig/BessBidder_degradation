"""
Myopic Multi-Market Simulation Script

This script:
- Loads results from the day-ahead MILP optimisation (single market).
- Simulates myopic rolling intrinsic bidding on the intraday market.
- Uses intelligent stacking of quarter-hourly products.
- Writes simulation logs and outputs to a versioned logging path.

Requires:
- Day-ahead results from MILP (CSV)
- Configuration from `src/shared/config.py`
- Simulation functions for quarter-hourly stacking
"""

import os
import pandas as pd
from loguru import logger

from src.coordinated_multi_market.rolling_intrinsic.testing_rolling_intrinsic_qh_intelligent_stacking import (
    simulate_days_stacked_quarterhourly_products,
)


from src.shared.config import (
    BUCKET_SIZE,
    C_RATE,
    INPUT_DIR_DA,
    LOGGING_PATH_MYOPIC,
    MAX_CYCLES_PER_YEAR,
    MIN_TRADES,
    RTE, START, END
)
from src.shared.calculate_cost_of_use import OPTIMAL_COU
COU = OPTIMAL_COU
if __name__ == "__main__":
    # Path where output logs and results will be stored
    versioned_log_path = LOGGING_PATH_MYOPIC

    # Load day-ahead optimisation results
    df_milp = pd.read_csv(INPUT_DIR_DA)
    df_milp["time"] = pd.to_datetime(df_milp["time"]).dt.tz_convert("Europe/Berlin")

    # Construct RI output path dynamically based on configuration
    ri_qh_output_path = os.path.join(
        versioned_log_path,
        "rolling_intrinsic_stacked_on_day_ahead_qh",
        f"bs{BUCKET_SIZE}cr{C_RATE}rto{RTE}mc{MAX_CYCLES_PER_YEAR}mt{MIN_TRADES}",
    )

    # Run the rolling intrinsic simulation with intelligent stacking (QH products)
    simulate_days_stacked_quarterhourly_products(
        da_bids_path=INPUT_DIR_DA,
        output_path=ri_qh_output_path,
        start_day=START,
        end_day=END,
        discount_rate=0,
        #bucket_size=BUCKET_SIZE,
        c_rate=C_RATE,
        roundtrip_eff=RTE,
        max_cycles=10e6,
        min_trades=MIN_TRADES,
        cou = COU
    )

    logger.info(
        "Finished calculating intelligently stacked rolling intrinsic revenues with quarterhourly products."
    )

#Further Results:

path = f"output/myopic_multi_market/rolling_intrinsic_stacked_on_day_ahead_qh/bs15cr1rto0.86mc365mt10/profit.csv"
profits = os.path.join(path)
df = pd.read_csv(profits)
total_profit = df["profit"].sum()
path2 = f"output/single_market/day_ahead_milp/day_ahead_milp_results_2019-04-01_to_2020-03-27.csv"
df2 = pd.read_csv(path2)
df2["profit"] = df2["discharge_revenues"] + df2["charge_costs"]
total_profit += df2["profit"].sum()

#Calculate cycles by energy quantity traded --> Virtual + Real
total_cycles = 0

end = END.normalize() + pd.Timedelta(days=-1)
start = START.normalize()
for date in pd.date_range(start,end, freq="D"):

    date_str = date.strftime("%Y-%m-%d")

    trades = os.path.join(path, f"output/myopic_multi_market/rolling_intrinsic_stacked_on_day_ahead_qh/bs15cr1rto0.86mc365mt10/trades/trades_{date_str}.csv")
    df = pd.read_csv(trades)

    if not os.path.exists(trades):
        print(f"No trades for {date_str}")
        continue
    total_cycles += (df["quantity"].sum()/8) #/4 because of 15 min and /2 because of cap
print(f"Total cycles (virtual + real): {total_cycles:.2f}")

print(f"Total profit: {total_profit:.2f}")