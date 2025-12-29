
import os
import pandas as pd
from dotenv import load_dotenv
import pyomo.environ as pyo
import importlib.util

from src.single_market.day_ahead_market_optimizer import DayAheadMarketOptimizationModel
from src.shared.config import (
    START,
    END,
    DATA_PATH_DA,
    MAX_CYCLES_LIFETIME,
    LIFETIME_YEARS,
    BATTERY_CAPACITY,
    EFFICIENCY,
    START_END_SOC,
    C_RATE,
)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# Battery parameters from 02a_single_market_day_ahead_milp.py
BATTERY_CAPACITY = BATTERY_CAPACITY
CHARGE_RATE = C_RATE * BATTERY_CAPACITY  # in MW
DISCHARGE_RATE = C_RATE * BATTERY_CAPACITY  # in MW
EFFICIENCY = EFFICIENCY
MAX_CYCLES = MAX_CYCLES_LIFETIME
START_END_SOC = START_END_SOC

# Battery degradation parameters from config.py
MAX_CYCLES_LIFETIME = float(MAX_CYCLES_LIFETIME)
LIFETIME_YEARS = float(LIFETIME_YEARS)

# --- COU calibration (subgradient method) ---
MAX_ITERATIONS = 30
ALLOWED_CYCLES_ERROR = 0.5  # stop when within this many cycles of the target



def load_data():

    data_path = os.path.join(project_root, DATA_PATH_DA)
    da_auction_prices = pd.read_csv(data_path, index_col=0, parse_dates=True)[
        ["epex_spot_60min_de_lu_eur_per_mwh", "exaa_15min_de_lu_eur_per_mwh"]
    ]
    da_auction_prices.index = da_auction_prices.index.tz_convert("Europe/Berlin")

    return {
        "da_prices_forecast": da_auction_prices["exaa_15min_de_lu_eur_per_mwh"],
        "da_prices": da_auction_prices["epex_spot_60min_de_lu_eur_per_mwh"],
    }

def main():

    # Get daily timestamps
    days = [
        t.date().isoformat()
        for t in pd.date_range(START, END - pd.Timedelta(days=1), freq="1d")
    ]
    # limit days to START and END from config.py
    days = [day for day in days if
            (pd.Timestamp(day).date() >= START.date()) and (pd.Timestamp(day).date() < END.date())]

    data = load_data()

    # Calculate target number of cycles for the analyzed time horizon
    # based on LIFETIME_YEARS_YEARS and MAX_CYCLES_LIFETIME
    horizon_years = (END - START).days / 365.25
    target_cycles = MAX_CYCLES_LIFETIME * (horizon_years / LIFETIME_YEARS)

    #Log COU iterations for transparency
    cou_history = []
    def log_cou_iteration(iteration, cou, cycles, target_cycles, profit):
        cou_history.append({
            "iteration": iteration,
            "cou": float(cou),
            "cycles": float(cycles),
            "target_cycles": float(target_cycles),
            "mismatch": float(cycles - target_cycles),
            "market_profit": float(profit),
        })

    def run_dispatch(cost_of_use: float):

        results = []
        cycles_used = 0.0

        for day in days:
            model = DayAheadMarketOptimizationModel(
                time_index=data["da_prices"].loc[day].index,
                da_prices_forecast=data["da_prices_forecast"].loc[day].values,
                da_prices=data["da_prices"].loc[day].values,
                battery_capacity=BATTERY_CAPACITY,
                charge_rate=CHARGE_RATE,
                discharge_rate=DISCHARGE_RATE,
                efficiency=EFFICIENCY,
                max_cycles=target_cycles,
                start_end_soc=START_END_SOC,
                cost_of_use=cost_of_use,
                cycles_used_init=cycles_used,
            )
            temp_results = model.solve()
            cycles_used += float(pyo.value(model.model.delta_cycles))
            results.append(temp_results)

        df = pd.concat(results).sort_index()
        df["profit"] = df["discharge_revenues"] + df["charge_costs"] #Excludes COU because of indirect calculation
        return float(df["profit"].sum() ), float(cycles_used)

    #initialize cou with a high value e.g. 100 EUR per FEC
    # COU is initialized high, thus creating a situation where too few cycles are used
    cou = float(100)

    #Continously optimize and decrease COU until target_cycles == cycles_used within deviation of ALLOWED_CYCLES_ERROR
    for k in range(MAX_ITERATIONS):
        profit_sum, cycles_used = run_dispatch(cou)
        mismatch = cycles_used - target_cycles  # >0 means too many cycles

        log_cou_iteration(
            iteration=k,
            cou=cou,
            cycles=cycles_used,
            target_cycles=target_cycles,
            profit=profit_sum,
        )

        if abs(mismatch) <= ALLOWED_CYCLES_ERROR:
            break

        # COU is decreased as long as too few cycles are used
        if mismatch >= -0.01 * target_cycles:
            cou = max(0.0, cou / 1.01)  # decrease very slightly if close to target cycles
        elif mismatch >= -0.1 * target_cycles:
            cou = max(0.0, cou / 1.05)
        elif mismatch >= -0.2 * target_cycles:
            cou = max(0.0, cou / 1.1)
        elif mismatch >= -0.5 * target_cycles:
            cou = max(0.0, cou / 1.5)  # decrease slightly if close to target cycles
        else:
            cou = max(0.0, cou/2) #decrease heavily if far away from target cycles


    print(f"\nFinal COU: {cou:.6f} EUR/FEC")
    df_cou = pd.DataFrame(cou_history)
    print(df_cou)
    return cou

def get_optimal_cou():
    optimal_cou = main()
    return optimal_cou
OPTIMAL_COU = get_optimal_cou()

if __name__ == "__main__":
    main()


