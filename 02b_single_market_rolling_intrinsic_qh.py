import warnings

from dotenv import load_dotenv

from src.shared.config import (
    BUCKET_SIZE,
    C_RATE,
    MAX_CYCLES_PER_YEAR,
    MIN_TRADES,
    RTE,
    START,
    END,
)

from src.single_market.rolling_intrinsic_gurobi_qh import simulate_period

load_dotenv()

warnings.simplefilter(action="ignore", category=FutureWarning)

from src.shared.calculate_cost_of_use import OPTIMAL_COU
COU = OPTIMAL_COU # in EUR/FEC

if __name__ == "__main__":
    simulate_period(
        START,
        END,
        discount_rate=0,
        c_rate=C_RATE,
        roundtrip_eff=RTE,
        max_cycles=10e6,   #So high, that it is non binding
        min_trades=MIN_TRADES,
        cou = COU,
    )

import pandas as pd
import os
path = f"/Users/robin/PycharmProjects/BessBidder_degradation/output/single_market/rolling_intrinsic/ri_basic/qh/2019/cr1rto0.86cou{COU}mt10"
profits = os.path.join(path, "profit.csv")
df = pd.read_csv(profits)
total_profit = df["profit"].sum()
cycles_a = df["cycles"].sum()
#print(f"Total cycles (real): {cycles_a:.2f}")

#Rückrechnung, wieviel gecycled wurde über die quanity --> Virtuell + real
total_cycles = 0

end = END.normalize() + pd.Timedelta(days=-1)
start = START.normalize()
for date in pd.date_range(start,end, freq="D"):

    date_str = date.strftime("%Y-%m-%d")

    trades = os.path.join(path, f"trades/trades_{date_str}.csv")
    df = pd.read_csv(trades)

    if not os.path.exists(trades):
        print(f"No trades for {date_str}")
        continue
    total_cycles += (df["quantity"].sum()/8) #/4 because of 15 min and /2 because of cap
print(f"Total cycles (virtual + real): {total_cycles:.2f}")

print(f"Total profit: {total_profit:.2f}")
