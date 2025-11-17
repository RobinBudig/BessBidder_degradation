import warnings
import pandas as pd

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
#from src.single_market.rolling_intrinsic_new import simulate_period
from src.single_market.rolling_intrinsic_gurobi import simulate_period

load_dotenv()

warnings.simplefilter(action="ignore", category=FutureWarning)

if __name__ == "__main__":

    simulate_period(
        START,
        END,
        discount_rate=0,
        bucket_size=BUCKET_SIZE,
        c_rate=C_RATE,
        roundtrip_eff=RTE,
        max_cycles=MAX_CYCLES_PER_YEAR,
        min_trades=MIN_TRADES,
    )
    
"""
if __name__ == "__main__":
    # Beispielaufruf:
    start_day = pd.Timestamp("2020-01-01", tz="Europe/Berlin")
    end_day   = pd.Timestamp("2020-01-11", tz="Europe/Berlin")  # 10 Tage

    run_period_parallel(
        start_day=start_day,
        end_day=end_day,
        discount_rate=0.0,
        bucket_size=15,
        c_rate=0.5,
        roundtrip_eff=0.86,
        max_cycles=365,
        min_trades=10,
        base_output_path="output_parallel",
        processes=8,   # Ryzen 7 -> 8 Kerne
    )

"""