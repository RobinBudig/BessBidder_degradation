import os

import numpy as np
import pandas as pd
from loguru import logger
from pulp import GUROBI, PULP_CBC_CMD, LpMaximize, LpProblem, LpVariable, lpSum

import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine
import time

load_dotenv()

PRECOMPUTED_VWAP_PATH = os.path.join("data", "precomputed_vwaps")



PASSWORD = os.getenv("SQL_PASSWORD")
if PASSWORD:
    password_for_url = f":{PASSWORD}"
else:
    password_for_url = ""

THESIS_DB_NAME = os.getenv("POSTGRES_DB_NAME")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_DB_HOST = os.getenv("POSTGRES_DB_HOST")

CONNECTION = (
    f"postgres://{POSTGRES_USER}{password_for_url}@{POSTGRES_DB_HOST}/{THESIS_DB_NAME}"
)
CONNECTION_ALCHEMY = f"postgresql://{POSTGRES_USER}{password_for_url}@{POSTGRES_DB_HOST}/{THESIS_DB_NAME}"
conn = psycopg2.connect(CONNECTION)
conn_alchemy = create_engine(CONNECTION_ALCHEMY)
cursor = conn.cursor()
cursor.execute("ROLLBACK")


def load_vwap_matrix_for_day(current_day, base_path=PRECOMPUTED_VWAP_PATH):
    """
    Lädt die vorcomputierte VWAP-Matrix für einen Tag aus einer Parquet-Datei.

    Parquet-Format:
        - index: bucket_end (Execution-Fenster-Ende)
        - columns: deliverystart (Produkt)
        - Werte: VWAP (float / NaN)
    """
    fname = os.path.join(base_path, f"vwaps_{current_day:%Y-%m-%d}.parquet")

    if not os.path.exists(fname):
        raise FileNotFoundError(f"VWAP-Parquet für {current_day:%Y-%m-%d} nicht gefunden: {fname}")

    matrix = pd.read_parquet(fname)

    # Index / Columns wieder zu Timestamps machen
    matrix.index = pd.to_datetime(matrix.index)
    matrix.columns = pd.to_datetime(matrix.columns)

    # Falls noch keine Zeitzone dran ist, auf Europe/Berlin setzen
    if matrix.index.tz is None:
        matrix.index = matrix.index.tz_localize("Europe/Berlin")
    if matrix.columns.tz is None:
        matrix.columns = matrix.columns.tz_localize("Europe/Berlin")

    return matrix




def get_vwap_from_matrix_for_bucket(vwap_matrix, execution_time_end, end_date):
    """
    Holt aus der VWAP-Matrix die passende Zeile für ein bestimmtes execution_time_end
    und formt sie in dasselbe Format wie get_average_prices(...):

        DataFrame:
            index: 15-Minuten-Intervalle von start_of_day bis end_of_day
            Spalte: "price"

    end_date ist wie im Original: das Ende des Betrachtungszeitraums (trading_end),
    wird verwendet, um start_of_day / end_of_day zu bestimmen.
    """

    # start_of_day / end_of_day genauso wie in get_average_prices
    start_of_day = pd.to_datetime(end_date) - pd.Timedelta(hours=2)
    start_of_day = start_of_day.replace(hour=0, minute=0)
    end_of_day = start_of_day.replace(hour=23, minute=45)

    # Falls das Execution-Fenster nicht in der Matrix existiert:
    if execution_time_end not in vwap_matrix.index:
        idx = pd.date_range(start_of_day, end_of_day, freq="15min")
        return pd.DataFrame(index=idx, columns=["price"], data=np.nan)

    # Zeile für dieses execution_time_end holen
    row = vwap_matrix.loc[execution_time_end]       # Series: index = deliverystart, values = vwap

    # In DataFrame mit Spalte "price" verwandeln
    vwap_df = row.to_frame(name="price")

    # Sicherstellen, dass wir das vollständige 15-Minuten-Raster haben
    full_index = pd.date_range(start_of_day, end_of_day, freq="15min")
    vwap_df = vwap_df.reindex(full_index)

    return vwap_df




def get_prices_day(df, execution_time_start, day):
    # set start_of_day to day at 00:00:00
    start_of_day = pd.to_datetime(day)

    # set hour and minute to 0 (europe/berlin time)
    start_of_day = start_of_day.replace(hour=0, minute=0)

    end_of_day = start_of_day

    end_of_day = end_of_day.replace(hour=23, minute=45)

    filtered_df = df[df["execution_time_start"] == execution_time_start]

    # filter so product is <= end_of_day
    filtered_df = filtered_df[filtered_df["product"] <= end_of_day]

    # remove column execution_time_start
    filtered_df = filtered_df.drop(columns=["execution_time_start"])

    # set index to product
    filtered_df.set_index("product", inplace=True)

    # set index to be all 15 minute intervals from start_of_day to end_of_day, filling missing values with NaN
    filtered_df = filtered_df.reindex(
        pd.date_range(start_of_day, end_of_day, freq="15min")
    )

    return filtered_df


def calculate_discounted_price(price, current_time, delivery_time, discount_rate):
    time_difference = (
        delivery_time - current_time
    ).total_seconds() / 3600  # difference in hours

    if time_difference <= 1:  # if less than one hour, return the original price
        return price

    if price < 0:
        discount_factor = np.exp((discount_rate / 100) * time_difference)
    else:
        discount_factor = np.exp(-(discount_rate / 100) * time_difference)

    return price * discount_factor


def run_optimization_quarterhours_repositioning(
    prices_qh,
    execution_time,
    cap,
    c_rate,
    roundtrip_eff,
    max_cycles,
    discount_rate,
    prev_net_trades=pd.DataFrame(
        columns=["sum_buy", "sum_sell", "net_buy", "net_sell", "product"]
    ),
):
    # copy prices_qh
    prices_qh_adj = prices_qh.copy()

    # loop through prices_qh and adjust prices
    for i in prices_qh_adj.index:
        if not pd.isna(prices_qh_adj.loc[i, "price"]):
            prices_qh_adj.loc[i, "price"] = calculate_discounted_price(
                prices_qh_adj.loc[i, "price"], execution_time, i, discount_rate
            )

            # round prices to 2 decimals
            prices_qh_adj.loc[i, "price"] = round(prices_qh_adj.loc[i, "price"], 2)

    # copy prices_qh
    prices_qh_adj_buy = prices_qh.copy()

    # loop through prices_qh and adjust prices
    for i in prices_qh_adj_buy.index:
        if not pd.isna(prices_qh_adj_buy.loc[i, "price"]):
            prices_qh_adj_buy.loc[i, "price"] = calculate_discounted_price(
                prices_qh_adj_buy.loc[i, "price"], execution_time, i, -discount_rate
            )

            # round prices to 2 decimals
            prices_qh_adj_buy.loc[i, "price"] = round(
                prices_qh_adj_buy.loc[i, "price"], 2
            )

    prices_qh["price"] = round(prices_qh["price"], 2)

    # # merhe prices_qh_adj to prices_qh with column name "price_adj"
    # prices_qh = pd.merge(prices_qh, prices_qh_adj, left_index=True, right_index=True, suffixes=('', '_adj'))

    # print(prices_qh)

    # Create the 'battery' model
    m_battery = LpProblem("battery", LpMaximize)

    # Create variables using the DataFrame's index
    current_buy_qh = LpVariable.dicts("current_buy_qh", prices_qh.index, lowBound=0)
    current_sell_qh = LpVariable.dicts("current_sell_qh", prices_qh.index, lowBound=0)
    battery_soc = LpVariable.dicts("battery_soc", prices_qh.index, lowBound=0)

    # Create net variables
    net_buy = LpVariable.dicts("net_buy", prices_qh.index, lowBound=0)
    net_sell = LpVariable.dicts("net_sell", prices_qh.index, lowBound=0)
    charge_sign = LpVariable.dicts("charge_sign", prices_qh.index, cat="Binary")

    # Introduce auxiliary variables
    z = LpVariable.dicts("z", prices_qh.index, lowBound=0)
    w = LpVariable.dicts("w", prices_qh.index, lowBound=0)

    M = 100

    e = 0.01

    efficiency = roundtrip_eff**0.5

    # Objective function
    # Adjusted objective component for cases where previous trades < e
    adjusted_obj = [
        (
            (
                current_sell_qh[i]
                * (
                    prices_qh_adj.loc[i, "price"]
                    - 0.1 / 2  # assumed transaction costs per trade
                    - e
                )
                # * efficiency  # monetary efficiency consideration
            )
            - (
                current_buy_qh[i]
                * (
                    prices_qh_adj_buy.loc[i, "price"]
                    + 0.1 / 2  # assumed transaction costs per trade
                    + e
                )
                
                #/ efficiency  # monetary efficiency consideration
            )
        )
        * 1.0
        / 4.0
        for i in prices_qh.index
        if not pd.isna(prices_qh.loc[i, "price"])
        and (
            prev_net_trades.loc[i, "net_buy"] < e
            and prev_net_trades.loc[i, "net_sell"] < e
        )
    ]

    # Original objective component for cases where previous trades >= e
    original_obj = [
        (
            current_sell_qh[i] * (prices_qh.loc[i, "price"] - e)
            - current_buy_qh[i] * (prices_qh.loc[i, "price"] + e)
        )
        * 1.0
        / 4.0
        for i in prices_qh.index
        if not pd.isna(prices_qh.loc[i, "price"])
        and (
            prev_net_trades.loc[i, "net_buy"] >= e
            or prev_net_trades.loc[i, "net_sell"] >= e
        )
    ]

    # Combine and set the objective
    m_battery += lpSum(original_obj + adjusted_obj)

    # Constraints
    previous_index = prices_qh.index[0]

    for i in prices_qh.index[1:]:
        m_battery += (
            battery_soc[i]
            == battery_soc[previous_index]
            + net_buy[previous_index] * efficiency *1.0 / 4.0
            - net_sell[previous_index] * 1.0 / 4.0 / efficiency,
            f"BatteryBalance_{i}",
        )
        previous_index = i

    m_battery += battery_soc[prices_qh.index[0]] == 0, "InitialBatterySOC"

    for i in prices_qh.index:
        # Handling NaN values by setting buy and sell quantities to 0
        if pd.isna(prices_qh.loc[i, "price"]):
            m_battery += current_buy_qh[i] == 0, f"NaNBuy_{i}"
            m_battery += current_sell_qh[i] == 0, f"NaNSell_{i}"
        else:
            m_battery += battery_soc[i] <= cap, f"Cap_{i}"
            m_battery += net_buy[i] <= cap * c_rate, f"BuyRate_{i}"
            m_battery += net_sell[i] <= cap * c_rate, f"SellRate_{i}"
            m_battery += (
                net_sell[i] * 1.0 / efficiency  / 4.0 <= battery_soc[i],
                f"SellVsSOC_{i}",
            )

        # big M constraints for net buy and sell
        m_battery += net_buy[i] <= M * charge_sign[i], f"NetBuyBigM_{i}"
        m_battery += net_sell[i] <= M * (1 - charge_sign[i]), f"NetSellBigM_{i}"

        m_battery += z[i] <= charge_sign[i] * M, f"ZUpper_{i}"
        m_battery += z[i] <= net_buy[i], f"ZNetBuy_{i}"
        m_battery += z[i] >= net_buy[i] - (1 - charge_sign[i]) * M, f"ZLower_{i}"
        m_battery += z[i] >= 0, f"ZNonNeg_{i}"

        m_battery += w[i] <= (1 - charge_sign[i]) * M, f"WUpper_{i}"
        m_battery += w[i] <= net_sell[i], f"WNetSell_{i}"
        m_battery += w[i] >= net_sell[i] - charge_sign[i] * M, f"WLower_{i}"
        m_battery += w[i] >= 0, f"WNonNeg_{i}"

        m_battery += (
            z[i] - w[i]
            == current_buy_qh[i]
            + prev_net_trades.loc[i, "net_buy"]
            - current_sell_qh[i]
            - prev_net_trades.loc[i, "net_sell"],
            f"Netting_{i}",
        )

    # set efficiency as sqrt of roundtrip efficiency
    m_battery += (
        lpSum(net_buy[i] * efficiency * 1.0 / 4.0 for i in prices_qh.index) <= max_cycles * cap,
        "MaxCycles",
    )

    # Solve the problem
    m_battery.solve(GUROBI(msg=0))

    # Solve the problem
    #m_battery.solve(PULP_CBC_CMD(msg=0))

    # print(f"Status: {LpStatus[m_battery.status]}")
    # print(f"Objective value: {m_battery.objective.value()}")

    results = pd.DataFrame(
        columns=["current_buy_qh", "current_sell_qh", "battery_soc"],
        index=prices_qh.index,
    )

    trades = pd.DataFrame(
        columns=["execution_time", "side", "quantity", "price", "product", "profit"]
    )

    for i in prices_qh.index:
        if current_buy_qh[i].value() and current_buy_qh[i].value() > 0:
            # create buy trade
            new_trade = {
                "execution_time": [execution_time],
                "side": ["buy"],
                "quantity": [current_buy_qh[i].value()],
                "price": [prices_qh.loc[i, "price"]],
                "product": [i],
                "profit": [-current_buy_qh[i].value() * prices_qh.loc[i, "price"] / 4],
            }

            # append new trade using concat
            trades = pd.concat([trades, pd.DataFrame(new_trade)], ignore_index=True)

        if current_sell_qh[i].value() and current_sell_qh[i].value() > 0:
            # create sell trade
            new_trade = {
                "execution_time": [execution_time],
                "side": ["sell"],
                "quantity": [current_sell_qh[i].value()],
                "price": [prices_qh.loc[i, "price"]],
                "product": [i],
                "profit": [current_sell_qh[i].value() * prices_qh.loc[i, "price"] / 4],
            }

            # append new trade using concat
            trades = pd.concat([trades, pd.DataFrame(new_trade)], ignore_index=True)

    for i in prices_qh.index:
        results.loc[i, "current_buy_qh"] = current_buy_qh[i].value()
        results.loc[i, "current_sell_qh"] = current_sell_qh[i].value()
        results.loc[i, "net_buy"] = net_buy[i].value()
        results.loc[i, "net_sell"] = net_sell[i].value()
        results.loc[i, "charge_sign"] = charge_sign[i].value()
        results.loc[i, "battery_soc"] = battery_soc[i].value()

    return results, trades, m_battery.objective.value()


def get_net_trades(trades, end_date):
    # create a new empty dataframe with the columns "net_buy" and "net_sell"
    net_trades = pd.DataFrame(
        columns=["sum_buy", "sum_sell", "net_buy", "net_sell", "product"]
    )

    # based on trades, calculate the net buy and net sell for each product
    for product in trades["product"].unique():
        product_trades = trades[trades["product"] == product]
        sum_buy = product_trades[product_trades["side"] == "buy"]["quantity"].sum()
        sum_sell = product_trades[product_trades["side"] == "sell"]["quantity"].sum()
        # add to net_trades using concat
        net_trades = pd.concat(
            [
                net_trades,
                pd.DataFrame(
                    [[sum_buy, sum_sell, product]],
                    columns=["sum_buy", "sum_sell", "product"],
                ),
            ],
            ignore_index=True,
        )

    # add the columns "net_buy" and "net_sell" to net_trades, net_buy = sum_buy - sum_sell (if > 0), net_sell = sum_sell - sum_buy (if > 0)
    net_trades["net_buy"] = net_trades["sum_buy"] - net_trades["sum_sell"]
    net_trades["net_sell"] = net_trades["sum_sell"] - net_trades["sum_buy"]

    # remove values < 0 for net_buy and net_sell
    net_trades.loc[net_trades["net_buy"] < 0, "net_buy"] = 0
    net_trades.loc[net_trades["net_sell"] < 0, "net_sell"] = 0

    # set column product to index
    net_trades = net_trades.set_index("product")

    # set start_of_day to end_date minus 1 day
    start_of_day = pd.to_datetime(end_date) - pd.Timedelta(hours=2)

    # set hour and minute to 0 (europe/berlin time)
    start_of_day = start_of_day.replace(hour=0, minute=0)
    end_of_day = start_of_day
    end_of_day = end_of_day.replace(hour=23, minute=45)

    net_trades = net_trades.reindex(
        pd.date_range(start_of_day, end_of_day, freq="15min")
    )

    # fill NaN values with 0
    net_trades = net_trades.fillna(0)

    # set index to datetime
    net_trades.index = pd.to_datetime(net_trades.index)

    # return the net_trades dataframe
    return net_trades


def simulate_period(
    start_day,
    end_day,
    discount_rate,
    bucket_size,
    c_rate,
    roundtrip_eff,
    max_cycles,
    min_trades,
    precomputed_vwap_path=PRECOMPUTED_VWAP_PATH
):
    log_message = (
        "Running Rolling intrinsic QH with the following parameters:\n"
        "Start Day: {start_day}\n"
        "End Day: {end_day}\n"
        "Discount Rate: {discount_rate}\n"
        "Bucket Size: {bucket_size}\n"
        "C Rate: {c_rate}\n"
        "Roundtrip Efficiency: {roundtrip_eff}\n"
        "Max Cycles: {max_cycles}\n"
        "Min Trades: {min_trades}"
    ).format(
        start_day=start_day,
        end_day=end_day,
        discount_rate=discount_rate,
        bucket_size=bucket_size,
        c_rate=c_rate,
        roundtrip_eff=roundtrip_eff,
        max_cycles=max_cycles,
        min_trades=min_trades,
    )

    logger.info(log_message)
    year = start_day.year

    # set path as ./ma_results/threshold
    path = os.path.join(
        "output",
        "single_market",
        "rolling_intrinsic",
        "ri_basic",
        "qh",
        str(year),
        "bs"
        + str(bucket_size)
        + "cr"
        + str(c_rate)
        + "rto"
        + str(roundtrip_eff)
        + "mc"
        + str(max_cycles)
        + "mt"
        + str(min_trades),
    )
    tradepath = os.path.join(path, "trades")
    vwappath = os.path.join(path, "vwap")

    # create directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(tradepath):
        os.makedirs(tradepath)

    if not os.path.exists(vwappath):
        os.makedirs(vwappath)

    profitpath = os.path.join(path, "profit.csv")
    # check if profits.csv exists in path
    if os.path.exists(profitpath):
        # read profits.csv
        profits = pd.read_csv(profitpath)
    else:
        # create profits.csv
        profits = pd.DataFrame(columns=["day", "profit", "cycles"])

    if len(profits) > 0:
        # set current_day to last "day" in profits.csv
        current_day = (
            pd.Timestamp(profits.iloc[-1]["day"], tz="Europe/Berlin")
            + pd.Timedelta(days=1)
            + pd.Timedelta(hours=2)
        )
        # set current_cycles to last "cycles" in profits.csv
        current_cycles = profits.iloc[-1]["cycles"]
    else:
        current_day = start_day
        current_cycles = 0

    net_trades = pd.DataFrame(
        columns=["sum_buy", "sum_sell", "net_buy", "net_sell", "product"]
    )

    while current_day < end_day:
        current_day = current_day.replace(hour=0, minute=0, second=0, microsecond=0)

        print("current_day: ", current_day)

        db_time_total = 0.0
        solver_time_total = 0.0
        vwap_csv_time_total = 0.0  

        all_trades = pd.DataFrame(
            columns=["execution_time", "side", "quantity", "price", "product", "profit"]
        )

        # set trading_start to current_day minus 3 hours
        trading_start = current_day - pd.Timedelta(hours=8)
        # set trading_end to current_day plus 1 day
        trading_end = current_day + pd.Timedelta(days=1)

        print("trading_start: ", trading_start)
        print("trading_end: ", trading_end)

        try:
            vwap_matrix = load_vwap_matrix_for_day(current_day, base_path=precomputed_vwap_path)
        except FileNotFoundError as e:
            logger.warning(f"Keine vorcomputierten VWAPs für {current_day:%Y-%m-%d}: {e}")
            # Tag überspringen und mit nächstem Tag weitermachen
            current_day = current_day + pd.Timedelta(days=1) + pd.Timedelta(hours=2)
            continue

        # set execution_time_start to trading_start
        execution_time_start = trading_start
        # set execution_time_end to trading_start plus 15 minutes
        execution_time_end = trading_start + pd.Timedelta(minutes=bucket_size)

        # calculate number of days until end_day
        days_left = (end_day - current_day).days
        days_done = (current_day - start_day).days

        # allowed_cycles = max_cycles / 365 + (
        #    (max_cycles / 365 * (365 - days_left)) - current_cycles
        # )

        allowed_cycles = 1 + max(0, days_done - current_cycles)

        # allowed_cycles = (500 - current_cycles) / days_left

        print("Days left: ", days_left)
        print("Current cycles: ", current_cycles)
        print("Allowed cycles: ", allowed_cycles)

        while execution_time_end < trading_end:
            # -----------------------------------------
            # VWAP aus vorcomputierter Matrix holen
            # -----------------------------------------
            t0 = time.perf_counter()
            vwap = get_vwap_from_matrix_for_bucket(
                vwap_matrix,
                execution_time_end=execution_time_end,
                end_date=trading_end,
            )
            db_time_total += time.perf_counter() - t0  # "DB"-Zeit ist jetzt Datei-/Lookup-Zeit

            # --- Optional: weiterhin VWAP-CSV pro Tag schreiben, wie bisher ---
            vwaps_for_logging = (
                vwap.copy().rename(columns={"price": execution_time_end}).T
            )
            vwap_filename = os.path.join(
                vwappath, "vwaps_" + current_day.strftime("%Y-%m-%d") + ".csv"
            )

            t_csv0 = time.perf_counter()
            if not os.path.exists(vwap_filename):
                vwaps_for_logging.to_csv(
                    vwap_filename,
                    mode="a",
                    header=True,
                    index=True,
                )
            elif os.path.exists(vwap_filename) and (execution_time_start == trading_start):
                os.remove(vwap_filename)
                vwaps_for_logging.to_csv(
                    vwap_filename,
                    mode="a",
                    header=False,
                    index=True,
                )
            else:
                vwaps_for_logging.to_csv(
                    vwap_filename,
                    mode="a",
                    header=False,
                    index=True,
                )
            vwap_csv_time_total += time.perf_counter() - t_csv0
            # --- Ende CSV-Logging ---

            net_trades = get_net_trades(all_trades, trading_end)

            # if all vwap["price"] are NaN
            if vwap["price"].isnull().all():
                execution_time_start = execution_time_end
                execution_time_end = execution_time_start + pd.Timedelta(
                    minutes=bucket_size
                )
                continue
            else:
                try:
                    t_s0 = time.perf_counter()
                    results, trades, profit = run_optimization_quarterhours_repositioning(
                        vwap,
                        execution_time_start,
                        1,
                        c_rate,
                        roundtrip_eff,
                        allowed_cycles,
                        discount_rate,
                        net_trades,
                    )
                    solver_time_total += time.perf_counter() - t_s0

                    # append trades to all_trades using concat
                    all_trades = pd.concat([all_trades, trades])
                except ValueError:
                    print("Error in optimization")
                    print("execution_time_start: ", execution_time_start)
                    execution_time_start = execution_time_end
                    execution_time_end = execution_time_start + pd.Timedelta(
                        minutes=bucket_size
                    )
                    continue

            execution_time_start = execution_time_end
            execution_time_end = execution_time_start + pd.Timedelta(minutes=bucket_size)


        # calculate daily_profit as sum of all_trades["profit"]
        daily_profit = all_trades["profit"].sum()

        current_cycles += net_trades["net_buy"].sum() / 4.0 * roundtrip_eff**0.5

        # save trades
        all_trades.to_csv(
            os.path.join(
                tradepath, "trades_" + current_day.strftime("%Y-%m-%d") + ".csv"
            ),
            index=False,
        )

        # append daily_profit to profits.csv using concat
        profits = pd.concat(
            [
                profits,
                pd.DataFrame(
                    [[current_day, daily_profit, current_cycles]],
                    columns=["day", "profit", "cycles"],
                ),
            ]
        )

        # save profits.csv
        profits.to_csv(os.path.join(path, "profit.csv"), index=False)
        
        logger.info(f"DB-total time:       {db_time_total:.2f} s")
        logger.info(f"Solver total time:   {solver_time_total:.2f} s")
        logger.info(f"VWAP-CSV total time:      {vwap_csv_time_total:.2f} s")
        logger.info(f"simulate_one_day: {current_day}")

        # set current day to current_day plus 1 day
        current_day = current_day + pd.Timedelta(days=1) + pd.Timedelta(hours=2)
