import os

import numpy as np
import pandas as pd
from loguru import logger
from pulp import GUROBI, PULP_CBC_CMD, LpMaximize, LpProblem, LpVariable, lpSum
import gurobipy as gp

import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine




load_dotenv()

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


def get_average_prices(
    cursor, side, execution_time_start, execution_time_end, end_date, min_trades=10
):
    # set start_of_day to end_date minus 1 day
    start_of_day = pd.to_datetime(end_date) - pd.Timedelta(hours=2)

    # set hour and minute to 0 (europe/berlin time)
    start_of_day = start_of_day.replace(hour=0, minute=0)
    end_of_day = start_of_day.replace(hour=23, minute=45)

    table_name = "transactions_intraday_de"

    # transform dates to work with str format
    execution_time_start_str = execution_time_start.strftime("%Y-%m-%d %H:%M:%S")
    execution_time_end_str = execution_time_end.strftime("%Y-%m-%d %H:%M:%S")
    start_of_day_str = start_of_day.strftime("%Y-%m-%d %H:%M:%S")
    end_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        f"""
        SELECT
            deliverystart,
            SUM(weighted_avg_price * volume) / SUM(volume) AS weighted_avg_price
        FROM
            {table_name}
        WHERE
            (executiontime BETWEEN '{execution_time_start_str}' AND '{execution_time_end_str}')
            AND side = '{side}'
            AND deliverystart >= '{start_of_day_str}'
            AND deliverystart < '{end_date_str}'
        GROUP BY
            deliverystart
        HAVING
            SUM(trade_count) >= {min_trades};
        """
    )

    result = cursor.fetchall()

    df = pd.DataFrame(result, columns=["product", "price"])
    df.set_index("product", inplace=True)

    df = df.reindex(pd.date_range(start_of_day, end_of_day, freq="15min"))
    return df



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


def adjust_prices_block(prices_qh: pd.DataFrame, execution_time: pd.Timestamp, discount_rate: float) -> pd.DataFrame:
    """
    Erzeugt zwei vektorisierte, diskontierte Preisspalten:
    - price_sell_adj: für Verkäufe (mit deinem Vorzeichen-Schema)
    - price_buy_adj:  für Käufe  (inverse Diskontierung)

    Logik entspricht deiner bisherigen Funktion, nur ohne Python-Loops.
    """
    out = prices_qh.copy()

    # Basis: price auf 2 Nachkommastellen runden (vektorisiert)
    if "price" not in out.columns:
        raise ValueError("prices_qh benötigt eine Spalte 'price'.")
    out["price"] = out["price"].round(2)

    # Index -> Stundenabstand zur execution_time (float)
    idx = out.index.to_numpy(dtype="datetime64[ns]")
    exec_ts = np.datetime64(execution_time.tz_convert(None), "ns")
    hours = (idx - exec_ts) / np.timedelta64(1, "h")  # ndarray[float]

    price = out["price"].to_numpy(dtype=float)
    is_nan = np.isnan(price)

    # Vorzeichenregel wie in deiner calculate_discounted_price:
    # Preis < 0 => +, sonst - im Exponenten
    sign = np.where(price < 0, +1.0, -1.0)

    # Diskontfaktor pro Zeile (NaNs erstmal egal, masken wir später raus)
    factor = np.exp((discount_rate / 100.0) * sign * hours)

    # Wenn <= 1h bis Lieferung: originaler Preis
    use_orig = hours <= 1.0

    # Verkauf (sell): Preis * factor, außer <=1h -> original
    price_sell_adj = np.where(use_orig, price, price * factor)

    # Kauf (buy): Preis / factor, außer <=1h -> original
    price_buy_adj = np.where(use_orig, price, price / factor)

    # NaNs sauber durchreichen
    price_sell_adj = np.where(is_nan, np.nan, price_sell_adj)
    price_buy_adj  = np.where(is_nan, np.nan, price_buy_adj)

    out["price_sell_adj"] = np.round(price_sell_adj, 2)
    out["price_buy_adj"]  = np.round(price_buy_adj,  2)
    return out

def get_net_trades(trades, end_date):

    if trades.empty:
        idx = pd.date_range(
            end_date.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=22),
            end_date.normalize() + pd.Timedelta(hours=21, minutes=45),
            freq="15min",
        )
        df = pd.DataFrame(0, index=idx,
            columns=["sum_buy","sum_sell","net_buy","net_sell"])
        return df

    grouped = trades.groupby(["product", "side"])["quantity"].sum().unstack(fill_value=0)

    grouped["net_buy"] = grouped.get("buy", 0) - grouped.get("sell", 0)
    grouped["net_sell"] = grouped.get("sell", 0) - grouped.get("buy", 0)

    # negative Werte auf 0
    grouped["net_buy"] = grouped["net_buy"].clip(lower=0)
    grouped["net_sell"] = grouped["net_sell"].clip(lower=0)

    # vollständiger Index für alle Quarterhours
    start = end_date - pd.Timedelta(hours=2)
    start = start.replace(hour=0, minute=0)
    end   = start.replace(hour=23, minute=45)

    idx = pd.date_range(start, end, freq="15min")

    return grouped.reindex(idx, fill_value=0)



def build_battery_model(T, cap, c_rate, roundtrip_eff, max_cycles):
    """
    Baut das Gurobi-Modell einmalig für einen Tages-Index T (15-Minuten-Produkte).
    Gibt zurück:
        - model: Gurobi-Modell
        - vars:  Dict mit allen Variablen
        - netting_constr: Dict {t -> Netting-Constraint}, um später RHS updaten zu können
    """
    efficiency = roundtrip_eff ** 0.5
    M = cap * c_rate

    m = gp.Model("battery_persistent")
    m.Params.OutputFlag = 0  # keine Solver-Logs
    m.Params.Threads = 1  # Nur einen Thread verwenden

    # Variablen
    current_buy_qh  = m.addVars(T, lb=0.0, name="current_buy_qh")
    current_sell_qh = m.addVars(T, lb=0.0, name="current_sell_qh")
    battery_soc     = m.addVars(T, lb=0.0, name="battery_soc")

    net_buy   = m.addVars(T, lb=0.0, name="net_buy")
    net_sell  = m.addVars(T, lb=0.0, name="net_sell")
    charge_sign = m.addVars(T, vtype=gp.GRB.BINARY, name="charge_sign")
    
    z = m.addVars(T, lb=0.0, name="z")
    w = m.addVars(T, lb=0.0, name="w")

    # SOC-Dynamik
    previous_index = T[0]
    for i in T[1:]:
        m.addConstr(
            battery_soc[i]
            == battery_soc[previous_index]
            + net_buy[previous_index] * efficiency / 4.0
            - net_sell[previous_index] / 4.0 / efficiency,
            name=f"BatteryBalance_{i}",
        )
        previous_index = i

    # initialer SOC
    m.addConstr(battery_soc[T[0]] == 0.0, name="InitialBatterySOC")

    # Kapazität / Raten / Big-M-Logik (unabhängig von Preisen)
    for i in T:
        m.addConstr(battery_soc[i] <= cap,        name=f"Cap_{i}")
        m.addConstr(net_buy[i]   <= cap * c_rate, name=f"BuyRate_{i}")
        m.addConstr(net_sell[i]  <= cap * c_rate, name=f"SellRate_{i}")
        m.addConstr(
            net_sell[i] / efficiency / 4.0 <= battery_soc[i],
            name=f"SellVsSOC_{i}",
        )

        m.addConstr(net_buy[i]  <= M * charge_sign[i],       name=f"NetBuyBigM_{i}")
        m.addConstr(net_sell[i] <= M * (1 - charge_sign[i]), name=f"NetSellBigM_{i}")

        m.addConstr(z[i] <= charge_sign[i] * M,                      name=f"ZUpper_{i}")
        m.addConstr(z[i] <= net_buy[i],                              name=f"ZNetBuy_{i}")
        m.addConstr(z[i] >= net_buy[i] - (1 - charge_sign[i]) * M,   name=f"ZLower_{i}")
        m.addConstr(z[i] >= 0.0,                                     name=f"ZNonNeg_{i}")

        m.addConstr(w[i] <= (1 - charge_sign[i]) * M,                name=f"WUpper_{i}")
        m.addConstr(w[i] <= net_sell[i],                             name=f"WNetSell_{i}")
        m.addConstr(w[i] >= net_sell[i] - charge_sign[i] * M,        name=f"WLower_{i}")
        m.addConstr(w[i] >= 0.0,                                     name=f"WNonNeg_{i}")

    # Netting-Constraints: so formulieren, dass RHS die "alten Trades" wird
    netting_constr = {}
    for i in T:
        # z[i] - w[i] - current_buy_qh[i] + current_sell_qh[i] = rhs
        # rhs wird später als (prev_net_buy - prev_net_sell) gesetzt
        c = m.addConstr(
            z[i] - w[i] - current_buy_qh[i] + current_sell_qh[i] == 0.0,
            name=f"Netting_{i}",
        )
        netting_constr[i] = c

    # Zyklenlimit
    m.addConstr(
        gp.quicksum(net_buy[i] * efficiency / 4.0 for i in T) <= max_cycles * cap,
        name="MaxCycles",
    )

    # Noch keine Zielfunktion – die setzen wir pro Bucket neu
    m.setObjective(0.0, gp.GRB.MAXIMIZE)

    vars = {
        "current_buy_qh": current_buy_qh,
        "current_sell_qh": current_sell_qh,
        "battery_soc": battery_soc,
        "net_buy": net_buy,
        "net_sell": net_sell,
        "charge_sign": charge_sign,
        "z": z,
        "w": w,
        "efficiency": efficiency,
    }

    m.update()
    return m, vars, netting_constr





def solve_bucket_with_model(
    model,
    vars,
    netting_constr,
    prices_qh,
    execution_time,
    discount_rate,
    prev_net_trades,
):
    """
    Nutzt ein vorher gebautes Modell und löst die Optimierung
    für einen bestimmten Bucket (mit neuen Preisen & prev_net_trades).
    """
    # Preise diskontieren
    prices_qh_adj_all = adjust_prices_block(prices_qh, execution_time, discount_rate)

    # prev_net_trades auf Index ausrichten
    prev_net_trades = prev_net_trades.reindex(prices_qh.index).fillna(0.0)

    e = 0.01
    efficiency = vars["efficiency"]
    T = list(prices_qh.index)

    current_buy_qh  = vars["current_buy_qh"]
    current_sell_qh = vars["current_sell_qh"]
    net_buy         = vars["net_buy"]
    net_sell        = vars["net_sell"]
    charge_sign     = vars["charge_sign"]

    # --- Zielfunktion neu aufbauen ---
    obj = gp.LinExpr()

    for i in T:
        price = prices_qh.loc[i, "price"]
        if pd.isna(price):
            # keine Trades bei NaN: Bounds später auf 0 setzen
            continue

        prev_nb = prev_net_trades.loc[i, "net_buy"]
        prev_ns = prev_net_trades.loc[i, "net_sell"]

        price_sell_adj = prices_qh_adj_all.loc[i, "price_sell_adj"]
        price_buy_adj  = prices_qh_adj_all.loc[i, "price_buy_adj"]

        if prev_nb < e and prev_ns < e:
            term = (
                current_sell_qh[i] * (price_sell_adj - 0.1 / 2 - e)
                - current_buy_qh[i] * (price_buy_adj + 0.1 / 2 + e)
            ) / 4.0
        else:
            term = (
                current_sell_qh[i] * (price_sell_adj - e)
                - current_buy_qh[i] * (price_buy_adj + e)
            ) / 4.0

        obj += term

    model.setObjective(obj, gp.GRB.MAXIMIZE)

    # --- Netting-RHS updaten ---
    for i in T:
        prev_nb = prev_net_trades.loc[i, "net_buy"]
        prev_ns = prev_net_trades.loc[i, "net_sell"]
        rhs = prev_nb - prev_ns
        netting_constr[i].RHS = rhs

    # --- NaN-Preise: Trades verbieten (Bounds anpassen) ---
    for i in T:
        price = prices_qh.loc[i, "price"]
        if pd.isna(price):
            current_buy_qh[i].UB  = 0.0
            current_sell_qh[i].UB = 0.0
        else:
            current_buy_qh[i].UB  = gp.GRB.INFINITY
            current_sell_qh[i].UB = gp.GRB.INFINITY

    model.update()
    model.optimize()

    # --- Ergebnisse einsammeln ---
    results = pd.DataFrame(
        columns=["current_buy_qh", "current_sell_qh", "battery_soc"],
        index=prices_qh.index,
    )

    trade_rows = []

    for i in T:
        cb = current_buy_qh[i].X
        cs = current_sell_qh[i].X

        if cb is not None and cb > 0:
            trade_rows.append((
                execution_time,
                "buy",
                cb,
                prices_qh.loc[i, "price"],
                i,
                -cb * prices_qh.loc[i, "price"] / 4.0,
            ))

        if cs is not None and cs > 0:
            trade_rows.append((
                execution_time,
                "sell",
                cs,
                prices_qh.loc[i, "price"],
                i,
                cs * prices_qh.loc[i, "price"] / 4.0,
            ))

        results.loc[i, "current_buy_qh"]  = cb
        results.loc[i, "current_sell_qh"] = cs
        results.loc[i, "net_buy"]         = net_buy[i].X
        results.loc[i, "net_sell"]        = net_sell[i].X
        results.loc[i, "charge_sign"]     = charge_sign[i].X
        results.loc[i, "battery_soc"]     = vars["battery_soc"][i].X

    trades = pd.DataFrame(
        trade_rows,
        columns=["execution_time", "side", "quantity", "price", "product", "profit"],
    )

    return results, trades, model.ObjVal


def simulate_one_day(
    current_day,
    discount_rate,
    bucket_size,
    c_rate,
    roundtrip_eff,
    max_cycles,
    min_trades,
    base_output_path,
):
    """
    Simuliert EINEN Tag (current_day) und gibt (day, profit, cycles) zurück.
    Jeder Aufruf kann in einem eigenen Prozess laufen.
    """
    import time

    total_db_time = 0.0
    total_solver_time = 0.0
    total_vwap_csv_time = 0.0



    # eigene DB-Connection für diesen Prozess / Tag
    conn = psycopg2.connect(CONNECTION)
    cursor = conn.cursor()

    # Logging-Pfade
    year = current_day.year
    path = os.path.join(
        base_output_path,
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

    os.makedirs(tradepath, exist_ok=True)
    os.makedirs(vwappath, exist_ok=True)

    # Tagesstart normalisieren
    current_day = current_day.replace(hour=0, minute=0, second=0, microsecond=0)
    print("simulate_one_day:", current_day)

    all_trades = pd.DataFrame(
        columns=["execution_time", "side", "quantity", "price", "product", "profit"]
    )

    trading_start = current_day - pd.Timedelta(hours=8)
    trading_end   = current_day + pd.Timedelta(days=1)

    execution_time_start = trading_start
    execution_time_end   = trading_start + pd.Timedelta(minutes=bucket_size)

    # simple cycles-Handling pro Tag (kannst du ggf. anpassen)
    current_cycles = 0.0
    allowed_cycles = max_cycles   # hier pro Tag volle Cycles erlauben, oder deine Logik nutzen

    # Tagesindex T für das Modell
    start_of_day = (trading_end - pd.Timedelta(hours=2)).replace(hour=0, minute=0)
    end_of_day   = start_of_day.replace(hour=23, minute=45)
    T = list(pd.date_range(start_of_day, end_of_day, freq="15min"))

    model, vars_dict, netting_constr = build_battery_model(
        T, cap=1, c_rate=c_rate, roundtrip_eff=roundtrip_eff, max_cycles=allowed_cycles
    )

    net_trades = pd.DataFrame(
        columns=["sum_buy", "sum_sell", "net_buy", "net_sell", "product"]
    )

    # Datei für VWAP-Logging dieses Tages (falls du es behalten willst)
    vwap_filename = os.path.join(
        vwappath, "vwaps_" + current_day.strftime("%Y-%m-%d") + ".csv"
    )
    if os.path.exists(vwap_filename):
        os.remove(vwap_filename)

    while execution_time_end < trading_end:
         
        t0 = time.perf_counter()

        # VWAP aus DB
        vwap = get_average_prices(
            cursor,
            "BUY",
            execution_time_start,
            execution_time_end,
            trading_end,
            min_trades=min_trades,
        )
        total_db_time += time.perf_counter() - t0

        # Optionales VWAP-Logging – kannst du bei Performance-Tests auch auskommentieren
        t0 = time.perf_counter()

        vwaps_for_logging = (
            vwap.copy().rename(columns={"price": execution_time_end}).T
        )
        vwaps_for_logging.to_csv(
            vwap_filename,
            mode="a",
            header=not os.path.exists(vwap_filename),
            index=True,
        )
        total_vwap_csv_time += time.perf_counter() - t0

        net_trades = get_net_trades(all_trades, trading_end)

        if vwap["price"].isnull().all():
            execution_time_start = execution_time_end
            execution_time_end   = execution_time_start + pd.Timedelta(
                minutes=bucket_size
            )
            continue

        try:
            t0 = time.perf_counter()
            results, trades, profit = solve_bucket_with_model(
                model,
                vars_dict,
                netting_constr,
                vwap,
                execution_time_start,
                discount_rate,
                net_trades,
            )
            total_solver_time += time.perf_counter() - t0
            
            all_trades = pd.concat([all_trades, trades])
        except ValueError:
            print("Error in optimization at", execution_time_start)

        execution_time_start = execution_time_end
        execution_time_end   = execution_time_start + pd.Timedelta(
            minutes=bucket_size
        )

    daily_profit = all_trades["profit"].sum()
    current_cycles += net_trades["net_buy"].sum() / 4.0 * roundtrip_eff**0.5

    # Trades des Tages speichern
    trades_filename = os.path.join(
        tradepath, "trades_" + current_day.strftime("%Y-%m-%d") + ".csv"
    )
    all_trades.to_csv(trades_filename, index=False)

    cursor.close()
    conn.close()

    print(f"DB-Zeit gesamt:       {total_db_time:.2f} s")
    print(f"Solver-Zeit gesamt:   {total_solver_time:.2f} s")
    print(f"VWAP-CSV gesamt:      {total_vwap_csv_time:.2f} s")

    return {
        "day": current_day,
        "profit": float(daily_profit),
        "cycles": float(current_cycles),
    }


from multiprocessing import Pool
import time

def run_period_parallel(
    start_day,
    end_day,
    discount_rate,
    bucket_size,
    c_rate,
    roundtrip_eff,
    max_cycles,
    min_trades,
    base_output_path="output",
    processes=8,
):
    # Liste aller Tage erzeugen
    days = []
    d = start_day
    while d < end_day:
        days.append(d)
        d = d + pd.Timedelta(days=1)

    tasks = []
    for day in days:
        tasks.append((
            day,
            discount_rate,
            bucket_size,
            c_rate,
            roundtrip_eff,
            max_cycles,
            min_trades,
            base_output_path,
        ))

    t0 = time.perf_counter()
    with Pool(processes=processes) as pool:
        results = pool.starmap(simulate_one_day, tasks)
    t1 = time.perf_counter()

    print(f"Parallel run finished in {t1 - t0:.1f} seconds")

    profits_df = pd.DataFrame(results)
    profits_df.sort_values("day", inplace=True)
    profits_df.to_csv(os.path.join(base_output_path, "profit_all_days.csv"), index=False)

    return profits_df
