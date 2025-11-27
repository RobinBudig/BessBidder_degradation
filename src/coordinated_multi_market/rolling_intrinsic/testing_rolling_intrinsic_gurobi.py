import os

import numpy as np
import pandas as pd
from loguru import logger
import gurobipy as gp
import warnings

from dotenv import load_dotenv
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



load_dotenv()

PRECOMPUTED_VWAP_PATH = os.path.join("data", "precomputed_vwaps")

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





def get_net_trades(trades: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
    # vollständiger Index für alle Quarterhours dieses Liefertages
    start = end_date - pd.Timedelta(hours=2)
    start = start.replace(hour=0, minute=0)
    end   = start.replace(hour=23, minute=45)
    idx = pd.date_range(start, end, freq="15min")

    # Fall 1: keine Trades -> einfach alles 0
    if trades.empty:
        return pd.DataFrame(
            0.0,
            index=idx,
            columns=["sum_buy", "sum_sell", "net_buy", "net_sell"],
        )

    # Fall 2: es gibt Trades -> aggregieren, dann auf idx reindexen
    grouped = trades.groupby(["product", "side"])["quantity"].sum().unstack(fill_value=0)

    # Falls die Columns "buy"/"sell" nicht existieren, gib get(...) 0 zurück
    grouped["sum_buy"]  = grouped.get("buy", 0.0)
    grouped["sum_sell"] = grouped.get("sell", 0.0)

    grouped["net_buy"]  = grouped["sum_buy"]  - grouped["sum_sell"]
    grouped["net_sell"] = grouped["sum_sell"] - grouped["sum_buy"]

    # negative Werte auf 0
    grouped["net_buy"]  = grouped["net_buy"].clip(lower=0.0)
    grouped["net_sell"] = grouped["net_sell"].clip(lower=0.0)

    # nur die relevanten Spalten behalten
    grouped = grouped[["sum_buy", "sum_sell", "net_buy", "net_sell"]]

    # auf kompletten Tagesindex bringen, fehlende mit 0 auffüllen
    return grouped.reindex(idx, fill_value=0.0)









def load_vwaps_for_day(current_day: pd.Timestamp,
                       vwaps_base_path: str = PRECOMPUTED_VWAP_PATH) -> pd.DataFrame:
    
    fname = os.path.join(vwaps_base_path, f"vwaps_{current_day:%Y-%m-%d}.parquet")

    if not os.path.exists(fname):
        raise FileNotFoundError(f"VWAP-Parquet für {current_day:%Y-%m-%d} nicht gefunden: {fname}")

    matrix = pd.read_parquet(fname)

    # Index: execution_time_end (Bucket-Ende)
    matrix.index = (
        pd.to_datetime(matrix.index, utc=True)
          .tz_convert("Europe/Berlin")
    )

    # Columns: deliverystart-Zeiten
    matrix.columns = (
        pd.to_datetime(matrix.columns, utc=True)
          .tz_convert("Europe/Berlin")
    )

    return matrix



def get_vwap_from_precomputed(
    vwaps_day: pd.DataFrame,
    execution_time_end: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    # end_date sollte bereits tz='Europe/Berlin' haben
    end_date = end_date.tz_convert("Europe/Berlin")

    start_of_day = end_date - pd.Timedelta(hours=2)
    start_of_day = start_of_day.replace(hour=0, minute=0)
    end_of_day   = start_of_day.replace(hour=23, minute=45)

    product_index = pd.date_range(start_of_day, end_of_day, freq="15min", tz="Europe/Berlin")

    if execution_time_end not in vwaps_day.index:
        return pd.DataFrame(index=product_index, columns=["price"], dtype=float)

    row = vwaps_day.loc[execution_time_end]  # Series

    vwap = row.to_frame(name="price")
    vwap = vwap.reindex(product_index)

    return vwap





def build_battery_model(T, cap, c_rate, roundtrip_eff):
    efficiency = roundtrip_eff ** 0.5
    M = cap * c_rate

    m = gp.Model("battery_persistent")
    m.Params.OutputFlag = 0

    current_buy_qh  = m.addVars(T, lb=0.0, name="current_buy_qh")
    current_sell_qh = m.addVars(T, lb=0.0, name="current_sell_qh")
    battery_soc     = m.addVars(T, lb=0.0, name="battery_soc")

    net_buy   = m.addVars(T, lb=0.0, name="net_buy")
    net_sell  = m.addVars(T, lb=0.0, name="net_sell")
    charge_sign = m.addVars(T, vtype=gp.GRB.BINARY, name="charge_sign")

    z = m.addVars(T, lb=0.0, name="z")
    w = m.addVars(T, lb=0.0, name="w")

    # SOC-Dynamik
    prev = T[0]
    for i in T[1:]:
        m.addConstr(
            battery_soc[i]
            == battery_soc[prev]
            + net_buy[prev] * efficiency / 4.0
            - net_sell[prev] / 4.0 / efficiency,
            name=f"BatteryBalance_{i}",
        )
        prev = i

    m.addConstr(battery_soc[T[0]] == 0.0, name="InitialBatterySOC")

    # Zeitunabhängige Constraints
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

        m.addConstr(z[i] <= charge_sign[i] * M,                    name=f"ZUpper_{i}")
        m.addConstr(z[i] <= net_buy[i],                            name=f"ZNetBuy_{i}")
        m.addConstr(z[i] >= net_buy[i] - (1 - charge_sign[i]) * M, name=f"ZLower_{i}")
        m.addConstr(z[i] >= 0.0,                                   name=f"ZNonNeg_{i}")

        m.addConstr(w[i] <= (1 - charge_sign[i]) * M,              name=f"WUpper_{i}")
        m.addConstr(w[i] <= net_sell[i],                           name=f"WNetSell_{i}")
        m.addConstr(w[i] >= net_sell[i] - charge_sign[i] * M,      name=f"WLower_{i}")
        m.addConstr(w[i] >= 0.0,                                   name=f"WNonNeg_{i}")

    # Netting-Constraints mit RHS=0, später per RHS angepasst:
    netting_constr = {}
    for i in T:
        # z[i] - w[i] - current_buy_qh[i] + current_sell_qh[i] = RHS
        # RHS wird später zu (prev_net_buy - prev_net_sell) gesetzt
        c = m.addConstr(
            z[i] - w[i] - current_buy_qh[i] + current_sell_qh[i] == 0.0,
            name=f"Netting_{i}",
        )
        netting_constr[i] = c

    # Zyklen-Constraint; RHS wird später je nach allowed_cycles gesetzt
    max_cycles_constr = m.addConstr(
        gp.quicksum(net_buy[i] * efficiency / 4.0 for i in T) <= 0.0,
        name="MaxCycles",
    )

    m.setObjective(0.0, gp.GRB.MAXIMIZE)
    m.update()

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

    return m, vars, netting_constr, max_cycles_constr








def solve_bucket_with_persistent_model(
    m,
    vars,
    netting_constr,
    max_cycles_constr,
    prices_qh,
    execution_time,
    discount_rate,
    prev_net_trades,
    allowed_cycles,
):
    T = list(prices_qh.index)
    current_buy_qh  = vars["current_buy_qh"]
    current_sell_qh = vars["current_sell_qh"]
    net_buy         = vars["net_buy"]
    net_sell        = vars["net_sell"]
    charge_sign     = vars["charge_sign"]
    z               = vars["z"]
    w               = vars["w"]

    # 1) Diskontierte Preise vektoriell vorbereiten
    prices_qh_adj_all = adjust_prices_block(prices_qh, execution_time, discount_rate)
    prev_net_trades = prev_net_trades.reindex(prices_qh.index).fillna(0.0)

    e = 0.01

    # 2) MaxCycles-RHS setzen
    max_cycles_constr.RHS = allowed_cycles * 1.0  # *cap (falls cap!=1)

    # 3) Netting-RHS und Var-Bounds (NaNs) setzen
    for i in T:
        prev_nb = prev_net_trades.loc[i, "net_buy"]
        prev_ns = prev_net_trades.loc[i, "net_sell"]

        # RHS = prev_nb - prev_ns (siehe Herleitung)
        netting_constr[i].RHS = float(prev_nb - prev_ns)

        price = prices_qh.loc[i, "price"]
        if pd.isna(price):
            current_buy_qh[i].UB  = 0.0
            current_sell_qh[i].UB = 0.0
        else:
            # wieder „normal“ freigeben (z.B. cap*c_rate)
            # hier als Beispiel unbeschränkt nach oben:
            current_buy_qh[i].UB  = gp.GRB.INFINITY
            current_sell_qh[i].UB = gp.GRB.INFINITY

    # 4) Zielfunktion neu aufbauen
    obj = gp.LinExpr()
    for i in T:
        price = prices_qh.loc[i, "price"]
        if pd.isna(price):
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

    m.setObjective(obj, gp.GRB.MAXIMIZE)

    # 5) Optimieren
    m.optimize()


    # Check if a feasible solution was found
    if m.status != gp.GRB.OPTIMAL:
        # No optimal solution found, return empty results
        empty_trades = pd.DataFrame(
            columns=["execution_time", "side", "quantity", "price", "product", "profit"]
        )
        return None, empty_trades, 0.0

    # 6) Ergebnisse einsammeln (wie bisher)
    results = pd.DataFrame(
        index=prices_qh.index,
        columns=["current_buy_qh", "current_sell_qh", "net_buy", "net_sell", "charge_sign", "battery_soc"],
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

    return results, trades, m.ObjVal






def derive_day_ahead_trades_from_drl_output(
    output: pd.DataFrame,
    current_day: pd.Timestamp
) -> pd.DataFrame:
    """
    Übernommen aus dem alten Test-Skript:
    DRL-Output (hourly capacity_trade etc.) -> QH-Day-Ahead-Trades.

    Spalten im Rückgabewert:
    execution_time, side, quantity, price, product, profit
    """
    day_ahead_trades = {}

    # Zeile(n) des DRL-Outputs für genau diesen Tag holen
    df = output.loc[current_day.date().isoformat()].copy().round(2)

    # Nur Einträge mit tatsächlichem Trade
    mask = df.capacity_trade != 0
    df = df[mask]

    # buy/sell + Betrag
    df["side"] = ["buy" if x < 0 else "sell" for x in df.capacity_trade]
    df["net_volume"] = [abs(x) for x in df.capacity_trade]

    # Stundenprofit
    df["profit"] = df.capacity_trade * df.epex_spot_60min_de_lu_eur_per_mwh

    # time-Spalte in normaler Form
    df.reset_index(inplace=True)

    # Day-Ahead-Clearing: Vortag um 13:00
    day_ahead_market_clearing = (current_day - pd.Timedelta(days=1)).replace(hour=13)

    for _, row in df.iterrows():
        # Stundenposition auf 4 Viertelstunden verteilen
        product_indexes = pd.date_range(row["time"], periods=4, freq="15min")

        for product_index in product_indexes:
            day_ahead_trades.update(
                {
                    product_index: {
                        "execution_time": day_ahead_market_clearing,
                        "side": row["side"],
                        "quantity": row["net_volume"],
                        "price": row["epex_spot_60min_de_lu_eur_per_mwh"],
                        "product": product_index,
                        "profit": row["profit"] / 4,
                    }
                }
            )

    return pd.DataFrame(day_ahead_trades).T.reset_index(drop=True)






def simulate_days_stacked_quarterhourly_products(
    da_bids_path: str,
    output_path: str,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    discount_rate: float,
    bucket_size: int,
    c_rate: float,
    roundtrip_eff: float,
    max_cycles: float,
    min_trades: float,
    vwaps_base_path: str = PRECOMPUTED_VWAP_PATH,
) -> None:
    """
    "Testmodus" mit CSV-Outputs wie im alten Skript,
    aber mit dem schnellen persistenten Gurobi-Modell.
    Kein Return, keine PPO-Rückgabe.
    """

    log_message = (
        "Running FAST Rolling intrinsic QH TEST with the following parameters:\n"
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

    tradepath = os.path.join(output_path, "trades")
    vwappath = os.path.join(output_path, "vwap")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tradepath, exist_ok=True)
    os.makedirs(vwappath, exist_ok=True)

    profitpath = os.path.join(output_path, "profit.csv")

    # Profit-/Zyklen-Historie laden oder anlegen
    if os.path.exists(profitpath):
        profits = pd.read_csv(profitpath)
    else:
        profits = pd.DataFrame(columns=["day", "profit", "cycles"])

    if len(profits) > 0:
        current_day = (
            pd.Timestamp(profits.iloc[-1]["day"], tz="Europe/Berlin")
            + pd.Timedelta(days=1)
            + pd.Timedelta(hours=2)
        )
        current_cycles = profits.iloc[-1]["cycles"]
    else:
        current_day = start_day
        current_cycles = 0.0

    # DRL-DayAhead-Bids
    drl_output = pd.read_csv(da_bids_path, index_col="time", parse_dates=True)
    drl_output.index = drl_output.index.tz_convert("Europe/Berlin")

    efficiency = roundtrip_eff ** 0.5  # für Zyklen-Tracking

    while current_day < end_day:
        current_day = current_day.replace(hour=0, minute=0, second=0, microsecond=0)
        print("current_day: ", current_day)

        all_trades = pd.DataFrame(
            columns=["execution_time", "side", "quantity", "price", "product", "profit"]
        )

        # Day-Ahead-Trades aus DRL-Output ableiten
        try:
            day_ahead_trades_drl = derive_day_ahead_trades_from_drl_output(
                drl_output, current_day
            )
            all_trades = pd.concat([all_trades, day_ahead_trades_drl], ignore_index=True)
        except KeyError:
            logger.warning(
                f"Keine DRL-DayAhead-Trades für {current_day:%Y-%m-%d} im File {da_bids_path} – "
                "es wird nur Intraday optimiert."
            )

        trading_start = current_day - pd.Timedelta(hours=8)
        trading_end   = current_day + pd.Timedelta(days=1)

        print("trading_start: ", trading_start)
        print("trading_end: ", trading_end)

        # Batterie-Zeitindex (Delivery-QHs)
        start_of_day = trading_end - pd.Timedelta(hours=2)
        start_of_day = start_of_day.replace(hour=0, minute=0)
        end_of_day   = start_of_day.replace(hour=23, minute=45)

        T = list(pd.date_range(start_of_day, end_of_day, freq="15min", tz="Europe/Berlin"))

        # Persistentes Gurobi-Modell bauen
        m, vars, netting_constr, max_cycles_constr = build_battery_model(
            T, cap=1.0, c_rate=c_rate, roundtrip_eff=roundtrip_eff
        )

        # VWAP-Matrix für diesen Tag laden
        try:
            vwaps_day = load_vwaps_for_day(current_day, vwaps_base_path)
        except FileNotFoundError as e:
            logger.warning(
                f"Keine vorcomputierten VWAPs für {current_day:%Y-%m-%d}: {e}"
            )
            current_day = current_day + pd.Timedelta(days=1) + pd.Timedelta(hours=2)
            continue

        execution_time_start = trading_start
        execution_time_end   = trading_start + pd.Timedelta(minutes=bucket_size)

        days_left = (end_day - current_day).days
        days_done = (current_day - start_day).days
        # gleiche (zugegeben etwas ad-hoc) Logik wie im alten Testskript
        allowed_cycles = 1 + max(0, days_done - current_cycles)

        print("Days left: ", days_left)
        print("Current cycles: ", current_cycles)
        print("Allowed cycles (per day): ", allowed_cycles)

        # Intraday-Simulation über alle Buckets
        while execution_time_end < trading_end:
            vwap = get_vwap_from_precomputed(
                vwaps_day,
                execution_time_end=execution_time_end,
                end_date=trading_end,
            )

            # VWAP Logging analog zum alten Testskript
            vwaps_for_logging = (
                vwap.copy().rename(columns={"price": execution_time_end}).T
            )
            vwap_filename = os.path.join(
                vwappath, "vwaps_" + current_day.strftime("%Y-%m-%d") + ".csv"
            )

            if not os.path.exists(vwap_filename):
                vwaps_for_logging.to_csv(
                    vwap_filename,
                    mode="a",
                    header=True,
                    index=True,
                )
            elif os.path.exists(vwap_filename) and (
                execution_time_start == trading_start
            ):
                os.remove(vwap_filename)
                vwaps_for_logging.to_csv(
                    vwap_filename,
                    mode="a",
                    header=True,
                    index=True,
                )
            else:
                vwaps_for_logging.to_csv(
                    vwap_filename,
                    mode="a",
                    header=False,
                    index=True,
                )

            net_trades = get_net_trades(all_trades, trading_end)

            if vwap["price"].isnull().all():
                print("No VWAP prices in this bucket – skipping")
                execution_time_start = execution_time_end
                execution_time_end   = execution_time_start + pd.Timedelta(
                    minutes=bucket_size
                )
                continue

            try:
                results, trades, profit = solve_bucket_with_persistent_model(
                    m=m,
                    vars=vars,
                    netting_constr=netting_constr,
                    max_cycles_constr=max_cycles_constr,
                    prices_qh=vwap,
                    execution_time=execution_time_start,
                    discount_rate=discount_rate,
                    prev_net_trades=net_trades,
                    allowed_cycles=allowed_cycles,
                )
                # trades kann leer sein, aber concat ist robust
                all_trades = pd.concat([all_trades, trades], ignore_index=True)
            except ValueError as e:
                print("Error in optimization:", e)
                print("execution_time_start: ", execution_time_start)

            execution_time_start = execution_time_end
            execution_time_end   = execution_time_start + pd.Timedelta(
                minutes=bucket_size
            )

        # Tagesprofit & Zyklen
        daily_profit = all_trades["profit"].sum()

        # net_trades aus dem letzten Bucket -> Zyklenupdate
        net_trades = get_net_trades(all_trades, trading_end)
        current_cycles += net_trades["net_buy"].sum() / 4.0 * efficiency

        # Trades speichern
        all_trades.to_csv(
            os.path.join(
                tradepath, "trades_" + current_day.strftime("%Y-%m-%d") + ".csv"
            ),
            index=False,
        )

        # Profit / Zyklen anhängen
        profits = pd.concat(
            [
                profits,
                pd.DataFrame(
                    [[current_day, daily_profit, current_cycles]],
                    columns=["day", "profit", "cycles"],
                ),
            ],
            ignore_index=True,
        )
        profits.to_csv(profitpath, index=False)

        # Nächster Tag
        current_day = current_day + pd.Timedelta(days=1) + pd.Timedelta(hours=2)
