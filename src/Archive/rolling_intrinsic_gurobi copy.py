import os

import numpy as np
import pandas as pd
from loguru import logger
from pulp import GUROBI, PULP_CBC_CMD, LpMaximize, LpProblem, LpVariable, lpSum
import gurobipy as gp

from dotenv import load_dotenv

load_dotenv()

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



def load_vwaps_for_day(current_day: pd.Timestamp, vwaps_base_path: str) -> pd.DataFrame:
    """
    Lädt die vorab berechnete VWAP-Matrix für einen Tag aus einer Parquet-Datei.

    Erwartetes Format (wie im Precompute-Skript gespeichert):
      - index: bucket_end (Execution Time Ende), als String -> hier zurück zu Datetime
      - columns: deliverystart-Produkte, als String -> hier zurück zu Datetime
      - values: VWAP-Preis
    """
    fname = os.path.join(vwaps_base_path, f"vwaps_{current_day:%Y-%m-%d}.parquet")
    vwaps_day = pd.read_parquet(fname)

    # Strings zurück in Timestamps
    vwaps_day.index = pd.to_datetime(vwaps_day.index)
    vwaps_day.columns = pd.to_datetime(vwaps_day.columns)

    return vwaps_day


def get_vwap_from_precomputed(
    vwaps_day: pd.DataFrame,
    execution_time_end: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Liefert ein DataFrame im gleichen Format wie früher get_average_prices():
    index = Produkte (15-min deliverystart), Spalte 'price'.

    Nutzt die vorab berechnete VWAP-Matrix eines Tages.
    """

    # wie im alten get_average_prices: Produktindex eines Tages bauen
    start_of_day = pd.to_datetime(end_date) - pd.Timedelta(hours=2)
    start_of_day = start_of_day.replace(hour=0, minute=0)
    end_of_day = start_of_day.replace(hour=23, minute=45)

    product_index = pd.date_range(start_of_day, end_of_day, freq="15min")

    # Falls für dieses Bucket-Ende nichts vorliegt:
    if execution_time_end not in vwaps_day.index:
        vwap = pd.DataFrame(index=product_index, columns=["price"], dtype=float)
        return vwap

    # Reihe aus der Matrix: index = Produkt-Timestamps, values = Preis
    row = vwaps_day.loc[execution_time_end]  # Series

    vwap = pd.DataFrame({"price": row})
    vwap.index = pd.to_datetime(vwap.index)
    vwap = vwap.reindex(product_index)

    return vwap




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





def run_optimization_quarterhours_repositioning_gurobi(
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
    # --- 1) Daten vorbereiten -------------------------------------------------
    # Diskontierte Preise (sell/buy) wie bisher, nur einmal vektoriell
    prices_qh_adj_all = adjust_prices_block(prices_qh, execution_time, discount_rate)

    # prev_net_trades an den Index von prices_qh anpassen, fehlende mit 0 füllen
    prev_net_trades = prev_net_trades.reindex(prices_qh.index).fillna(0.0)

    e = 0.01
    efficiency = roundtrip_eff ** 0.5
    M = cap * c_rate

    T = list(prices_qh.index)

    # --- 2) Modell aufbauen ---------------------------------------------------
    m = gp.Model("battery")
    m.Params.OutputFlag = 0  # keine Textausgabe von Gurobi

    # Variablen
    current_buy_qh  = m.addVars(T, lb=0.0, name="current_buy_qh")
    current_sell_qh = m.addVars(T, lb=0.0, name="current_sell_qh")
    battery_soc     = m.addVars(T, lb=0.0, name="battery_soc")

    net_buy   = m.addVars(T, lb=0.0, name="net_buy")
    net_sell  = m.addVars(T, lb=0.0, name="net_sell")
    charge_sign = m.addVars(T, vtype=gp.GRB.BINARY, name="charge_sign")

    z = m.addVars(T, lb=0.0, name="z")
    w = m.addVars(T, lb=0.0, name="w")

    # --- 3) Zielfunktion ------------------------------------------------------
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
            # "adjusted_obj" Fall
            term = (
                current_sell_qh[i] * (price_sell_adj - 0.1 / 2 - e)
                - current_buy_qh[i] * (price_buy_adj + 0.1 / 2 + e)
            ) / 4.0
        else:
            # "original_obj" Fall
            term = (
                current_sell_qh[i] * (price_sell_adj - e)
                - current_buy_qh[i] * (price_buy_adj + e)
            ) / 4.0

        obj += term

    m.setObjective(obj, gp.GRB.MAXIMIZE)

    # --- 4) Nebenbedingungen --------------------------------------------------
    # SOC-Dynamik (mit deiner aktuellen efficiency-Logik)
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

    # Initialer SOC
    m.addConstr(battery_soc[T[0]] == 0.0, name="InitialBatterySOC")

    # Pro Zeitschritt Constraints
    for i in T:
        price = prices_qh.loc[i, "price"]

        if pd.isna(price):
            # Keine Preise -> keine Trades
            m.addConstr(current_buy_qh[i] == 0.0,  name=f"NaNBuy_{i}")
            m.addConstr(current_sell_qh[i] == 0.0, name=f"NaNSell_{i}")
        else:
            m.addConstr(battery_soc[i] <= cap,           name=f"Cap_{i}")
            m.addConstr(net_buy[i]   <= cap * c_rate,    name=f"BuyRate_{i}")
            m.addConstr(net_sell[i]  <= cap * c_rate,    name=f"SellRate_{i}")
            m.addConstr(
                net_sell[i] / efficiency / 4.0 <= battery_soc[i],
                name=f"SellVsSOC_{i}",
            )

        # Big-M-Logik bzgl. charge_sign
        m.addConstr(net_buy[i]  <= M * charge_sign[i],         name=f"NetBuyBigM_{i}")
        m.addConstr(net_sell[i] <= M * (1 - charge_sign[i]),   name=f"NetSellBigM_{i}")

        m.addConstr(z[i] <= charge_sign[i] * M,             name=f"ZUpper_{i}")
        m.addConstr(z[i] <= net_buy[i],                     name=f"ZNetBuy_{i}")
        m.addConstr(z[i] >= net_buy[i] - (1 - charge_sign[i]) * M, name=f"ZLower_{i}")
        m.addConstr(z[i] >= 0.0,                            name=f"ZNonNeg_{i}")

        m.addConstr(w[i] <= (1 - charge_sign[i]) * M,       name=f"WUpper_{i}")
        m.addConstr(w[i] <= net_sell[i],                    name=f"WNetSell_{i}")
        m.addConstr(w[i] >= net_sell[i] - charge_sign[i] * M, name=f"WLower_{i}")
        m.addConstr(w[i] >= 0.0,                            name=f"WNonNeg_{i}")

        # Nettung
        m.addConstr(
            z[i] - w[i]
            == current_buy_qh[i]
            + prev_net_trades.loc[i, "net_buy"]
            - current_sell_qh[i]
            - prev_net_trades.loc[i, "net_sell"],
            name=f"Netting_{i}",
        )

    # Zyklenlimit
    m.addConstr(
        gp.quicksum(net_buy[i] * efficiency / 4.0 for i in T) <= max_cycles * cap,
        name="MaxCycles",
    )

    # --- 5) Lösen -------------------------------------------------------------
    m.optimize()

    # --- 6) Ergebnisse einsammeln ---------------------------------------------
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
        results.loc[i, "battery_soc"]     = battery_soc[i].X

    trades = pd.DataFrame(
        trade_rows,
        columns=["execution_time", "side", "quantity", "price", "product", "profit"],
    )

    return results, trades, m.ObjVal




def simulate_period(
    start_day,
    end_day,
    discount_rate,
    bucket_size,
    c_rate,
    roundtrip_eff,
    max_cycles,
    min_trades,
    vwaps_base_path=os.path.join("data", "precomputed_vwaps")
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

    # Output-Pfade wie bisher
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

    os.makedirs(path, exist_ok=True)
    os.makedirs(tradepath, exist_ok=True)
    os.makedirs(vwappath, exist_ok=True)

    profitpath = os.path.join(path, "profit.csv")
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
        current_cycles = 0

    net_trades = pd.DataFrame(
        columns=["sum_buy", "sum_sell", "net_buy", "net_sell", "product"]
    )

    while current_day < end_day:
        current_day = current_day.replace(hour=0, minute=0, second=0, microsecond=0)
        print("current_day: ", current_day)

        all_trades = pd.DataFrame(
            columns=["execution_time", "side", "quantity", "price", "product", "profit"]
        )

        # Trading-Fenster wie gehabt
        trading_start = current_day - pd.Timedelta(hours=8)
        trading_end   = current_day + pd.Timedelta(days=1)

        print("trading_start: ", trading_start)
        print("trading_end: ", trading_end)

        # VWAP-Matrix für diesen Tag aus Parquet laden
        vwaps_day = load_vwaps_for_day(current_day, vwaps_base_path)

        execution_time_start = trading_start
        execution_time_end   = trading_start + pd.Timedelta(minutes=bucket_size)

        days_left = (end_day - current_day).days
        days_done = (current_day - start_day).days

        allowed_cycles = 1 + max(0, days_done - current_cycles)

        print("Days left: ", days_left)
        print("Current cycles: ", current_cycles)
        print("Allowed cycles: ", allowed_cycles)

        while execution_time_end < trading_end:
            # statt DB-Abfrage: VWAP für diesen Bucket aus der precomputed Matrix holen
            vwap = get_vwap_from_precomputed(
                vwaps_day,
                execution_time_end,
                trading_end,
            )

            # optional: Logging der VWAPs wie vorher
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
            else:
                vwaps_for_logging.to_csv(
                    vwap_filename,
                    mode="a",
                    header=False,
                    index=True,
                )

            net_trades = get_net_trades(all_trades, trading_end)

            # wie bisher: wenn alle Preise NaN sind → skip
            if vwap["price"].isnull().all():
                execution_time_start = execution_time_end
                execution_time_end   = execution_time_start + pd.Timedelta(
                    minutes=bucket_size
                )
                continue

            try:
                results, trades, profit = run_optimization_quarterhours_repositioning_gurobi(
                    vwap,
                    execution_time_start,
                    1,
                    c_rate,
                    roundtrip_eff,
                    allowed_cycles,
                    discount_rate,
                    net_trades,
                )
                all_trades = pd.concat([all_trades, trades])
            except ValueError:
                print("Error in optimization")
                print("execution_time_start: ", execution_time_start)

            execution_time_start = execution_time_end
            execution_time_end   = execution_time_start + pd.Timedelta(
                minutes=bucket_size
            )

        daily_profit = all_trades["profit"].sum()
        current_cycles += net_trades["net_buy"].sum() / 4.0 * roundtrip_eff**0.5

        all_trades.to_csv(
            os.path.join(
                tradepath, "trades_" + current_day.strftime("%Y-%m-%d") + ".csv"
            ),
            index=False,
        )

        profits = pd.concat(
            [
                profits,
                pd.DataFrame(
                    [[current_day, daily_profit, current_cycles]],
                    columns=["day", "profit", "cycles"],
                ),
            ]
        )

        profits.to_csv(os.path.join(path, "profit.csv"), index=False)

        current_day = current_day + pd.Timedelta(days=1) + pd.Timedelta(hours=2)

