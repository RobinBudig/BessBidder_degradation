import os
import pandas as pd
import numpy as np
import psycopg2

from dotenv import load_dotenv
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


def precompute_vwaps_for_day(current_day, bucket_size, min_trades, output_path):
    """
    Für EINEN Tag:
        - hole alle Trades aus der DB
        - bucketiere sie in execution_time_end-Fenster
        - berechne VWAPs
        - speichere als Parquet
    """

    conn = psycopg2.connect(CONNECTION)
    cursor = conn.cursor()

    trading_start = current_day - pd.Timedelta(hours=8)
    trading_end   = current_day + pd.Timedelta(days=1)

    print("Precomputing:", current_day.date())

    # -------------------
    # 1. ALLE Trades für diesen Zeitraum holen
    # -------------------
    cursor.execute(f"""
        SELECT 
            executiontime,
            deliverystart,
            weighted_avg_price,
            volume,
            trade_count
        FROM transactions_intraday_de
        WHERE executiontime >= '{trading_start:%Y-%m-%d %H:%M:%S}'
          AND executiontime <  '{trading_end:%Y-%m-%d %H:%M:%S}'
          AND side = 'BUY';
    """)

    raw = cursor.fetchall()
    columns = ["executiontime", "deliverystart", "price", "volume", "trade_count"]

    df = pd.DataFrame(raw, columns=columns)

    if df.empty:
        print(" → Keine Trades gefunden, überspringe Tag.")
        return

    df["executiontime"] = (
    pd.to_datetime(df["executiontime"], utc=True)
      .dt.tz_convert("Europe/Berlin")
      )

    df["deliverystart"] = (
    pd.to_datetime(df["deliverystart"], utc=True)
      .dt.tz_convert("Europe/Berlin")
      )


    # -------------------
    # 2. Bucket-Index bestimmen
    # -------------------
    # Zuerst alles in UTC runden, dann zurück nach Europe/Berlin
    et_utc = df["executiontime"].dt.tz_convert("UTC")

    df["bucket_end"] = (
        et_utc.dt.floor(f"{bucket_size}min")
        + pd.Timedelta(minutes=bucket_size)
    )
    # Bucket-Enden wieder in Europe/Berlin, damit sie zu trading_start/bucket_index passen
    df["bucket_end"] = df["bucket_end"].dt.tz_convert("Europe/Berlin")


    # Alle möglichen Bucket-Enden
    bucket_index = pd.date_range(
        trading_start + pd.Timedelta(minutes=bucket_size),
        trading_end,
        freq=f"{bucket_size}min",
    )

    # Alle Produkte des Tages
    start_of_day = (trading_end - pd.Timedelta(hours=2)).replace(hour=0, minute=0)
    end_of_day   = start_of_day.replace(hour=23, minute=45)
    product_index = pd.date_range(start_of_day, end_of_day, freq="15min")

    # -------------------
    # 3. VWAPs aggregieren
    # -------------------
    grouped = df.groupby(["bucket_end", "deliverystart"]).agg(
        total_value=("price",  lambda x: (x * df.loc[x.index, "volume"]).sum()),
        total_volume=("volume", "sum"),
        total_trades=("trade_count", "sum"),
    )

    grouped = grouped[grouped["total_trades"] >= min_trades]

    grouped["vwap"] = grouped["total_value"] / grouped["total_volume"]

    # -------------------
    # 4. Matrix formen (bucket_end x deliverystart)
    # -------------------
    matrix = grouped["vwap"].unstack(level="deliverystart")

    # Index/Columns vollständig machen
    matrix = matrix.reindex(index=bucket_index, columns=product_index)

    # WICHTIG:
    # Index: als Datetime lassen (mit tz)
    # Columns: in String
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    

    # -------------------
    # 5. Speichern
    # -------------------
    os.makedirs(output_path, exist_ok=True)
    fname = os.path.join(output_path, f"vwaps_{current_day:%Y-%m-%d}.parquet")

    matrix.to_parquet(fname)
    print(" → gespeichert:", fname)

    cursor.close()
    conn.close()



def _get_existing_parquet_days(output_path):
    """
    Liest alle vorhandenen vwaps_YYYY-MM-DD.parquet Dateien im output_path ein
    und gibt ein Set von datetime.date-Objekten zurück.
    """
    if not os.path.isdir(output_path):
        return set()

    existing_days = set()
    for fname in os.listdir(output_path):
        if fname.startswith("vwaps_") and fname.endswith(".parquet"):
            date_str = fname[len("vwaps_"):-len(".parquet")]
            try:
                day = pd.to_datetime(date_str).date()
                existing_days.add(day)
            except Exception:
                # Falls mal eine Datei nicht dem Schema entspricht, einfach ignorieren
                continue
    return existing_days


def precompute_range(start_day, end_day, bucket_size, min_trades, output_path):
    # 1. Existierende Dateien im Zielordner prüfen
    existing_days = _get_existing_parquet_days(output_path)

    start_date = start_day.date()
    end_date   = end_day.date()

    # 2. Falls schon Dateien existieren: ab dem Tag nach dem letzten vorhandenen weitermachen
    if existing_days:
        last_existing = max(d for d in existing_days if d >= start_date and d < end_date) \
                        if any(d >= start_date and d < end_date for d in existing_days) \
                        else None
        if last_existing is not None:
            resume_date = last_existing + pd.Timedelta(days=1)
            # gleiche Zeitzone wie start_day
            d = pd.Timestamp(resume_date, tz=start_day.tzinfo)
            print(f"Starte nicht bei {start_day.date()}, "
                  f"sondern fahre fort ab {d.date()} (letzte vorhandene Datei: {last_existing}).")
        else:
            d = start_day
    else:
        d = start_day

    # 3. Tage bis end_day durchgehen und nur fehlende berechnen
    while d < end_day:
        if d.date() in existing_days:
            print(f"Parquet für {d.date()} existiert bereits – überspringe.")
        else:
            precompute_vwaps_for_day(d, bucket_size, min_trades, output_path)
        d += pd.Timedelta(days=1)


if __name__ == "__main__":
    start = pd.Timestamp("2019-01-01", tz="Europe/Berlin")
    end   = pd.Timestamp("2024-01-01", tz="Europe/Berlin")

    output_path = os.path.join("data", "precomputed_vwaps")

    precompute_range(
        start_day=start,
        end_day=end,
        bucket_size=15,
        min_trades=10,
        output_path=output_path,
    )
