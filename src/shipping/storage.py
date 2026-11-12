"""SQLite is the local system of record; primary keys make retries safe."""

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from shipping.config import paths
from shipping.validation import HISTORY_COLUMNS, QUOTE_COLUMNS, validate_quotes


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect():
    con = sqlite3.connect(paths().db, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    try:
        yield con
        con.commit()
    except BaseException:
        con.rollback()
        raise
    finally:
        con.close()


def init_db():
    common = """
        quote_id TEXT PRIMARY KEY, quote_timestamp TEXT NOT NULL,
        shipping_zone INTEGER NOT NULL CHECK(shipping_zone BETWEEN 2 AND 8),
        service_level TEXT NOT NULL, customer_segment TEXT NOT NULL,
        weight_lbs REAL NOT NULL CHECK(weight_lbs > 0),
        length_in REAL NOT NULL CHECK(length_in > 0),
        width_in REAL NOT NULL CHECK(width_in > 0),
        height_in REAL NOT NULL CHECK(height_in > 0),
        fuel_surcharge_pct REAL NOT NULL, estimated_shipping_cost REAL NOT NULL,
        quoted_price REAL NOT NULL
    """
    with connect() as con:
        con.execute(f"""CREATE TABLE IF NOT EXISTS historical_quotes ({common},
            accepted_quote INTEGER NOT NULL CHECK(accepted_quote IN (0,1)),
            outcome_timestamp TEXT NOT NULL)""")
        con.execute(f"CREATE TABLE IF NOT EXISTS daily_quotes ({common}, batch_date TEXT NOT NULL)")
        con.executescript("""
            CREATE TABLE IF NOT EXISTS recommendations (
                quote_id TEXT PRIMARY KEY, batch_date TEXT NOT NULL, model_version TEXT NOT NULL,
                recommended_price REAL NOT NULL, predicted_acceptance_probability REAL NOT NULL,
                expected_contribution REAL NOT NULL, estimated_shipping_cost REAL NOT NULL,
                latency_ms REAL NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, job_name TEXT NOT NULL, started_at TEXT NOT NULL,
                finished_at TEXT, duration_seconds REAL, status TEXT NOT NULL,
                record_count INTEGER NOT NULL DEFAULT 0, error TEXT);
            CREATE TABLE IF NOT EXISTS monitoring (
                run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, report_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS api_events (
                event_id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, endpoint TEXT NOT NULL,
                status_code INTEGER NOT NULL, latency_ms REAL NOT NULL, model_version TEXT);
            CREATE TABLE IF NOT EXISTS observed_outcomes (
                quote_id TEXT PRIMARY KEY, accepted_quote INTEGER NOT NULL CHECK(accepted_quote IN (0,1)),
                outcome_timestamp TEXT NOT NULL);
        """)


def save_quotes(frame, *, labeled, batch_date=None):
    valid = validate_quotes(frame, labeled=labeled)
    init_db()
    table = "historical_quotes" if labeled else "daily_quotes"
    columns = HISTORY_COLUMNS if labeled else QUOTE_COLUMNS + ["batch_date"]
    if not labeled:
        if batch_date is None:
            raise ValueError("batch_date is required")
        valid["batch_date"] = batch_date
    for name in ["quote_timestamp"] + (["outcome_timestamp"] if labeled else []):
        valid[name] = valid[name].map(lambda x: x.isoformat())
    with connect() as con:
        con.executemany(
            f"INSERT OR IGNORE INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            valid[columns].itertuples(index=False, name=None),
        )


def load_history():
    init_db()
    with connect() as con:
        return pd.read_sql_query("SELECT * FROM historical_quotes ORDER BY quote_timestamp", con)


def load_daily(batch_date):
    init_db()
    with connect() as con:
        return pd.read_sql_query(
            "SELECT " + ",".join(QUOTE_COLUMNS) + " FROM daily_quotes WHERE batch_date=? ORDER BY quote_id",
            con, params=[batch_date],
        )


def counts():
    init_db()
    tables = ["historical_quotes", "daily_quotes", "recommendations", "runs", "monitoring", "api_events"]
    with connect() as con:
        return {name: con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] for name in tables}


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n")


def start_run(run_id, job_name):
    init_db()
    with connect() as con:
        con.execute("INSERT INTO runs(run_id,job_name,started_at,status) VALUES(?,?,?,?)",
                    (run_id, job_name, utc_now(), "RUNNING"))


def finish_run(run_id, status, record_count=0, error=None):
    with connect() as con:
        row = con.execute("SELECT started_at FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row:
            now = datetime.now(timezone.utc)
            duration = (now - datetime.fromisoformat(row[0])).total_seconds()
            con.execute("""UPDATE runs SET finished_at=?,duration_seconds=?,status=?,record_count=?,error=?
                           WHERE run_id=?""", (now.isoformat(), duration, status, record_count, error, run_id))


def score_batch(model, entry, frame):
    from shipping.pricing import recommend

    valid = validate_quotes(frame, labeled=False)
    output = []
    for quote in valid.to_dict(orient="records"):
        start = time.perf_counter()
        result, _ = recommend(model, quote, entry["version"])
        result["estimated_shipping_cost"] = quote["estimated_shipping_cost"]
        result["latency_ms"] = (time.perf_counter() - start) * 1000
        output.append(result)
    return output


def save_recommendations(rows, batch_date):
    # One transaction: failed batches cannot leave a partially saved recommendation set.
    with connect() as con:
        con.executemany("""INSERT INTO recommendations(
            quote_id,batch_date,model_version,recommended_price,predicted_acceptance_probability,
            expected_contribution,estimated_shipping_cost,latency_ms,created_at) VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(quote_id) DO NOTHING""", [
            (x["quote_id"], batch_date, x["model_version"], x["recommended_price"],
             x["predicted_acceptance_probability"], x["expected_contribution"],
             x["estimated_shipping_cost"], x["latency_ms"], utc_now()) for x in rows
        ])
        return con.execute("SELECT COUNT(*) FROM recommendations WHERE batch_date=?", (batch_date,)).fetchone()[0]


def record_api_event(endpoint, status_code, latency_ms, model_version):
    with connect() as con:
        con.execute("""INSERT INTO api_events(created_at,endpoint,status_code,latency_ms,model_version)
                       VALUES(?,?,?,?,?)""", (utc_now(), endpoint, status_code, latency_ms, model_version))
