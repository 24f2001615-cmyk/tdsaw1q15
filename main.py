import ast
import asyncio
import json
import math
import os
import re
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

BASE = Path(__file__).parent
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
MODEL = os.environ.get("MODEL", "openai/gpt-4.1")
LLM_URL = "https://aipipe.org/openrouter/v1/chat/completions"
TOTAL_BUDGET = 13.0  # seconds; grader allows 15


# --------------------------------------------------------------------------
# Data: latest row per id (by real instant), IST business dates, USD amounts
# --------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    rates = json.loads((BASE / "rates.json").read_text())["usd_per_unit"]
    rows = []
    for line in (BASE / "export.ndjson").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="ISO8601")
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True, format="ISO8601")
    # an order id can repeat: the row with the latest updated_at (as an instant) is current
    df = df.sort_values("updated_at", kind="stable").drop_duplicates("id", keep="last")
    df["created_at"] = df["created_at"].dt.tz_convert("Asia/Kolkata")
    df["updated_at"] = df["updated_at"].dt.tz_convert("Asia/Kolkata")
    df["month"] = df["created_at"].dt.strftime("%Y-%m")
    df["date"] = df["created_at"].dt.strftime("%Y-%m-%d")
    df["usd"] = df["amount"].astype(float) * df["currency"].map(rates).astype(float)
    return df.sort_values("created_at").reset_index(drop=True)


DF = load_data()
PRODUCTS = sorted(DF["product"].unique().tolist())
REGIONS = sorted(DF["region"].unique().tolist())
CURRENCIES = sorted(DF["currency"].unique().tolist())
STATUSES = sorted(DF["status"].unique().tolist())

SYSTEM = f"""You answer questions about the Acme Appliances order ledger by writing Python (pandas).

A DataFrame `df` is already loaded (one row per order, already de-duplicated to the CURRENT version of each order). Also available: `pd`, `np`. No imports allowed.
Columns:
- id (str), customer (str like 'C1016'), region (str), product (str), qty (int), unit_price (float), amount (float, in the order's own currency), currency (str)
- status (str): one of {STATUSES}
- created_at, updated_at: tz-aware datetimes in Asia/Kolkata (IST)
- month (str 'YYYY-MM', IST month of created_at), date (str 'YYYY-MM-DD', IST date of created_at)
- usd (float): amount converted to USD (this is the column to use for ALL money answers)

Known values: regions {REGIONS}; products {PRODUCTS}; currencies {CURRENCIES}. Data covers {DF['month'].min()} to {DF['month'].max()}.

Business rules:
- "Revenue"/"sales" = sum of `usd` over orders with status == 'paid' only.
- "Refunds": orders with status == 'refunded'; refund amount = sum of `usd` of those orders; refund count = number of those orders.
- Status 'void' orders are neither revenue nor refunds.
- Time ("in March 2026", "Q1", "in April") always refers to the `month`/`date` columns (IST, based on created_at). Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec. If no year is given, use 2026.
- "Units sold" = sum of `qty` over paid orders unless the question says otherwise. "Number of orders" with no status mentioned = paid orders, unless the question is about refunds/voids/all orders.
- "Customers": distinct values of `customer`. "Top"/"best" = highest; for ties pick the alphabetically first name/id.
- Use product, region and customer names exactly as they appear in the data (match case-insensitively when filtering).

Output rules:
- Write ONLY Python code, no markdown fences, no explanation.
- Assign the final result to a variable named `answer`.
- Money: a float rounded to 2 decimals (round(x, 2)). Counts: int. Names/IDs: str. Lists: a Python list of str/numbers. If asked for several values, a dict with descriptive keys.
- Do not compute from rounded intermediates: sum first, round only at the end.
"""

# --------------------------------------------------------------------------
# Sandbox for the generated code
# --------------------------------------------------------------------------
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals", "locals",
    "getattr", "setattr", "delattr", "vars", "breakpoint", "exit", "quit", "help", "dir",
}
SAFE_BUILTINS = {
    n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
    for n in ["len", "sum", "min", "max", "round", "sorted", "set", "list", "dict", "tuple",
              "str", "int", "float", "abs", "range", "enumerate", "zip", "any", "all", "bool",
              "isinstance", "reversed", "map", "filter", "print"]
}


def validate(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            raise ValueError("imports are not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("private attributes are not allowed")
        if isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            raise ValueError(f"name not allowed: {node.id}")


def run_code(code: str):
    validate(code)
    env = {"__builtins__": SAFE_BUILTINS, "df": DF.copy(), "pd": pd, "np": np}
    exec(compile(code, "<generated>", "exec"), env)
    if "answer" not in env:
        raise ValueError("code did not set `answer`")
    return env["answer"]


def normalise(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        v = float(v)
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 2)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Series):
        if len(v) == 1:
            return normalise(v.iloc[0])
        return {str(k): normalise(x) for k, x in v.items()}
    if isinstance(v, pd.DataFrame):
        return json.loads(v.to_json(orient="records"))
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, dict):
        return {str(k): normalise(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [normalise(x) for x in v]
    return v


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------
def strip_fences(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


async def generate_code(question: str, timeout: float, prev_code: str = "", error: str = "") -> str:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Question: {question}"}]
    if error:
        messages.append({"role": "assistant", "content": prev_code})
        messages.append({"role": "user", "content": f"That code failed with: {error}\nReturn corrected code only."})
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            LLM_URL,
            headers={"Authorization": f"Bearer {AIPIPE_TOKEN}", "Content-Type": "application/json"},
            json={"model": MODEL, "temperature": 0, "messages": messages},
        )
        r.raise_for_status()
        return strip_fences(r.json()["choices"][0]["message"]["content"])


LAST = {"error": None, "code": None}


async def answer_question(question: str):
    start = time.monotonic()
    code, error = "", ""
    for attempt in range(2):
        remaining = TOTAL_BUDGET - (time.monotonic() - start)
        if remaining < 3:
            break
        try:
            code = await generate_code(question, remaining - 1, code, error)
            LAST["code"] = code
            value = await asyncio.wait_for(asyncio.to_thread(run_code, code), timeout=3)
            return normalise(value)
        except Exception as e:  # feed the error back once
            error = f"{type(e).__name__}: {e}"
            if isinstance(e, httpx.HTTPStatusError):
                error += " | " + e.response.text[:300]
            LAST["error"] = error
            print("ERROR:", error, flush=True)
    return None


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def health():
    return {"ok": True, "orders": len(DF), "token_set": bool(AIPIPE_TOKEN),
            "model": MODEL, "last_error": LAST["error"], "last_code": LAST["code"]}


@app.post("/")
@app.post("/ask")
async def ask(request: Request):
    body = await request.json()
    question = str(body.get("question", "")).strip()
    return {"answer": await answer_question(question)}
