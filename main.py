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
MODEL = os.environ.get("MODEL", "openai/gpt-4.1-mini")
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
# Rule-based fallback (used when the language model is unavailable or fails)
# --------------------------------------------------------------------------
MONTH_WORDS = {"january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4,
               "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
               "september": 9, "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11,
               "nov": 11, "december": 12, "dec": 12}
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10}
RANK_HI = r"\b(top|best|highest|most|largest|biggest|leading|greatest|maximum|max|best-selling)\b"
RANK_LO = r"\b(lowest|least|worst|fewest|smallest|bottom|minimum|min)\b"
DIM_WORDS = {"product": r"\bproducts?\b|\bitem\b|\bappliances?\b", "customer": r"\bcustomers?\b|\bbuyers?\b|\bclients?\b",
             "region": r"\bregions?\b", "month": r"\bmonths?\b"}
DIM_COL = {"product": "product", "customer": "customer", "region": "region", "month": "month"}


def _period(t):
    ym = re.search(r"\b(20\d\d)\b", t)
    year = int(ym.group(1)) if ym else 2026
    ments = []
    for m in re.finditer(r"\b(january|february|march|april|may|june|july|august|september|october|"
                         r"november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\b", t):
        w = m.group(1)
        if w == "may" and not re.search(r"\b(in|of|during|for|from|to|through|until|and|between|since)\s+may\b|\bmay\s*(20\d\d|[,?.]|$)", t):
            continue
        ments.append((m.start(), MONTH_WORDS[w]))
    months = None
    if ments:
        nums = [n for _, n in ments]
        between = t[ments[0][0]:ments[-1][0]]
        is_range = bool(re.search(r"\b(to|through|until|thru)\b|[-\u2013]", between)) or (
            bool(re.search(r"\bbetween\b", t[:ments[0][0]])) and bool(re.search(r"\band\b", between)))
        if len(nums) >= 2 and is_range and nums[0] <= nums[-1]:
            nums = list(range(nums[0], nums[-1] + 1))
        months = {f"{year}-{n:02d}" for n in nums}
    q = re.search(r"\bq([1-4])\b|\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter", t)
    if q:
        k = int(q.group(1)) if q.group(1) else {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}[q.group(2)]
        months = {f"{year}-{n:02d}" for n in range(3 * k - 2, 3 * k + 1)}
    h = re.search(r"\b(h1|h2|first half|second half)\b", t)
    if h:
        rng = range(1, 7) if h.group(1) in ("h1", "first half") else range(7, 13)
        months = {f"{year}-{n:02d}" for n in rng}
    return months, (year if ym and not months else None)


def rule_answer(question: str):
    t = question.lower()
    df = DF
    months, year_only = _period(t)
    if months:
        df = df[df["month"].isin(months)]
    elif year_only:
        df = df[df["month"].str.startswith(str(year_only))]

    regions = [r for r in REGIONS if re.search(rf"\b{re.escape(r.lower())}\b", t)]
    products = [p for p in PRODUCTS if re.search(rf"\b{re.escape(p.lower())}s?\b", t)]
    custs = [c.upper() for c in re.findall(r"\bc\d{3,4}\b", t)]
    if regions:
        df = df[df["region"].isin(regions)]
    if products:
        df = df[df["product"].isin(products)]
    if custs:
        df = df[df["customer"].isin(custs)]

    status = "refunded" if "refund" in t else "void" if re.search(r"\bvoid|cancel", t) else "paid"
    count_intent = bool(re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", t))
    money_words = bool(re.search(r"amount|value|usd|\$|dollar|worth|total|revenue|sales|how much|spen[dt]|earn", t))
    units_words = bool(re.search(r"\bunits?\b|\bquantit|\bpieces\b|\bitems\b", t))
    sold_rank = bool(re.search(r"best-selling|most sold|selling|\bsold\b", t)) and not re.search(r"revenue|sales|earn|spen[dt]", t)

    # ---- ranking questions ("which product ...", "top 3 customers ...")
    t_rank = re.sub(r"\bat (least|most)\b", " ", t)
    # distinct-customer counts ("how many different customers ... at least one paid order ...")
    if re.search(r"\b(how many|number of|count of|total)\s+(distinct |unique |different |active |paying )?customers\b", t) or \
            re.search(r"\bcustomers?\b.*\b(distinct|unique|different)\b|\b(distinct|unique|different)\s+customers?\b", t):
        csub = df[df["status"] == status]
        am = re.search(r"\bat least\s+(\d+|" + "|".join(NUM_WORDS) + r")\b", t)
        if am:
            k = int(am.group(1)) if am.group(1).isdigit() else NUM_WORDS[am.group(1)]
            per = csub.groupby("customer")["id"].count()
            return int((per >= k).sum())
        return int(csub["customer"].nunique())
    rank_hi, rank_lo = re.search(RANK_HI, t_rank), re.search(RANK_LO, t_rank)
    dim = None
    wm = re.search(r"\b(which|what)\s+(?:\w+\s+){0,2}?(product|customer|region|month)\b", t)
    if wm:
        dim = wm.group(2)
    elif rank_hi or rank_lo:
        for d, pat in DIM_WORDS.items():
            if re.search(pat, t) and not ((d == "region" and regions) or (d == "product" and products)
                                          or (d == "customer" and custs) or (d == "month" and months)):
                dim = d
                break
    if dim and (rank_hi or rank_lo or wm):
        sub = df[df["status"] == status] if status != "paid" or True else df
        col = DIM_COL[dim]
        if "customers" in t and dim != "customer" and not re.search(r"revenue|sales", t):
            g = sub.groupby(col)["customer"].nunique()
        elif units_words or sold_rank:
            g = sub.groupby(col)["qty"].sum()
        elif status == "refunded" and not re.search(r"amount|value|usd|\$|worth|total refunded", t):
            g = sub.groupby(col)["id"].count()
        elif re.search(r"\borders?\b", t) and not money_words:
            g = sub.groupby(col)["id"].count()
        else:
            g = sub.groupby(col)["usd"].sum()
        n = 1
        nm = re.search(r"\b(?:top|bottom|best|worst|lowest|highest)\s+(\d+|" + "|".join(NUM_WORDS) + r")\b", t)
        if nm:
            n = int(nm.group(1)) if nm.group(1).isdigit() else NUM_WORDS[nm.group(1)]
        asc = bool(rank_lo) and not rank_hi
        items = sorted(g.items(), key=lambda kv: ((kv[1] if asc else -kv[1]), kv[0]))
        names = [k for k, _ in items[:n]]
        return names if n > 1 else (names[0] if names else None)

    # ---- scalar questions
    sub = df[df["status"] == status]
    if re.search(r"\b(how many|number of|count of|total)\s+(distinct |unique |different |active |paying )?customers\b", t) or \
            re.search(r"\bcustomers?\b.*\b(distinct|unique|different)\b|\b(distinct|unique|different)\s+customers?\b", t):
        return int(sub["customer"].nunique())
    if products and re.search(r"\b(buy|bought|purchase[ds]?|sold|sell)\b", t) and count_intent and not re.search(r"\borders?\b", t):
        return int(sub["qty"].sum())
    if re.search(r"\b(average|avg|mean)\b", t):
        return round(float(sub["usd"].mean()), 2)
    if units_words and not re.search(r"revenue|sales", t):
        return int(sub["qty"].sum())
    if count_intent and not (re.search(r"how much", t)):
        return int(len(sub))
    return round(float(sub["usd"].sum()), 2)

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
            json={"model": MODEL, "temperature": 0, "max_tokens": 700, "messages": messages},
        )
        r.raise_for_status()
        return strip_fences(r.json()["choices"][0]["message"]["content"])


LAST = {"error": None, "code": None, "fallback_used": False}


async def answer_question(question: str):
    start = time.monotonic()
    code, error = "", ""
    if AIPIPE_TOKEN:
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
                fatal = False
                if isinstance(e, httpx.HTTPStatusError):
                    error += " | " + e.response.text[:300]
                    fatal = e.response.status_code in (401, 402, 403, 429)
                LAST["error"] = error
                print("ERROR:", error, flush=True)
                if fatal:
                    break
    try:
        LAST["fallback_used"] = True
        return normalise(rule_answer(question))
    except Exception as e:
        LAST["error"] = f"fallback failed: {type(e).__name__}: {e}"
        return None


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def health():
    return {"ok": True, "orders": len(DF), "token_set": bool(AIPIPE_TOKEN),
            "model": MODEL, "last_error": LAST["error"], "last_code": LAST["code"],
            "fallback_used": LAST["fallback_used"]}


@app.post("/")
@app.post("/ask")
async def ask(request: Request):
    body = await request.json()
    question = str(body.get("question", "")).strip()
    return {"answer": await answer_question(question)}
