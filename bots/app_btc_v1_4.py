# bots/app_btc_v1_6.py
import os
import sys
import json
import threading
import time
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import ccxt
from flask import Flask, request, jsonify

# stdlib HTTP (no external "requests" dep)
import urllib.request
import urllib.error
import gc

# === ApeX API imports ===
from apexomni.constants import APEX_OMNI_HTTP_MAIN, NETWORKID_OMNI_MAIN_ARB
from apexomni.http_private_sign import HttpPrivateSign
from apexomni.http_public import HttpPublic

# --------- make stdout/stderr line-buffered so Render logs don't “freeze” ----------
try:
    sys.stdout.reconfigure(line_buffering=True)  # py>=3.7
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# also force prints to flush on each call
print = lambda *a, **k: (__import__("builtins").print)(*a, **{**k, "flush": True})

# ---------------- Precision ----------------
getcontext().prec = 28

# ---------------- App ----------------
app = Flask(__name__)
APP_NAME = "Apex Omni BTC Bot (v1.6)"
STARTED_AT_UTC = datetime.utcnow().isoformat() + "Z"

# ---------------- Timezone (UTC+10) ----------------
UTC_PLUS_10 = timedelta(hours=10)
def now_utc10_dt() -> datetime:
    return datetime.utcnow() + UTC_PLUS_10
def now() -> str:
    return now_utc10_dt().strftime("[%Y-%m-%d %H:%M:%S UTC+10]")

# ---------------- ANSI Colors (can disable with NO_COLOR=1) ----------------
NO_COLOR = os.environ.get("NO_COLOR") == "1"
CLR = {
    "reset": "" if NO_COLOR else "\033[0m",
    "dim": ""   if NO_COLOR else "\033[2m",
    "bold": ""  if NO_COLOR else "\033[1m",
    "green": "" if NO_COLOR else "\033[92m",
    "red": ""   if NO_COLOR else "\033[91m",
    "yellow": ""if NO_COLOR else "\033[93m",
    "blue": ""  if NO_COLOR else "\033[94m",
    "cyan": ""  if NO_COLOR else "\033[96m",
    "mag": ""   if NO_COLOR else "\033[95m",
    "gray": ""  if NO_COLOR else "\033[90m",
}
def colorize(msg: str, color: str) -> str:
    code = CLR.get(color, "")
    reset = CLR["reset"]
    return f"{code}{msg}{reset}" if code else msg

# ---------------- Config (BTC) ----------------
APEX_SYMBOL        = "BTC-USDT"
BINANCE_SYMBOL     = "BTC/USDT"
BOT_ID             = "BTC"

# Bias TF (1h)
BIAS_INTERVAL      = "1h"
EMA_PERIOD         = 50
ICT_EMA_SLOPE_BARS = 10
ICT_SWING_LOOKBACK = 5
ICT_BOS_BUFFER_PCT = 0.10
ICT_REQUIRE_BOS    = False
DEBUG_BIAS         = True

# Trade gates
USE_BIAS            = False     # trade only with bias when True
ALLOW_COUNTER_TREND = True      # allow against bias if True

# Exchange tick/steps (ApeX BTC)
TICK_SIZE       = Decimal("1")
SIZE_STEP       = Decimal("0.001")
MIN_ORDER_USDT  = Decimal("5")

# Capital Manager service (Render URL via env, fallback local)
CAPMGR_URL      = os.environ.get("CAPMGR_URL", "http://127.0.0.1:5015")

# Leverage & sizing
LEVERAGE        = Decimal(os.environ.get("LEVERAGE", "15"))
TRADE_SIZE_PCT  = Decimal(os.environ.get("TRADE_SIZE_PCT", "0.15"))   # % of allocation per entry

# Bias-aware TP/SL (fractions; 0.002 = 0.2%)
TREND_TP_PCT    = Decimal(os.environ.get("TREND_TP_PCT", "0.01"))
TREND_SL_PCT    = Decimal(os.environ.get("TREND_SL_PCT", "0.0075"))
CT_TP_PCT       = Decimal(os.environ.get("CT_TP_PCT", "0.0050"))
CT_SL_PCT       = Decimal(os.environ.get("CT_SL_PCT", "0.0050"))

# MF window (seconds)
MF_WAIT_SEC     = int(os.environ.get("MF_WAIT_SEC", "8000"))
MF_LEAD_SEC     = int(os.environ.get("MF_LEAD_SEC", "8000"))

# Optional 3rd confirmation webhook (provision)
ENABLE_THIRD_CONFIRMATION = os.environ.get("ENABLE_THIRD_CONFIRMATION", "0") == "1"

# Heartbeat logging (prevents “quiet logs look dead”)
DASH_HEARTBEAT_SEC = int(os.environ.get("DASH_HEARTBEAT_SEC", "30"))

# External log mirroring (optional)
LOG_WEBHOOK = os.environ.get("LOG_WEBHOOK")  # POST JSON to this URL if set

# Bias loop cadence (tunable to reduce memory/CPU)
BIAS_LOOP_SEC = int(os.environ.get("BIAS_LOOP_SEC", "60"))

# ---------------- Credentials (env-first; fallback to your current keys) ----------------
api_creds = {
    "key":        os.environ.get("APEX_API_KEY",        "3e965beb-41e2-f125-a7c5-569f45bfba21"),
    "secret":     os.environ.get("APEX_API_SECRET",     "NXtuAyq4hS9G4fVlytQtQGn9Qk5LUukXGAYg8SBj"),
    "passphrase": os.environ.get("APEX_API_PASSPHRASE", "GEy6yNGaZ5_0fuX4VBJ3"),
}
zk_seeds = os.environ.get("ZK_SEEDS", "0xd00ec9396facbafc423b5d92a289ea49adfdb0b918d3d5db26edbb978893ed5d0bd48c3fbe4309de2a09a3514cbec6d2c4012df85653a1421aca1cf599acda491c")
zk_l2Key = os.environ.get("ZK_L2KEY", "0xd6094a658c50dccf9be8f85cde1804e92a74a0482788766c9b8744cbc6fe8501")

# ---------------- ApeX Clients ----------------
client = HttpPrivateSign(
    APEX_OMNI_HTTP_MAIN,
    network_id=NETWORKID_OMNI_MAIN_ARB,
    api_key_credentials=api_creds,
    zk_seeds=zk_seeds,
    zk_l2Key=zk_l2Key
)
client.configs_v3()
http_public = HttpPublic(APEX_OMNI_HTTP_MAIN)

# ---------------- Safe Decimal + Rounding ----------------
def _D(x) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))

def _D_safe(x):
    try:
        if x is None: return None
        s = str(x).strip()
        if s == "" or s.lower() in ("none","nan","inf","-inf"): return None
        return Decimal(s)
    except Exception:
        return None

def _decimals_for_step(step: Decimal) -> int:
    t = _D(step).as_tuple()
    return max(0, -t.exponent)

def fmt_fixed(value: Decimal, step: Decimal) -> str:
    d = _decimals_for_step(step)
    if d == 0:
        return f"{_D(value).to_integral_value(rounding=ROUND_DOWN):f}"
    q = _D(value).quantize(Decimal((0,(1,),-d)))
    s = f"{q:f}"
    if "." in s:
        cur = len(s.split(".")[1])
        if cur < d:
            s += "0" * (d - cur)
    else:
        s += "." + ("0" * d)
    return s

def floor_to_tick(price: Decimal) -> Decimal:
    return (_D(price) / TICK_SIZE).to_integral_value(rounding=ROUND_DOWN) * TICK_SIZE

def ceil_to_tick(price: Decimal) -> Decimal:
    return (_D(price) / TICK_SIZE).to_integral_value(rounding=ROUND_UP) * TICK_SIZE

def ceil_to_step(size: Decimal) -> Decimal:
    s2 = (_D(size) / SIZE_STEP).to_integral_value(rounding=ROUND_UP) * SIZE_STEP
    return s2 if s2 >= SIZE_STEP else SIZE_STEP

def ensure_min_notional(size: Decimal, price: Decimal) -> Decimal:
    p = _D(price)
    if p <= 0: return Decimal("0")
    target = MIN_ORDER_USDT / p
    return ceil_to_step(max(_D(size), target))

def fmt_price(x: Decimal) -> str: return fmt_fixed(_D(x), TICK_SIZE)
def fmt_size(x: Decimal)  -> str: return fmt_fixed(_D(x), SIZE_STEP)

# ---------------- Terminal states ----------------
TERMINAL_STATES = {
    "FILLED", "TRIGGERED", "EXECUTED", "COMPLETED", "DONE",
    "CANCELED", "CANCELLED"
}

# ---------------- ApeX circuit breaker ----------------
API_MAX_TRIES = 4
API_COOLDOWN_SEC = 30
_APEX_CB = {"failures": 0, "open_until": 0.0}

def _apex_circuit_open() -> bool:
    return time.time() < _APEX_CB["open_until"]

def _note_apex_failure():
    _APEX_CB["failures"] += 1
    if _APEX_CB["failures"] >= API_MAX_TRIES:
        _APEX_CB["open_until"] = time.time() + API_COOLDOWN_SEC
        _APEX_CB["failures"] = 0
# ---------------- External log mirroring helpers ----------------
def _post_json_quick(url: str, payload: dict, timeout: float = 1.2):
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            _ = r.read(0)  # fire-and-forget
    except Exception:
        pass

def _async(fn, *args, **kwargs):
    threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()

def _mirror_to_webhook(event: str, msg: str, extra: Optional[dict]):
    if not LOG_WEBHOOK:
        return
    payload = {
        "app": APP_NAME,
        "ts": int(time.time()),
        "event": event,
        "message": msg,
        "extra": extra or {},
    }
    _async(_post_json_quick, LOG_WEBHOOK, payload)

# ---------------- Dashboard ----------------
DASH_LAST = {"bias": None, "connected": False}
HEARTBEAT_SEQ = 0
LAST_HEARTBEAT_TS = 0

def dash(event: str, msg: str, *, extra: Optional[dict] = None):
    icons = {"start":"🚀","ok":"✅","warn":"⚠️","error":"❌","signal":"🔔","state":"🖥️","trade":"📈","debug":"🔎"}
    colors = {"start":"cyan","ok":"green","warn":"yellow","error":"red","signal":"mag","state":"blue","trade":"green","debug":"gray"}
    icon = icons.get(event, "•"); col = colors.get(event, "reset")
    payload = f"{now()} {icon} {msg}"
    if extra:
        dim_l = CLR['dim']; dim_r = CLR['reset'] if CLR['dim'] else ""
        payload += f" {dim_l}{extra}{dim_r}" if dim_l else f" {extra}"
    print(colorize(payload, col))
    _mirror_to_webhook(event, msg, extra)

def dash_startup():
    if not DASH_LAST["connected"]:
        DASH_LAST["connected"] = True
        dash("start", f"{APP_NAME} starting")
        dash("ok", "ApeX client initialized", extra={"endpoint": APEX_OMNI_HTTP_MAIN})
        dash("state", f"USE_BIAS={USE_BIAS}, ALLOW_COUNTER_TREND={ALLOW_COUNTER_TREND}")
        dash("state", "Waiting for signals: MF + TREND", extra={"lead_sec": MF_LEAD_SEC, "wait_sec": MF_WAIT_SEC})

def dash_bias(new_bias: Optional[str]):
    if DASH_LAST["bias"] != new_bias:
        DASH_LAST["bias"] = new_bias
        human = new_bias if new_bias else "NEUTRAL"
        color = "green" if new_bias == "LONG" else "red" if new_bias == "SHORT" else "yellow"
        print(colorize(f"{now()} 🧭 ICT Bias → {human}", color))
        _mirror_to_webhook("state", f"ICT Bias → {human}", {"bias": new_bias})

# ---------------- Accounts & Price (with resilient fallbacks) ----------------
def get_usdt_contract_balance() -> Decimal:
    try:
        if _apex_circuit_open():
            raise RuntimeError("APEX circuit open; skipping account read temporarily")
        last_err = None
        for i in range(API_MAX_TRIES):
            try:
                acct = client.get_account_v3()
                for w in acct.get("contractWallets", []):
                    if w.get("token") == "USDT":
                        return Decimal(str(w.get("balance", "0")))
                return Decimal("0")
            except Exception as e:
                last_err = e
                msg = str(e)
                dash("warn", f"get_usdt_contract_balance error: {msg}")
                if any(k in msg.lower() for k in ["timeout","timed out","dns","name or service","temporary failure","max retries"]):
                    time.sleep(min(0.5 * (2 ** i), 6.0)); continue
                break
        _note_apex_failure()
        if last_err: raise last_err
    except Exception as e:
        dash("warn", f"get_usdt_contract_balance final error: {e}")
    return Decimal("0")

def get_open_position() -> Optional[dict]:
    try:
        if _apex_circuit_open():
            raise RuntimeError("APEX circuit open; skipping get_open_position temporarily")
        last_err = None
        for i in range(API_MAX_TRIES):
            try:
                acct = client.get_account_v3()
                for p in acct.get("positions", []):
                    if p.get("symbol") == APEX_SYMBOL and Decimal(str(p.get("size", "0"))) != Decimal("0"):
                        return p
                return None
            except Exception as e:
                last_err = e
                msg = str(e)
                dash("warn", f"get_open_position error: {msg}")
                if any(k in msg.lower() for k in ["timeout","timed out","dns","name or service","temporary failure","max retries"]):
                    time.sleep(min(0.5 * (2 ** i), 6.0)); continue
                break
        _note_apex_failure()
        if last_err: raise last_err
    except Exception as e:
        dash("warn", f"get_open_position final error: {e}")
    return None

def _binance_ticker_price(symbol_ccxt: str) -> Optional[Decimal]:
    try:
        ex = ccxt.binance({"timeout": 20000, "enableRateLimit": True})
        t = ex.fetch_ticker(symbol_ccxt)
        for k in ("info", "last", "close"):
            v = t.get(k)
            if v is None: continue
            try:
                return _D_safe(v) if k == "info" else _D(v)
            except Exception:
                continue
    except Exception:
        pass
    return None

def get_public_price() -> Decimal:
    try:
        t = http_public.ticker_v3(symbol=APEX_SYMBOL)
        row = (t.get("data") or [{}])[0]
        for key in ("markPrice","indexPrice","lastPrice"):
            v = _D_safe(row.get(key))
            if v is not None and v > 0:
                return v
    except Exception as e:
        dash("warn", f"http_public.ticker_v3 error: {e}")

    vb = _binance_ticker_price(BINANCE_SYMBOL)
    if vb and vb > 0:
        dash("state", "Price fallback used (Binance)")
        return vb

    raise ValueError("No valid mark/index/last price (ApeX) and Binance fallback failed")

# ---------------- Bias via Binance (1h) — hardened & memory-friendly ----------------
BIAS: Optional[str] = None
BIAS_STALE_TTL_SEC = 1800
_LAST_BIAS_TS = 0

BINANCE_HOSTS = ["api.binance.com", "api1.binance.com", "api2.binance.com", "api3.binance.com"]
_BINANCE_IDX = 0
_BINANCE = None
_LAST_MARKETS_LOAD = 0

def _binance_client():
    global _BINANCE, _LAST_MARKETS_LOAD
    if _BINANCE is None:
        _BINANCE = ccxt.binance({
            "timeout": 25000,
            "enableRateLimit": True,
            "options": {"adjustForTimeDifference": True},
            "hostname": BINANCE_HOSTS[_BINANCE_IDX],
        })
        try:
            _BINANCE.load_markets(reload=False)
            _LAST_MARKETS_LOAD = time.time()
        except Exception as e:
            dash("warn", f"binance load_markets error: {e}")
    return _BINANCE

def _rotate_binance_host():
    global _BINANCE_IDX, _BINANCE
    _BINANCE_IDX = (_BINANCE_IDX + 1) % len(BINANCE_HOSTS)
    _BINANCE = None

def fetch_binance_candles(symbol: str, interval: str, limit: int = 400) -> Optional[pd.DataFrame]:
    # limit trimmed for memory; still enough for EMA + swing logic
    backoff = 0.4
    for attempt in range(5):
        try:
            ex = _binance_client()
            global _LAST_MARKETS_LOAD
            if time.time() - _LAST_MARKETS_LOAD > 7200:
                ex.load_markets(reload=True)
                _LAST_MARKETS_LOAD = time.time()
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
            if not ohlcv:
                return None
            df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
            return df
        except Exception as e:
            err = str(e)
            dash("warn", f"fetch_binance_candles error: {err}")
            if any(k in err.lower() for k in ["timeout","timed out","temporary failure","name or service","dns","exchangeinfo","429"]):
                if attempt < 4:
                    _rotate_binance_host()
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
                    continue
            return None
    return None

def compute_ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
    df = df.copy()
    df["ema"] = df["close"].ewm(span=period, adjust=False).mean()
    return df

def compute_ict_bias_from_candles(df: pd.DataFrame) -> Optional[str]:
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    df = compute_ema(df, EMA_PERIOD)
    emas   = df["ema"].values

    lb = int(ICT_SWING_LOOKBACK)
    swh = [False] * len(df)
    swl = [False] * len(df)
    for i in range(lb, len(df) - lb):
        if highs[i] > max(highs[i - lb:i]) and highs[i] >= max(highs[i + 1:i + 1 + lb]):
            swh[i] = True
        if lows[i] < min(lows[i - lb:i]) and lows[i] <= min(lows[i + 1:i + 1 + lb]):
            swl[i] = True

    buffer_frac = float(ICT_BOS_BUFFER_PCT) / 100.0
    bos_events = []
    for i, v in enumerate(swh):
        if v:
            level = highs[i]; thresh = level * (1.0 + buffer_frac)
            j = next((k for k in range(i + 1, len(df)) if closes[k] > thresh), None)
            if j is not None: bos_events.append((j, "UP"))
    for i, v in enumerate(swl):
        if v:
            level = lows[i]; thresh = level * (1.0 - buffer_frac)
            j = next((k for k in range(i + 1, len(df)) if closes[k] < thresh), None)
            if j is not None: bos_events.append((j, "DOWN"))

    last_bos_dir = None
    if bos_events:
        _, last_bos_dir = max(bos_events, key=lambda x: x[0])

    ema_up = emas[-1] > emas[-1 - ICT_EMA_SLOPE_BARS]
    ema_down = emas[-1] < emas[-1 - ICT_EMA_SLOPE_BARS]
    price_above = closes[-1] > emas[-1]
    price_below = closes[-1] < emas[-1]

    if last_bos_dir == "UP" and ema_up and price_above:
        return "LONG"
    if last_bos_dir == "DOWN" and ema_down and price_below:
        return "SHORT"

    if not ICT_REQUIRE_BOS:
        if ema_up and price_above:   return "LONG"
        if ema_down and price_below: return "SHORT"

    return None

def compute_bias():
    """Hardened bias updater with sticky fallback."""
    global BIAS, _LAST_BIAS_TS
    try:
        limit = max(EMA_PERIOD + 120, 250)
        df = fetch_binance_candles(BINANCE_SYMBOL, BIAS_INTERVAL, limit=limit)
        if df is None or df.empty or len(df) < EMA_PERIOD + ICT_EMA_SLOPE_BARS + 10:
            age = int(time.time()) - _LAST_BIAS_TS
            if BIAS is not None and age <= BIAS_STALE_TTL_SEC:
                if DEBUG_BIAS:
                    dash("debug", "ICT Bias: data hiccup — keeping previous", extra={"bias": BIAS, "age_s": age})
                return
            if BIAS is not None:
                BIAS = None
                dash_bias(BIAS)
            if DEBUG_BIAS:
                dash("debug", "ICT Bias: insufficient data")
            return

        decided = compute_ict_bias_from_candles(df)
        del df  # free memory
        gc.collect()

        if decided != BIAS:
            BIAS = decided
            dash_bias(BIAS)
        _LAST_BIAS_TS = int(time.time())

        if DEBUG_BIAS:
            dash("debug", f"Bias calc done", extra={"bias": BIAS})
    except Exception as e:
        age = int(time.time()) - _LAST_BIAS_TS
        if BIAS is not None and age <= BIAS_STALE_TTL_SEC:
            dash("warn", f"Bias error; keeping previous bias (age {age}s): {e}")
            return
        if BIAS is not None:
            BIAS = None
            dash_bias(BIAS)
        dash("warn", f"ICT Bias error: {e}")

# ---------------- Capital Manager HTTP helpers ----------------
def _http_get_json(url: str, timeout: float = 2.0) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None

def _http_post_json(url: str, payload: dict, timeout: float = 3.0) -> Optional[dict]:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None

def ping_capital_manager():
    dash("state", "Pinging Capital Manager…", extra={"url": CAPMGR_URL})
    h = _http_get_json(f"{CAPMGR_URL}/health")
    if h and h.get("ok"):
        dash("ok", "Capital Manager reachable", extra={"url": CAPMGR_URL})
    else:
        dash("warn", "Capital Manager not reachable (bot will still run)")

def dash_capital_status_once():
    h = _http_get_json(f"{CAPMGR_URL}/health")
    if not (h and h.get("ok")):
        return
    bot_row = None
    for b in (h.get("bots") or []):
        if str(b.get("bot_id")).upper() == BOT_ID:
            bot_row = b; break
    extra = {
        "wallet_usdt": h.get("wallet_usdt"),
        "bot": BOT_ID,
        "symbol": APEX_SYMBOL,
        "initial_trade_pct": str(TRADE_SIZE_PCT)
    }
    if bot_row:
        extra.update({
            "weight": bot_row.get("weight"),
            "allocated_usdt": bot_row.get("allocated_usdt"),
            "reserved_usdt": bot_row.get("reserved_usdt"),
        })
    dash("ok", "Capital Manager status", extra=extra)

# ---------------- Runtime State ----------------
STATE: Dict[str, Any] = {
    "running": True,
    "last_mf": None,           # {'dir':'UP'|'DOWN','ts':int}
    "last_trend": None,        # {'dir':'UP'|'DOWN','ts':int}
    "last_confirm": None,      # {'ts':int}
    "mf_flip_since_entry": False,
}
POSITION: Dict[str, Any] = {
    "open": False, "side": None, "entry": None, "size": None,
    "order_id": None, "tp_id": None, "sl_id": None, "tp": None, "sl": None,
    "margin": None, "reserved_margin": None,
    "vector_close_timestamp": None,
    "vector_side": None,
}
POSITION_LOCK = threading.Lock()

def _normalize_apex_side(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = str(raw).upper()
    if s in ("BUY", "LONG", "B"):
        return "LONG"
    if s in ("SELL", "SHORT", "S"):
        return "SHORT"
    return None

def _extract_entry_price_from_position(p: dict) -> Optional[Decimal]:
    for k in [
        "avgEntryPrice", "entryPrice", "averageEntryPrice", "avgPrice",
        "openPrice", "averageOpenPrice", "fillPrice", "lastPrice", "markPrice",
    ]:
        v = _D_safe(p.get(k))
        if v and v > 0:
            return v
    return None
# ---------------- Terminal state helpers ----------------
def _ord_status(info: dict) -> str:
    return str((info or {}).get("status", "")).upper()

def _fetch_order(order_id: str) -> dict:
    if not order_id: return {}
    try:
        if _apex_circuit_open():
            raise RuntimeError("APEX circuit open; skipping get_order_v3 temporarily")
        last_err = None
        for i in range(API_MAX_TRIES):
            try:
                return client.get_order_v3(symbol=APEX_SYMBOL, orderId=str(order_id)).get("data") or {}
            except Exception as e:
                last_err = e
                msg = str(e)
                dash("warn", "get_order_v3 error", extra={"id": order_id, "err": msg})
                if any(k in msg.lower() for k in ["timeout","timed out","dns","name or service","temporary failure","max retries"]):
                    time.sleep(min(0.5 * (2 ** i), 6.0)); continue
                break
        _note_apex_failure()
        if last_err: raise last_err
    except Exception as e:
        dash("warn", "get_order_v3 final error", extra={"id": order_id, "err": str(e)})
        return {}
    return {}

def _cancel_id(order_id: str, label: str):
    if not order_id: return
    try:
        client.delete_order_v3(id=str(order_id))
        dash("state", f"Canceled {label} via delete_order_v3", extra={"id": order_id})
        return
    except Exception:
        pass
    try:
        client.cancel_order_v3(symbol=APEX_SYMBOL, orderId=str(order_id))
        dash("state", f"Canceled {label} via cancel_order_v3", extra={"id": order_id})
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("not found","filled","triggered","conflict")):
            dash("state", f"{label} already terminal", extra={"id": order_id})
        else:
            dash("warn", f"cancel {label} error", extra={"id": order_id, "err": str(e)})

def _full_reset(reason: str):
    with POSITION_LOCK:
        tp_id = POSITION.get("tp_id"); sl_id = POSITION.get("sl_id")
    if tp_id: _cancel_id(tp_id, "TP")
    if sl_id: _cancel_id(sl_id, "SL")

    capmgr_on_close()

    with POSITION_LOCK:
        POSITION.update({
            "open": False, "side": None, "entry": None, "size": None,
            "order_id": None, "tp_id": None, "sl_id": None, "tp": None, "sl": None,
            "margin": None, "reserved_margin": None,
            "vector_close_timestamp": None, "vector_side": None
        })
        STATE["last_mf"] = None
        STATE["last_trend"] = None
        STATE["last_confirm"] = None
        STATE["mf_flip_since_entry"] = False

    dash("trade", f"EXIT by {reason} — ready for next entry")

# ---------------- TP/SL utils (bias-aware selection) ----------------
def pick_tp_sl_pcts(entry_side: str) -> Tuple[Decimal, Decimal]:
    if BIAS is None:
        return TREND_TP_PCT, TREND_SL_PCT
    return ((TREND_TP_PCT, TREND_SL_PCT) if entry_side == BIAS else (CT_TP_PCT, CT_SL_PCT))

def tp_price_from(entry: Decimal, side: str, pct: Decimal) -> Decimal:
    entry = _D(entry); pct = _D(pct)
    return entry * (Decimal("1")+pct) if side=="LONG" else entry * (Decimal("1")-pct)

def sl_price_from(entry: Decimal, side: str, pct: Decimal) -> Decimal:
    entry = _D(entry); pct = _D(pct)
    return entry * (Decimal("1")-pct) if side=="LONG" else entry * (Decimal("1")+pct)

# ---------------- MARKET Entry (+ reduce-only TP/SL) ----------------
def place_market_order(direction: str, trade_size_pct: Optional[Decimal] = None) -> Optional[str]:
    try:
        direction = str(direction).upper()
        if direction not in ("LONG","SHORT"):
            dash("error", f"Invalid direction: {direction}")
            return None

        # Double-check live position gate
        live = get_open_position()
        if live:
            dash("warn", "Live position exists — aborting duplicate order.",
                 extra={"size": str(live.get("size")), "side": live.get("side")})
            return None

        # Reserve from Capital Manager
        t_pct = TRADE_SIZE_PCT if trade_size_pct is None else (_D_safe(trade_size_pct) or TRADE_SIZE_PCT)
        r = capmgr_reserve(t_pct)
        if not r or not r.get("ok"):
            dash("error", "Capital reservation failed", extra={"resp": r})
            return None

        approved = _D(r.get("approved_margin_usdt", "0"))
        allocated = _D(r.get("allocated_usdt", "0"))
        available = _D(r.get("available_usdt", "0"))
        dash("state", "Capital reservation",
             extra={"trade_pct": str(t_pct), "approved_margin": str(approved),
                    "allocated_usdt": str(allocated), "available_usdt": str(available)})

        if approved <= 0:
            dash("warn", "No capital available for this entry", extra={"trade_pct": str(t_pct)})
            return None

        # Price & rounding
        mark_price = get_public_price()
        mark_price_r = floor_to_tick(mark_price)  # BTC tick=1
        price_str   = fmt_price(mark_price_r)

        # Size from (approved_margin * leverage) / price
        raw_size = (approved * LEVERAGE) / mark_price_r
        size     = ensure_min_notional(ceil_to_step(raw_size), mark_price_r)
        if size < SIZE_STEP:
            dash("error", f"Size too small after rounding: {size}. Releasing reservation.")
            capmgr_release(approved)
            return None
        size_str = fmt_size(size)

        # Compute used margin from rounded size
        used_margin = (size * mark_price_r) / LEVERAGE
        unused = approved - used_margin
        if unused > Decimal("0.0001"):
            capmgr_release(unused)
            dash("state", "Released unused reserved margin", extra={"unused": str(unused), "used": str(used_margin)})

        side = "BUY" if direction == "LONG" else "SELL"

        # ENTRY
        resp = client.create_order_v3(
            symbol=APEX_SYMBOL, side=side, type="MARKET",
            size=size_str, price=price_str, timestampSeconds=int(time.time())
        )
        order_id = (resp.get("data") or {}).get("id")
        if not order_id:
            dash("error", "Entry rejected by exchange, releasing used_margin", extra={"resp": resp})
            capmgr_release(used_margin)
            return None

        # Bias-aware TP/SL
        tp_pct, sl_pct = pick_tp_sl_pcts(direction)
        raw_tp = tp_price_from(mark_price_r, direction, tp_pct)
        raw_sl = sl_price_from(mark_price_r, direction, sl_pct)

        if direction == "LONG":
            tp_price = floor_to_tick(raw_tp)
            sl_trig  = floor_to_tick(raw_sl)
            sl_exec  = floor_to_tick(sl_trig * Decimal("0.999"))
            tp_side  = sl_side = "SELL"
        else:
            tp_price = ceil_to_tick(raw_tp)
            sl_trig  = ceil_to_tick(raw_sl)
            sl_exec  = ceil_to_tick(sl_trig * Decimal("1.001"))
            tp_side  = sl_side = "BUY"

        tp_str      = fmt_price(tp_price)
        sl_trig_str = fmt_price(sl_trig)
        sl_exec_str = fmt_price(sl_exec)

        # TP (TAKE_PROFIT_MARKET)
        tp_order = client.create_order_v3(
            symbol=APEX_SYMBOL, side=tp_side, type="TAKE_PROFIT_MARKET",
            triggerPrice=tp_str, price=tp_str, size=size_str,
            reduceOnly=True, timestampSeconds=int(time.time())
        )
        tp_id = (tp_order.get("data") or {}).get("id")

        # SL (STOP_MARKET)
        sl_order = client.create_order_v3(
            symbol=APEX_SYMBOL, side=sl_side, type="STOP_MARKET",
            triggerPrice=sl_trig_str, price=sl_exec_str, size=size_str,
            reduceOnly=True, timestampSeconds=int(time.time())
        )
        sl_id = (sl_order.get("data") or {}).get("id")

        with POSITION_LOCK:
            POSITION.update({
                "open": True, "side": direction, "entry": mark_price_r, "size": size,
                "order_id": order_id, "tp_id": tp_id, "sl_id": sl_id,
                "tp": tp_price, "sl": sl_exec, "margin": used_margin, "reserved_margin": used_margin
            })
        STATE["mf_flip_since_entry"] = False

        dash("trade", f"ENTRY {direction} @ {price_str}",
             extra={"size": size_str, "tp": tp_str, "sl": sl_exec_str, "order_id": order_id})
        return order_id

    except Exception as e:
        dash("error", f"Order failed", extra={"err": repr(e)})
        with POSITION_LOCK:
            res = POSITION.get("reserved_margin")
            if res:
                capmgr_release(_D(res))
                POSITION["reserved_margin"] = None
        return None

# ---------------- Latching / Bias gate ----------------
def _within_window(ts_a: int, ts_b: int) -> bool:
    dt = ts_b - ts_a
    return (-MF_LEAD_SEC <= dt <= MF_WAIT_SEC)

def _bias_gate(desired: str) -> bool:
    if not USE_BIAS:
        return True
    if BIAS is None:
        return False
    if desired == BIAS:
        return True
    return ALLOW_COUNTER_TREND

def check_and_latch(desired_side: Optional[str]) -> Optional[str]:
    if desired_side not in ("LONG","SHORT"):
        return None
    if not _bias_gate(desired_side):
        dash("state", f"Bias gate blocked {desired_side}", extra={"bias": BIAS})
        return None

    l_mf, l_tr = STATE["last_mf"], STATE["last_trend"]
    if not l_mf or not l_tr:
        return None

    need = ("UP" if desired_side=="LONG" else "DOWN")
    if l_mf["dir"] != need or l_tr["dir"] != need:
        return None
    if not _within_window(l_mf["ts"], l_tr["ts"]):
        return None

    if ENABLE_THIRD_CONFIRMATION and not STATE["last_confirm"]:
        return None
    return desired_side

# ---------------- Instant exit on TREND flip + re-latch support ----------------
def close_position_market(reason: str) -> bool:
    """Instant reduce-only MARKET close of the current position, cancel TP/SL, then clean reset."""
    try:
        with POSITION_LOCK:
            if not POSITION["open"]:
                return False
            cur_side = POSITION.get("side")
            size     = POSITION.get("size")
            tp_id    = POSITION.get("tp_id")
            sl_id    = POSITION.get("sl_id")

        if tp_id: _cancel_id(tp_id, "TP")
        if sl_id: _cancel_id(sl_id, "SL")

        mark = get_public_price()
        price_str = fmt_price(floor_to_tick(mark))
        size_str  = fmt_size(size)
        opp = "SELL" if cur_side == "LONG" else "BUY"

        resp = client.create_order_v3(
            symbol=APEX_SYMBOL, side=opp, type="MARKET",
            size=size_str, price=price_str, reduceOnly=True,
            timestampSeconds=int(time.time())
        )
        dash("trade", f"EXIT MARKET by {reason}",
             extra={"closed_side": cur_side, "size": size_str, "hint_price": price_str, "resp": (resp.get("data") or {})})
    except Exception as e:
        dash("warn", f"close_position_market error: {e}")
    finally:
        _full_reset(reason)
    return True

def _attempt_latch_after_close(desired_side: str):
    try:
        time.sleep(0.25)
        side = check_and_latch(desired_side)
        if side:
            place_market_order(side)
    except Exception as e:
        dash("warn", f"_attempt_latch_after_close error: {e}")

# ---------------- Reconciliation (fixes missed TP/SL and cold start drift) ----------------
def _extract_pos_fields(p: dict):
    side = _normalize_apex_side(p.get("side"))
    size = _D_safe(p.get("size"))
    entry = _extract_entry_price_from_position(p)
    return side, size, entry

def reconcile_with_exchange():
    """If exchange says FLAT → hard reset. If exchange has a live pos → rehydrate local POSITION."""
    try:
        pos = get_open_position()
        if pos:
            side, size, entry = _extract_pos_fields(pos)
            if size and size > 0:
                with POSITION_LOCK:
                    was_open = POSITION["open"]
                    POSITION["open"] = True
                    if side:  POSITION["side"]  = side
                    if size:  POSITION["size"]  = size
                    if entry: POSITION["entry"] = entry
                if not was_open:
                    dash("state", "Rehydrated live position from exchange",
                         extra={"side": POSITION.get("side"), "size": str(POSITION.get("size")), "entry": str(POSITION.get("entry"))})
                return
        # Exchange flat:
        with POSITION_LOCK:
            was_open = POSITION["open"]
        if was_open:
            dash("trade", "Exchange flat detected — syncing local state")
            _full_reset("EXCHANGE_FLAT")
    except Exception as e:
        dash("warn", f"reconcile_with_exchange error: {e}")

# ---------------- Webhook helpers ----------------
def _extract_message_and_ts():
    try:
        data = request.get_json(force=False, silent=True) or {}
    except Exception:
        data = {}
    msg = data.get("message")
    ts = data.get("ts")
    if not msg:
        msg = request.form.get("message")
    if ts is None:
        ts = request.form.get("ts")
    if not msg:
        raw = (request.data or b"").decode("utf-8", "ignore").strip()
        if raw:
            msg = raw
    try:
        ts = int(ts) if ts is not None else int(time.time())
    except Exception:
        ts = int(time.time())
    return (msg or "").strip(), ts

def parse_up_down(msg: str) -> Optional[str]:
    s = str(msg or "").upper()
    if "UP" in s: return "UP"
    if "DOWN" in s: return "DOWN"
    return None

# ---------------- Webhooks (async, tolerant, always 200) ----------------
@app.route("/webhook_mf", methods=["POST","GET"], strict_slashes=False)
def webhook_mf():
    if request.method == "GET":
        return jsonify({"ok": True, "hint": 'POST {"message":"MF UP|MF DOWN","ts":<epoch optional>}'})
    try:
        msg, ts = _extract_message_and_ts()
        d = parse_up_down(msg)
        if d not in ("UP","DOWN"):
            return jsonify({"ok": False, "reason": "message must contain MF UP or MF DOWN"}), 200

        STATE["last_mf"] = {"dir": d, "ts": ts}
        dash("signal", f"MF {d}", extra={"ts": ts, "raw": msg})

        if POSITION["open"]:
            if POSITION["side"] == "LONG" and d == "DOWN": STATE["mf_flip_since_entry"] = True
            if POSITION["side"] == "SHORT" and d == "UP":  STATE["mf_flip_since_entry"] = True

        desired = "LONG" if d=="UP" else "SHORT"
        if not POSITION["open"]:
            side = check_and_latch(desired)
            if side:
                _async(place_market_order, side)
                return jsonify({"ok": True, "action": "entry_queued", "side": side, "ts": ts}), 200

        return jsonify({"ok": True, "latched": False, "ts": ts}), 200
    except Exception as e:
        dash("warn", f"webhook_mf handler error: {e}")
        return jsonify({"ok": False, "error": "handler_exception"}), 200

@app.route("/webhook_trend", methods=["POST","GET"], strict_slashes=False)
def webhook_trend():
    if request.method == "GET":
        return jsonify({"ok": True, "hint": 'POST {"message":"TREND UP|TREND DOWN","ts":<epoch optional>}'})
    try:
        msg, ts = _extract_message_and_ts()
        s = (msg or "").upper()
        if   "UP"   in s: d = "UP"
        elif "DOWN" in s: d = "DOWN"
        else:
            return jsonify({"ok": False, "reason": "message must contain TREND UP or TREND DOWN"}), 200

        STATE["last_trend"] = {"dir": d, "ts": ts}
        dash("signal", f"TREND {d}", extra={"ts": ts, "raw": msg})

        desired = "LONG" if d == "UP" else "SHORT"

        with POSITION_LOCK:
            open_ = POSITION["open"]
            cur_side = POSITION.get("side")

        if open_ and cur_side and cur_side != desired:
            dash("state", "TREND flip detected — closing current position now",
                 extra={"from": cur_side, "to": desired})
            _async(close_position_market, "TREND_FLIP")
            _async(_attempt_latch_after_close, desired)
            return jsonify({"ok": True, "flip_exit": True, "to": desired, "ts": ts}), 200

        if not open_:
            side = check_and_latch(desired)
            if side:
                _async(place_market_order, side)
                return jsonify({"ok": True, "action": "entry_queued", "side": side, "ts": ts}), 200

        return jsonify({"ok": True, "latched": False, "ts": ts}), 200
    except Exception as e:
        dash("warn", f"webhook_trend handler error: {e}")
        return jsonify({"ok": False, "error": "handler_exception"}), 200

@app.route("/webhook_confirm", methods=["POST","GET"], strict_slashes=False)
def webhook_confirm():
    if request.method == "GET":
        return jsonify({"ok": True, "hint": 'POST {"message":"CONFIRM","ts":<epoch optional>}'})
    try:
        _msg, ts = _extract_message_and_ts()
        STATE["last_confirm"] = {"ts": ts}
        dash("signal", "3rd CONFIRM", extra={"ts": ts})
        return jsonify({"ok": True, "confirm": True, "ts": ts}), 200
    except Exception as e:
        dash("warn", f"webhook_confirm handler error: {e}")
        return jsonify({"ok": False, "error": "handler_exception"}), 200

# ---------------- Force Test Entry ----------------
@app.route("/test/force_entry", methods=["POST","GET"], strict_slashes=False)
def test_force_entry():
    if request.method == "GET":
        return jsonify({"ok": True, "hint": 'POST {"side":"LONG|SHORT","trade_pct":0.10}'})
    data = request.get_json(silent=True) or {}
    side = str(data.get("side","")).upper()
    trade_pct = _D_safe(data.get("trade_pct"))
    if side not in ("LONG","SHORT"):
        return jsonify({"ok": False, "reason": "side must be LONG or SHORT"}), 200
    t_pct = trade_pct if trade_pct is not None else TRADE_SIZE_PCT
    dash("state", "FORCE ENTRY invoked", extra={"side": side, "trade_pct": str(t_pct)})
    _async(place_market_order, side, t_pct)
    return jsonify({"ok": True, "queued": True, "side": side, "trade_pct": str(t_pct)}), 200
# ---------------- TP/SL watchdog (explicit close + clean reset) ----------------
def tp_sl_watchdog():
    while STATE["running"]:
        try:
            with POSITION_LOCK:
                open_ = POSITION["open"]
                tp_id = POSITION.get("tp_id")
                sl_id = POSITION.get("sl_id")
                side  = POSITION.get("side")
                size  = POSITION.get("size")
                entry = POSITION.get("entry")
            if not open_:
                time.sleep(1); continue

            if tp_id:
                info = _fetch_order(tp_id)
                if _ord_status(info) in TERMINAL_STATES:
                    dash("trade", "TP TERMINAL", extra={"id": tp_id, "side": side, "size": str(size), "entry": str(entry)})
                    _full_reset("TP")
                    time.sleep(1); continue

            if sl_id:
                info = _fetch_order(sl_id)
                if _ord_status(info) in TERMINAL_STATES:
                    dash("trade", "SL TERMINAL", extra={"id": sl_id, "side": side, "size": str(size), "entry": str(entry)})
                    _full_reset("SL")
                    time.sleep(1); continue

            # If orders didn't show terminal but the account is flat → reset
            if not get_open_position():
                with POSITION_LOCK:
                    if POSITION["open"]:
                        dash("trade", "Position closed on exchange (no live pos) — resetting")
                        _full_reset("EXCHANGE_FLAT")

            time.sleep(1)
        except Exception as e:
            dash("warn", f"tp_sl_watchdog error: {e}")
            time.sleep(1)

# ---------------- Background Threads ----------------
def bg_bias_loop():
    dash_startup()
    while STATE["running"]:
        try:
            compute_bias()
        except Exception as e:
            dash("warn", f"bias loop error: {e}")
        time.sleep(max(15, BIAS_LOOP_SEC))

def bg_monitor_loop():
    last_open = None
    while STATE["running"]:
        try:
            pos = get_open_position()
            is_open = bool(pos)
            if is_open != last_open:
                last_open = is_open
                if is_open:
                    dash("state", "Exchange shows a live position",
                         extra={"size": pos.get("size"), "side": pos.get("side")})
                else:
                    dash("state", "Exchange shows FLAT")
            time.sleep(5)
        except Exception as e:
            dash("warn", f"monitor loop error: {e}")
            time.sleep(2)

def position_guard_loop():
    """Authoritative reconciliation loop so local state never drifts from exchange."""
    while STATE["running"]:
        try:
            reconcile_with_exchange()
        except Exception as e:
            dash("warn", f"position_guard_loop error: {e}")
        time.sleep(3)

def bg_heartbeat_loop():
    global HEARTBEAT_SEQ, LAST_HEARTBEAT_TS
    while STATE["running"]:
        try:
            with POSITION_LOCK:
                pos_snapshot = {
                    "open": POSITION["open"],
                    "side": POSITION["side"],
                    "size": str(POSITION["size"]) if POSITION["size"] else None,
                    "entry": str(POSITION["entry"]) if POSITION["entry"] else None,
                    "tp": str(POSITION["tp"]) if POSITION["tp"] else None,
                    "sl": str(POSITION["sl"]) if POSITION["sl"] else None,
                }
            HEARTBEAT_SEQ += 1
            LAST_HEARTBEAT_TS = int(time.time())
            dash("state", "heartbeat", extra={
                "hb_seq": HEARTBEAT_SEQ,
                "bias": BIAS,
                "mf_last": STATE["last_mf"],
                "trend_last": STATE["last_trend"],
                "position": pos_snapshot
            })
            time.sleep(max(10, DASH_HEARTBEAT_SEC))
        except Exception as e:
            dash("warn", f"heartbeat error: {e}")
            time.sleep(10)

_THREADS_STARTED = False
def start_background_threads_once():
    global _THREADS_STARTED
    if _THREADS_STARTED:
        return
    threading.Thread(target=bg_bias_loop,        name="bias",        daemon=True).start()
    threading.Thread(target=bg_monitor_loop,     name="monitor",     daemon=True).start()
    threading.Thread(target=tp_sl_watchdog,      name="tp_sl",       daemon=True).start()
    threading.Thread(target=position_guard_loop, name="pos_guard",   daemon=True).start()
    threading.Thread(target=bg_heartbeat_loop,   name="heartbeat",   daemon=True).start()
    _THREADS_STARTED = True

# ---------------- Cold-start sync ----------------
def cold_start_sync_position():
    """On boot, read exchange position and sync local POSITION for accurate logging/state."""
    try:
        pos = get_open_position()
        if not pos:
            dash("state", "Cold-start: exchange shows FLAT")
            return
        size = _D_safe(pos.get("size")) or Decimal("0")
        if size == 0:
            dash("state", "Cold-start: exchange position size=0 → FLAT")
            return
        side_norm = _normalize_apex_side(pos.get("side"))
        entry = _extract_entry_price_from_position(pos)
        with POSITION_LOCK:
            POSITION.update({
                "open": True,
                "side": side_norm,
                "size": size,
                "entry": entry,
                "order_id": None, "tp_id": None, "sl_id": None,
                "tp": None, "sl": None,
                "margin": None, "reserved_margin": None,
            })
        dash("state", "Cold-start: detected live position",
             extra={"side": side_norm, "size": str(size), "entry": str(entry) if entry else None})
    except Exception as e:
        dash("warn", f"cold_start_sync_position error: {e}")

# ---------------- Health & admin ----------------
@app.route("/__alive__", methods=["GET"])
def alive():
    return jsonify({
        "ok": True,
        "app": APP_NAME,
        "started_at_utc": STARTED_AT_UTC,
        "now_utc10": now(),
        "bias": BIAS,
        "use_bias": USE_BIAS,
        "heartbeat_seq": HEARTBEAT_SEQ,
        "last_heartbeat_ts": LAST_HEARTBEAT_TS,
        "mf_last": STATE["last_mf"],
        "trend_last": STATE["last_trend"],
        "mf_flip_since_entry": STATE["mf_flip_since_entry"],
        "position": {
            "open": POSITION["open"], "side": POSITION["side"],
            "entry": str(POSITION["entry"]) if POSITION["entry"] else None,
            "size": str(POSITION["size"]) if POSITION["size"] else None,
            "tp": str(POSITION["tp"]) if POSITION["tp"] else None,
            "sl": str(POSITION["sl"]) if POSITION["sl"] else None,
            "order_id": POSITION["order_id"], "tp_id": POSITION["tp_id"], "sl_id": POSITION["sl_id"]
        }
    }), 200

@app.route("/__threads__", methods=["GET"])
def threads_view():
    threads: List[threading.Thread] = list(threading.enumerate())
    rows = []
    for t in threads:
        rows.append({
            "name": t.name,
            "ident": t.ident,
            "daemon": t.daemon,
            "alive": t.is_alive(),
        })
    return jsonify({"ok": True, "count": len(rows), "threads": rows}), 200

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"pong": True, "ts": int(time.time()), "now_utc10": now()}), 200

@app.route("/admin/reset_state", methods=["POST","GET"])
def admin_reset_state():
    _full_reset("ADMIN")
    return jsonify({"ok": True, "msg": "state reset"}), 200

@app.route("/admin/reconcile", methods=["POST","GET"])
def admin_reconcile():
    reconcile_with_exchange()
    exch = bool(get_open_position())
    with POSITION_LOCK:
        local = {
            "open": POSITION["open"],
            "side": POSITION["side"],
            "size": str(POSITION["size"]) if POSITION["size"] else None,
            "entry": str(POSITION["entry"]) if POSITION["entry"] else None,
        }
    return jsonify({"ok": True, "exchange_open": exch, "local": local}), 200

# ---------------- Boot ----------------
def boot_on_import():
    # Called at module import (so gunicorn workers start processing immediately)
    dash_startup()
    dash("state", "Pinging Capital Manager…", extra={"url": CAPMGR_URL})
    ping_capital_manager()
    dash_capital_status_once()
    cold_start_sync_position()
    start_background_threads_once()

# Boot immediately unless explicitly disabled
if os.environ.get("DISABLE_BOOT_ON_IMPORT", "0") != "1":
    boot_on_import()

# ---------------- Local runner (not used on Render) ----------------
if __name__ == "__main__":
    app.config.update(ENV="production", DEBUG=False)
    port = int(os.environ.get("PORT", "5008"))
    dash("ok", f"Serving on 0.0.0.0:{port}")
    cold_start_sync_position()
    start_background_threads_once()
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )


