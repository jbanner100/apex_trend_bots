# app_btc_render.py — Apex Omni BTC Bot (Render-ready, v1.4R)

import os
import json
import threading
import time
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
import ccxt
from flask import Flask, request, jsonify

# stdlib HTTP (no external "requests" dep needed)
import urllib.request

# === ApeX API imports ===
from apexomni.constants import APEX_OMNI_HTTP_MAIN, NETWORKID_OMNI_MAIN_ARB
from apexomni.http_private_sign import HttpPrivateSign
from apexomni.http_public import HttpPublic

# ---------------- Precision ----------------
getcontext().prec = 28

# ---------------- App ----------------
app = Flask(__name__)
APP_NAME = "Apex Omni BTC Bot (v1.4R)"
STARTED_AT_UTC = datetime.utcnow().isoformat() + "Z"

# ---------------- Timezone (UTC+10) ----------------
UTC_PLUS_10 = timedelta(hours=10)
def now_utc10_dt() -> datetime:
    return datetime.utcnow() + UTC_PLUS_10
def now() -> str:
    return now_utc10_dt().strftime("[%Y-%m-%d %H:%M:%S UTC+10]")

# ---------------- ANSI Colors ----------------
CLR = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "green": "\033[92m",
    "red": "\033[91m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "mag": "\033[95m",
    "gray": "\033[90m",
}
def colorize(msg: str, color: str) -> str:
    return f"{CLR.get(color,'')}{msg}{CLR['reset']}"

# ---------------- Config (BTC) ----------------
APEX_SYMBOL        = os.environ.get("APEX_SYMBOL", "BTC-USDT")
BINANCE_SYMBOL     = os.environ.get("BINANCE_SYMBOL", "BTC/USDT")
BOT_ID             = os.environ.get("BOT_ID", "BTC")

# Bias TF (1h)
BIAS_INTERVAL      = os.environ.get("BIAS_INTERVAL", "1h")
EMA_PERIOD         = int(os.environ.get("EMA_PERIOD", "50"))
ICT_EMA_SLOPE_BARS = int(os.environ.get("ICT_EMA_SLOPE_BARS", "10"))
ICT_SWING_LOOKBACK = int(os.environ.get("ICT_SWING_LOOKBACK", "5"))
ICT_BOS_BUFFER_PCT = Decimal(os.environ.get("ICT_BOS_BUFFER_PCT", "0.10"))
ICT_REQUIRE_BOS    = (os.environ.get("ICT_REQUIRE_BOS", "false").lower() == "true")
DEBUG_BIAS         = (os.environ.get("DEBUG_BIAS", "true").lower() == "true")

# Trade gates
USE_BIAS            = (os.environ.get("USE_BIAS", "false").lower() == "true")
ALLOW_COUNTER_TREND = (os.environ.get("ALLOW_COUNTER_TREND", "true").lower() == "true")

# Exchange tick/steps (ApeX BTC)
TICK_SIZE       = Decimal(os.environ.get("TICK_SIZE", "1"))
SIZE_STEP       = Decimal(os.environ.get("SIZE_STEP", "0.001"))
MIN_ORDER_USDT  = Decimal(os.environ.get("MIN_ORDER_USDT", "5"))

# Capital Manager service (must be a separate Render service or reachable URL)
CAPMGR_URL      = os.environ.get("CAPMGR_URL", "http://127.0.0.1:5015")  # set to https://<your-capmgr>.onrender.com in Render

# Leverage & sizing
LEVERAGE        = Decimal(os.environ.get("LEVERAGE", "15"))
TRADE_SIZE_PCT  = Decimal(os.environ.get("TRADE_SIZE_PCT", "0.10"))   # % of allocation to request per entry

# Bias-aware TP/SL (fractions; 0.01 = 1%)
TREND_TP_PCT    = Decimal(os.environ.get("TREND_TP_PCT", "0.01"))
TREND_SL_PCT    = Decimal(os.environ.get("TREND_SL_PCT", "0.0075"))
CT_TP_PCT       = Decimal(os.environ.get("CT_TP_PCT", "0.0050"))
CT_SL_PCT       = Decimal(os.environ.get("CT_SL_PCT", "0.0050"))

# MF window (seconds)
MF_WAIT_SEC     = int(os.environ.get("MF_WAIT_SEC", "8000"))
MF_LEAD_SEC     = int(os.environ.get("MF_LEAD_SEC", "8000"))

# Optional 3rd confirmation webhook (provision)
ENABLE_THIRD_CONFIRMATION = (os.environ.get("ENABLE_THIRD_CONFIRMATION", "false").lower() == "true")

# ---------------- Credentials ----------------
api_creds = {
    "key":        os.environ.get("APEX_KEY",        "3e965beb-41e2-f125-a7c5-569f45bfba21"),
    "secret":     os.environ.get("APEX_SECRET",     "NXtuAyq4hS9G4fVlytQtQGn9Qk5LUukXGAYg8SBj"),
    "passphrase": os.environ.get("APEX_PASSPHRASE", "GEy6yNGaZ5_0fuX4VBJ3"),
}
zk_seeds = os.environ.get("APEX_ZK_SEEDS", "0xd00ec9396facbafc423b5d92a289ea49adfdb0b918d3d5db26edbb978893ed5d0bd48c3fbe4309de2a09a3514cbec6d2c4012df85653a1421aca1cf599acda491c")
zk_l2Key = os.environ.get("APEX_ZK_L2KEY", "0xd6094a658c50dccf9be8f85cde1804e92a74a0482788766c9b8744cbc6fe8501")

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

# ---------------- Accounts & Price ----------------
def get_usdt_contract_balance() -> Decimal:
    try:
        acct = client.get_account_v3()
        for w in acct.get("contractWallets", []):
            if w.get("token") == "USDT":
                return Decimal(str(w.get("balance", "0")))
    except Exception as e:
        dash("warn", f"get_usdt_contract_balance error: {e}")
    return Decimal("0")

def get_open_position() -> Optional[dict]:
    try:
        acct = client.get_account_v3()
        for p in acct.get("positions", []):
            if p.get("symbol") == APEX_SYMBOL and Decimal(str(p.get("size", "0"))) != Decimal("0"):
                return p
    except Exception as e:
        dash("warn", f"get_open_position error: {e}")
    return None

def get_public_price() -> Decimal:
    """
    Prefer markPrice → indexPrice → lastPrice. Always return > 0 or raise.
    """
    t = http_public.ticker_v3(symbol=APEX_SYMBOL)
    row = (t.get("data") or [{}])[0]
    for key in ("markPrice","indexPrice","lastPrice"):
        v = _D_safe(row.get(key))
        if v is not None and v > 0:
            return v
    raise ValueError("No valid mark/index/last price")

# ---------------- Dashboard ----------------
DASH_LAST = {"bias": None, "connected": False}
def dash(event: str, msg: str, *, extra: Optional[dict] = None):
    icons = {"start":"🚀","ok":"✅","warn":"⚠️","error":"❌","signal":"🔔","state":"🖥️","trade":"📈","debug":"🔎"}
    colors = {"start":"cyan","ok":"green","warn":"yellow","error":"red","signal":"mag","state":"blue","trade":"green","debug":"gray"}
    icon = icons.get(event, "•"); col = colors.get(event, "reset")
    payload = f"{now()} {icon} {msg}"
    if extra is not None: payload += f" {CLR['dim']}{extra}{CLR['reset']}"
    print(colorize(payload, col))

def dash_startup():
    if not DASH_LAST["connected"]:
        DASH_LAST["connected"] = True
        dash("start", f"{APP_NAME} starting")
        dash("ok", "ApeX client initialized", extra={"endpoint": APEX_OMNI_HTTP_MAIN})
        dash("state", f"USE_BIAS={USE_BIAS}, ALLOW_COUNTER_TREND={ALLOW_COUNTER_TREND}")
        dash("state", "Waiting for signals: MF + TREND", extra={"lead_sec": MF_LEAD_SEC, "wait_sec": MF_WAIT_SEC})
        dash("state", "Render mode: expecting external Capital Manager", extra={"CAPMGR_URL": CAPMGR_URL})

def dash_bias(new_bias: Optional[str]):
    if DASH_LAST["bias"] != new_bias:
        DASH_LAST["bias"] = new_bias
        human = new_bias if new_bias else "NEUTRAL"
        color = "green" if new_bias == "LONG" else "red" if new_bias == "SHORT" else "yellow"
        print(colorize(f"{now()} 🧭 ICT Bias → {human}", color))

# ---------------- Bias via Binance (1h) ----------------
BIAS: Optional[str] = None

def fetch_binance_candles(symbol: str, interval: str, limit: int = 500) -> Optional[pd.DataFrame]:
    try:
        ex = ccxt.binance()
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        if not ohlcv:
            return None
        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
        return df
    except Exception as e:
        dash("warn", f"fetch_binance_candles error: {e}")
        return None

def compute_ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
    df = df.copy()
    df["ema"] = df["close"].ewm(span=period, adjust=False).mean()
    return df

def compute_bias():
    global BIAS
    try:
        limit = max(EMA_PERIOD + 200, 300)
        df = fetch_binance_candles(BINANCE_SYMBOL, BIAS_INTERVAL, limit=limit)
        if df is None or df.empty or len(df) < EMA_PERIOD + ICT_EMA_SLOPE_BARS + 10:
            if BIAS is not None:
                BIAS = None
                dash_bias(BIAS)
            if DEBUG_BIAS:
                dash("debug", "ICT Bias: insufficient data")
            return

        df = compute_ema(df, EMA_PERIOD)
        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values
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

        decided = None
        if last_bos_dir == "UP" and ema_up and price_above:
            decided = "LONG"
        elif last_bos_dir == "DOWN" and ema_down and price_below:
            decided = "SHORT"
        else:
            if not ICT_REQUIRE_BOS:
                if ema_up and price_above:   decided = "LONG"
                elif ema_down and price_below: decided = "SHORT"
                else: decided = None
            else:
                decided = None

        if decided != BIAS:
            BIAS = decided
            dash_bias(BIAS)

        if DEBUG_BIAS:
            dash("debug", f"Bias calc done", extra={"bias": BIAS})

    except Exception as e:
        if BIAS is not None:
            BIAS = None
            dash_bias(BIAS)
        dash("warn", f"ICT Bias error: {e}")

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

# ---------------- Capital Manager HTTP helpers ----------------
def _http_get_json(url: str, timeout: float = 3.0) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None

def _http_post_json(url: str, payload: dict, timeout: float = 5.0) -> Optional[dict]:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None

def capmgr_up() -> bool:
    h = _http_get_json(f"{CAPMGR_URL}/health")
    return bool(h and h.get("ok"))

def dash_capital_status_once():
    h = _http_get_json(f"{CAPMGR_URL}/health")
    if not (h and h.get("ok")):
        dash("warn", "Capital Manager not reachable", extra={"url": CAPMGR_URL})
        return
    bot_row = None
    for b in (h.get("bots") or []):
        if str(b.get("bot_id")).upper() == BOT_ID:
            bot_row = b; break
    extra = {
        "wallet_usdt": h.get("wallet_usdt"),
        "bot": BOT_ID,
        "symbol": APEX_SYMBOL,
        "initial_trade_pct": str(TRADE_SIZE_PCT),
        "url": CAPMGR_URL
    }
    if bot_row:
        extra.update({
            "weight": bot_row.get("weight"),
            "allocated_usdt": bot_row.get("allocated_usdt"),
            "reserved_usdt": bot_row.get("reserved_usdt"),
        })
    dash("ok", "Capital Manager status", extra=extra)

def capmgr_reserve(trade_pct: Decimal) -> Optional[dict]:
    return _http_post_json(f"{CAPMGR_URL}/reserve", {"bot_id": BOT_ID, "trade_pct": str(trade_pct)})

def capmgr_release(amount_usdt: Decimal) -> Optional[dict]:
    return _http_post_json(f"{CAPMGR_URL}/release", {"bot_id": BOT_ID, "amount_usdt": str(amount_usdt)})

def capmgr_on_close() -> Optional[dict]:
    return _http_post_json(f"{CAPMGR_URL}/on_close", {"bot_id": BOT_ID})

# ---------------- Runtime State ----------------
STATE = {
    "running": True,
    "last_mf": None,           # {'dir':'UP'|'DOWN','ts':int}
    "last_trend": None,        # {'dir':'UP'|'DOWN','ts':int}
    "last_confirm": None,      # {'ts':int}
    "mf_flip_since_entry": False,
}
POSITION = {
    "open": False, "side": None, "entry": None, "size": None,
    "order_id": None, "tp_id": None, "sl_id": None, "tp": None, "sl": None,
    "margin": None, "reserved_margin": None,
    "vector_close_timestamp": None,
    "vector_side": None,
}
POSITION_LOCK = threading.Lock()

# ---------------- Terminal state helpers ----------------
TERMINAL_STATES = {"FILLED", "TRIGGERED", "EXECUTED", "COMPLETED", "DONE"}

def _ord_status(info: dict) -> str:
    return str((info or {}).get("status", "")).upper()

def _fetch_order(order_id: str) -> dict:
    if not order_id: return {}
    try:
        return client.get_order_v3(symbol=APEX_SYMBOL, orderId=str(order_id)).get("data") or {}
    except Exception as e:
        dash("warn", f"get_order_v3 error", extra={"id": order_id, "err": str(e)})
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
        reserved = POSITION.get("reserved_margin")
    if tp_id: _cancel_id(tp_id, "TP")
    if sl_id: _cancel_id(sl_id, "SL")
    # Notify allocator (refresh & zero reservations for this bot)
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

# ---------------- MARKET Entry (+ reduce-only TP/SL) ----------------
def place_market_order(direction: str, trade_size_pct: Optional[Decimal] = None) -> Optional[str]:
    try:
        direction = str(direction).upper()
        if direction not in ("LONG","SHORT"):
            dash("error", f"Invalid direction: {direction}")
            return None

        # Live position gate (fresh read)
        live = get_open_position()
        if live:
            dash("warn", "Live position exists — aborting duplicate order.",
                 extra={"size": str(live.get("size")), "side": live.get("side")})
            return None

        # Capital reservation
        t_pct = TRADE_SIZE_PCT if trade_size_pct is None else (_D_safe(trade_size_pct) or TRADE_SIZE_PCT)
        if not capmgr_up():
            dash("warn", "Capital Manager unreachable — entry aborted", extra={"url": CAPMGR_URL})
            return None
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
        mark_price_r = floor_to_tick(mark_price)
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
        if not tp_id:
            dash("warn", "TP placement failed", extra={"resp": tp_order})

        # SL (STOP_MARKET)
        sl_order = client.create_order_v3(
            symbol=APEX_SYMBOL, side=sl_side, type="STOP_MARKET",
            triggerPrice=sl_trig_str, price=sl_exec_str, size=size_str,
            reduceOnly=True, timestampSeconds=int(time.time())
        )
        sl_id = (sl_order.get("data") or {}).get("id")
        if not sl_id:
            dash("warn", "SL placement failed", extra={"resp": sl_order})

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

# ---------------- Webhooks ----------------
def parse_up_down(msg: str) -> Optional[str]:
    s = str(msg or "").upper()
    if "UP" in s: return "UP"
    if "DOWN" in s: return "DOWN"
    return None

@app.route("/webhook_mf", methods=["POST","GET"], strict_slashes=False)
def webhook_mf():
    if request.method == "GET":
        return jsonify({"ok": True, "hint": 'POST {"message":"MF UP|MF DOWN","ts":<epoch optional>}'})
    data = request.get_json(silent=True) or {}
    d = parse_up_down(data.get("message"))
    if d not in ("UP","DOWN"):
        return jsonify({"ok": False, "reason": "message must contain MF UP or MF DOWN"}), 400
    ts = int(data.get("ts") or time.time())

    STATE["last_mf"] = {"dir": d, "ts": ts}
    dash("signal", f"MF {d}", extra={"ts": ts})

    if POSITION["open"]:
        if POSITION["side"] == "LONG" and d == "DOWN": STATE["mf_flip_since_entry"] = True
        if POSITION["side"] == "SHORT" and d == "UP":  STATE["mf_flip_since_entry"] = True

    desired = "LONG" if d=="UP" else "SHORT"
    if not POSITION["open"]:
        side = check_and_latch(desired)
        if side:
            oid = place_market_order(side)
            return jsonify({"ok": bool(oid), "action": "entry", "side": side, "ts": ts})
    return jsonify({"ok": True, "latched": False, "ts": ts})

@app.route("/webhook_trend", methods=["POST","GET"], strict_slashes=False)
def webhook_trend():
    if request.method == "GET":
        return jsonify({"ok": True, "hint": 'POST {"message":"TREND UP|TREND DOWN","ts":<epoch optional>}'})
    data = request.get_json(silent=True) or {}
    s = str(data.get("message") or "").upper()
    if "UP" in s: d = "UP"
    elif "DOWN" in s: d = "DOWN"
    else:
        return jsonify({"ok": False, "reason": "message must contain TREND UP or TREND DOWN"}), 400
    ts = int(data.get("ts") or time.time())

    STATE["last_trend"] = {"dir": d, "ts": ts}
    dash("signal", f"TREND {d}", extra={"ts": ts})

    desired = "LONG" if d=="UP" else "SHORT"
    if not POSITION["open"]:
        side = check_and_latch(desired)
        if side:
            oid = place_market_order(side)
            return jsonify({"ok": bool(oid), "action": "entry", "side": side, "ts": ts})
    return jsonify({"ok": True, "latched": False, "ts": ts})

@app.route("/webhook_confirm", methods=["POST","GET"], strict_slashes=False)
def webhook_confirm():
    if request.method == "GET":
        return jsonify({"ok": True, "hint": 'POST {"message":"CONFIRM","ts":<epoch optional>}'})
    data = request.get_json(silent=True) or {}
    ts = int(data.get("ts") or time.time())
    STATE["last_confirm"] = {"ts": ts}
    dash("signal", "3rd CONFIRM", extra={"ts": ts})
    return jsonify({"ok": True, "confirm": True, "ts": ts})

# ---------------- Force Test Entry ----------------
@app.route("/test/force_entry", methods=["POST","GET"], strict_slashes=False)
def test_force_entry():
    if request.method == "GET":
        return jsonify({"ok": True, "hint": 'POST {"side":"LONG|SHORT","trade_pct":0.10}'})
    data = request.get_json(silent=True) or {}
    side = str(data.get("side","")).upper()
    trade_pct = _D_safe(data.get("trade_pct"))
    if side not in ("LONG","SHORT"):
        return jsonify({"ok": False, "reason": "side must be LONG or SHORT"}), 400
    t_pct = trade_pct if trade_pct is not None else TRADE_SIZE_PCT
    dash("state", "FORCE ENTRY invoked", extra={"side": side, "trade_pct": str(t_pct)})
    oid = place_market_order(side, t_pct)
    return jsonify({"ok": bool(oid), "order_id": oid})

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

            # TP?
            if tp_id:
                info = _fetch_order(tp_id)
                if _ord_status(info) in TERMINAL_STATES:
                    dash("trade", "TP FILLED", extra={"id": tp_id, "side": side, "size": str(size), "entry": str(entry)})
                    _full_reset("TP")
                    time.sleep(1); continue

            # SL?
            if sl_id:
                info = _fetch_order(sl_id)
                if _ord_status(info) in TERMININAL_STATES:
                    dash("trade", "SL FILLED", extra={"id": sl_id, "side": side, "size": str(size), "entry": str(entry)})
                    _full_reset("SL")
                    time.sleep(1); continue

            time.sleep(1)
        except Exception as e:
            dash("warn", f"tp_sl_watchdog error: {e}")
            time.sleep(1)

# (typo guard)
TERMININAL_STATES = TERMINAL_STATES  # alias to avoid accidental misspells above

# ---------------- Health ----------------
@app.route("/__alive__", methods=["GET"])
def alive():
    return jsonify({
        "ok": True,
        "app": APP_NAME,
        "started_at_utc": STARTED_AT_UTC,
        "now_utc10": now(),
        "bias": BIAS,
        "use_bias": USE_BIAS,
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

@app.route("/")
def root():
    # simple root to trigger @before_first_request under gunicorn and for Render health checks
    return jsonify({"ok": True, "service": APP_NAME, "now_utc10": now()})

# ---------------- Background Threads ----------------
def bg_bias_loop():
    dash_startup()
    while STATE["running"]:
        try:
            compute_bias()
        except Exception as e:
            dash("warn", f"bias loop error: {e}")
        time.sleep(60)

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

_bg_threads_started = False
def start_background_threads_once():
    global _bg_threads_started
    if _bg_threads_started:
        return
    threading.Thread(target=bg_bias_loop,   name="bias",    daemon=True).start()
    threading.Thread(target=bg_monitor_loop,name="monitor", daemon=True).start()
    threading.Thread(target=tp_sl_watchdog, name="tp_sl",   daemon=True).start()
    _bg_threads_started = True
    dash("ok", "Background threads started")

@app.before_first_request
def _boot_under_gunicorn():
    # Render/gunicorn path: no __main__ block runs, so boot here on first HTTP request
    dash_startup()
    dash_capital_status_once()
    start_background_threads_once()

# ---------------- Runner (local dev) ----------------
if __name__ == "__main__":
    app.config.update(ENV="production", DEBUG=False)
    dash_startup()
    dash_capital_status_once()

    # Creds check
    missing = [k for k, v in api_creds.items() if not v or v == "REPLACE_ME"]
    if missing or zk_seeds == "REPLACE_ME" or zk_l2Key == "REPLACE_ME":
        dash("warn", "Missing ApeX credentials — fill api_creds/zk_seeds/zk_l2Key before trading.")

    start_background_threads_once()

    port = int(os.environ.get("PORT", "5008"))
    dash("ok", f"Serving on 127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
