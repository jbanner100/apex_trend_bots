# ---------------- Webhooks (async, tolerant, always 200) ----------------
@app.route("/webhook_mf", methods=["POST","GET"], strict_slashes=False)
def webhook_mf():
    if request.method == "GET":
        return jsonify({"ok": True, "hint": 'POST {"message":"MF UP|MF DOWN","ts":<epoch optional>}'}), 200
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
        return jsonify({"ok": True, "hint": 'POST {"message":"TREND UP|TREND DOWN","ts":<epoch optional>}'}), 200
    try:
        msg, ts = _extract_message_and_ts()
        s = (msg or "").upper()
        if   "UP"   in s: d = "UP"
        elif "DOWN" in s: d = "DOWN"
        else:
            return jsonify({"ok": False, "reason": "message must contain TREND UP or TREND DOWN"}), 200

        STATE["last_trend"] = {"dir": d, "ts": ts}
        dash("signal", f"TREND {d}", extra={"ts": ts, "raw": msg})

        desired = "LONG" if d=="UP" else "SHORT"
        if not POSITION["open"]:
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
        return jsonify({"ok": True, "hint": 'POST {"message":"CONFIRM","ts":<epoch optional>}'}), 200
    try:
        _msg, ts = _extract_message_and_ts()
        STATE["last_confirm"] = {"ts": ts}
        dash("signal", "3rd CONFIRM", extra={"ts": ts})
        return jsonify({"ok": True, "confirm": True, "ts": ts}), 200
    except Exception as e:
        dash("warn", f"webhook_confirm handler error: {e}")
        return jsonify({"ok": False, "error": "handler_exception"}), 200

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
                    dash("trade", "TP FILLED", extra={"id": tp_id, "side": side, "size": str(size), "entry": str(entry)})
                    _full_reset("TP")
                    time.sleep(1); continue

            if sl_id:
                info = _fetch_order(sl_id)
                if _ord_status(info) in TERMINAL_STATES:
                    dash("trade", "SL FILLED", extra={"id": sl_id, "side": side, "size": str(size), "entry": str(entry)})
                    _full_reset("SL")
                    time.sleep(1); continue

            time.sleep(1)
        except Exception as e:
            dash("warn", f"tp_sl_watchdog error: {e}")
            time.sleep(1)

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

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"pong": True, "ts": int(time.time()), "now_utc10": now()}), 200

@app.route("/admin/reset_state", methods=["POST","GET"])
def admin_reset_state():
    _full_reset("ADMIN")
    return jsonify({"ok": True, "msg": "state reset"}), 200

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

def bg_heartbeat_loop():
    """Periodic status line so logs never go totally quiet on Render."""
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
            dash("state", "heartbeat", extra={
                "bias": BIAS,
                "mf_last": STATE["last_mf"],
                "trend_last": STATE["last_trend"],
                "position": pos_snapshot
            })
            time.sleep(max(10, int(os.environ.get("DASH_HEARTBEAT_SEC", "60"))))
        except Exception as e:
            dash("warn", f"heartbeat error: {e}")
            time.sleep(10)

_THREADS_STARTED = False
def start_background_threads_once():
    global _THREADS_STARTED
    if _THREADS_STARTED:
        return
    threading.Thread(target=bg_bias_loop,     name="bias",      daemon=True).start()
    threading.Thread(target=bg_monitor_loop,  name="monitor",   daemon=True).start()
    threading.Thread(target=tp_sl_watchdog,   name="tp_sl",     daemon=True).start()
    threading.Thread(target=bg_heartbeat_loop,name="heartbeat", daemon=True).start()
    _THREADS_STARTED = True

def boot_on_import():
    # Called at module import (so gunicorn workers start processing immediately)
    dash_startup()
    # Capital Manager connectivity check (non-fatal)
    dash("state", "Pinging Capital Manager…", extra={"url": CAPMGR_URL})
    ping_capital_manager()
    dash_capital_status_once()
    start_background_threads_once()

# Boot immediately unless explicitly disabled
if os.environ.get("DISABLE_BOOT_ON_IMPORT", "0") != "1":
    boot_on_import()

# ---------------- Local runner (not used on Render) ----------------
if __name__ == "__main__":
    app.config.update(ENV="production", DEBUG=False)
    port = int(os.environ.get("PORT", "5009"))
    dash("ok", f"Serving on 0.0.0.0:{port}")
    start_background_threads_once()
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )
