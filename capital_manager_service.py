# capital_manager_service.py — v1.4a (no envs; default weights at top)
from flask import Flask, request, jsonify
from decimal import Decimal
from capital_manager import CapitalManager, D, DB_PATH_DEFAULT

app = Flask(__name__)

# ====== BOT WEIGHTS (top-of-file config) ======
# Change these to add more bots/coins later.
DEFAULT_BOT_WEIGHTS = {
    "BTC": Decimal("0.50"),
    "SOL": Decimal("0.50"),
}
# ==============================================

# Engine (no envs)
manager = CapitalManager(db_path=DB_PATH_DEFAULT)

# Bootstrap: apply default weights & refresh wallet once at startup
try:
    manager.configure_bots({k: str(v) for k, v in DEFAULT_BOT_WEIGHTS.items()})
    live = manager.read_contract_wallet_usdt()
    manager.set_wallet_snapshot(live)
    manager.recalc_allocations()
except Exception:
    pass

@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok": True, "service": "capital_manager", "version": "v1.4a"}), 200

@app.route("/version", methods=["GET"])
def version():
    return jsonify({"ok": True, "version": "v1.4a"}), 200

@app.route("/health", methods=["GET"])
def health():
    h = manager.health()
    h["defaults"] = {k: str(v) for k, v in DEFAULT_BOT_WEIGHTS.items()}
    return jsonify(h), 200

@app.route("/configure", methods=["POST"])
def configure():
    """
    Body: {"bots":[{"id":"BTC","weight":"0.5"},{"id":"SOL","weight":"0.5"}]}
    """
    data = request.get_json(silent=True) or {}
    bots = data.get("bots") or []
    if not isinstance(bots, list) or not bots:
        return jsonify({"ok": False, "error": "bots list required"}), 400

    weights = {}
    try:
        for item in bots:
            bid = str(item["id"]).upper()
            w   = str(item["weight"])
            _ = Decimal(w)   # validate
            weights[bid] = w
    except Exception:
        return jsonify({"ok": False, "error": "bad weights payload"}), 400

    ok = manager.configure_bots(weights)
    if not ok:
        return jsonify({"ok": False, "error": "configure failed"}), 500
    return jsonify(manager.health()), 200

@app.route("/refresh_wallet", methods=["POST", "GET"])
def refresh_wallet():
    live = manager.read_contract_wallet_usdt()
    manager.set_wallet_snapshot(live)
    manager.recalc_allocations()
    return jsonify({"ok": True, "wallet_usdt": str(live), "health": manager.health()}), 200

@app.route("/reserve", methods=["POST"])
def reserve():
    """
    Body: {"bot_id":"SOL","trade_pct":"0.10"}
    Returns approved margin within allocation.
    """
    data = request.get_json(silent=True) or {}
    bot_id = data.get("bot_id")
    trade_pct = data.get("trade_pct")
    if not bot_id or trade_pct is None:
        return jsonify({"ok": False, "error": "bot_id and trade_pct required"}), 400
    res = manager.reserve(bot_id, str(trade_pct))
    code = 200 if res.get("ok") else 400
    return jsonify(res), code

@app.route("/release", methods=["POST"])
def release():
    """
    Body: {"bot_id":"SOL","amount_usdt":"50.0"}
    """
    data = request.get_json(silent=True) or {}
    bot_id = data.get("bot_id")
    amount = data.get("amount_usdt")
    if not bot_id or amount is None:
        return jsonify({"ok": False, "error": "bot_id and amount_usdt required"}), 400
    res = manager.release(bot_id, str(amount))
    code = 200 if res.get("ok") else 400
    return jsonify(res), code

@app.route("/on_close/<bot_id>", methods=["POST", "GET"])
def on_close(bot_id):
    res = manager.on_close(bot_id)
    return jsonify(res), 200

@app.route("/bot/<bot_id>", methods=["GET"])
def get_bot(bot_id):
    h = manager.health()
    found = None
    for b in h.get("bots", []):
        if str(b["bot_id"]).upper() == str(bot_id).upper():
            found = b
            break
    if not found:
        return jsonify({"ok": False, "error": "unknown bot"}), 404
    return jsonify({"ok": True, "bot": found, "wallet_usdt": h.get("wallet_usdt")}), 200

# Gunicorn entrypoint is "app" (we run via Dockerfile.capmgr)
