# capital_manager.py — v1.4c (engine; no envs)
# Purpose:
#   • Maintain a wallet snapshot (USDT contract) in SQLite
#   • Manage per-bot weights and normalized allocations
#   • Atomic reserve/release per bot with WAL + BEGIN IMMEDIATE
#
# Notes:
#   • Credentials are hardcoded by design here (“no envs”). For safety, rotate
#     any keys you pasted previously and move to env vars later if possible.
#   • File is self-contained; importing apexomni is “soft” so the module
#     can load even if the SDK isn’t installed yet (client=None).
#   • Python 3.12 recommended on Render. requirements.txt should pin:
#       apexomni==3.0.8, ecdsa==0.19.0, pandas==2.2.3, ccxt==4.3.79, web3<7

import time
import sqlite3
import threading
from decimal import Decimal, getcontext
from typing import Dict, Any, Optional

# ================== CREDENTIALS — FILL THESE IN ==================
# If you insist on “no envs”, hardcode here (recommended: rotate keys first).
key = api_key or os.environ.get("APEX_API_KEY") or "3e965beb-41e2-f125-a7c5-569f45bfba21"
        secret = api_secret or os.environ.get("APEX_API_SECRET") or "NXtuAyq4hS9G4fVlytQtQGn9Qk5LUukXGAYg8SBj"
        passphrase = api_passphrase or os.environ.get("APEX_API_PASSPHRASE") or "GEy6yNGaZ5_0fuX4VBJ3"
        seeds = zk_seeds or os.environ.get("ZK_SEEDS") or "0xd00ec9396facbafc423b5d92a289ea49adfdb0b918d3d5db26edbb978893ed5d0bd48c3fbe4309de2a09a3514cbec6d2c4012df85653a1421aca1cf599acda491c"
        l2key = zk_l2key or os.environ.get("ZK_L2KEY") or "0xd6094a658c50dccf9be8f85cde1804e92a74a0482788766c9b8744cbc6fe8501"

# Uppercase aliases so either naming style works elsewhere.
API_CREDS = api_creds
ZK_SEEDS  = zk_seeds
ZK_L2KEY  = zk_l2Key
# ================================================================

# ====== DB LOCATION (no envs) ======
DB_PATH_DEFAULT = "./capalloc.db"
# ===================================

# ---- ApeX SDK (soft import so this module can still load if not installed) ----
_APEX_OK = True
try:
    from apexomni.constants import APEX_OMNI_HTTP_MAIN, NETWORKID_OMNI_MAIN_ARB
    from apexomni.http_private_sign import HttpPrivateSign
except Exception:
    _APEX_OK = False
    APEX_OMNI_HTTP_MAIN = None           # type: ignore
    NETWORKID_OMNI_MAIN_ARB = None       # type: ignore
    class HttpPrivateSign:               # type: ignore
        pass

# ---------------- Precision ----------------
getcontext().prec = 28

def D(x) -> Decimal:
    """Safe Decimal conversion that tolerates None and non-Decimal inputs."""
    if isinstance(x, Decimal):
        return x
    if x is None:
        return Decimal("0")
    return Decimal(str(x))


# ---------------- SQLite helpers ----------------
def _connect(db_path: str):
    conn = sqlite3.connect(
        db_path,
        timeout=5.0,
        isolation_level=None,   # manual transactions
        check_same_thread=False,
    )
    # WAL for multi-process friendliness
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


class CapitalManager:
    """
    Manages:
      • wallet snapshot (contract USDT) — persisted in SQLite
      • per-bot weights & normalized allocations
      • atomic reservation/release per bot

    Concurrency: SQLite WAL + BEGIN IMMEDIATE with a process-wide lock.
    """

    def __init__(self, db_path: str = DB_PATH_DEFAULT):
        self.db_path = db_path
        self._lock = threading.Lock()
        self.client = self._build_apex_client()
        self._init_db()

    # ---------- ApeX ----------
    def _build_apex_client(self):
        """Construct ApeX client, or return None if apexomni isn’t available/ready."""
        if not _APEX_OK:
            print("[capital_manager] ⚠️ apexomni not available — running with client=None")
            return None
        try:
            client = HttpPrivateSign(
                APEX_OMNI_HTTP_MAIN,
                network_id=NETWORKID_OMNI_MAIN_ARB,
                api_key_credentials=api_creds,  # use lowercase names actually defined
                zk_seeds=zk_seeds,
                zk_l2Key=zk_l2Key,
            )
            client.configs_v3()
            return client
        except Exception as e:
            print(f"[capital_manager] ⚠️ failed to construct ApeX client: {e}")
            return None

    # ---------- DB ----------
    def _db(self):
        return _connect(self.db_path)

    def _init_db(self):
        with self._lock:
            conn = self._db()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("""
                CREATE TABLE IF NOT EXISTS bots(
                  bot_id TEXT PRIMARY KEY,
                  weight TEXT NOT NULL,
                  allocated_usdt TEXT NOT NULL,
                  reserved_usdt TEXT NOT NULL,
                  updated_ts INTEGER NOT NULL
                )""")
                conn.execute("""
                CREATE TABLE IF NOT EXISTS meta(
                  k TEXT PRIMARY KEY,
                  v TEXT NOT NULL
                )""")
                # Seed wallet snapshot (best-effort; don’t fail if network down)
                bal = str(self.read_contract_wallet_usdt())
                conn.execute(
                    "INSERT INTO meta(k,v) VALUES('wallet_usdt', ?) "
                    "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                    (bal,)
                )
                conn.commit()
            finally:
                conn.close()

    # ---------- Wallet ----------
    def read_contract_wallet_usdt(self) -> Decimal:
        """
        Live call to ApeX to fetch USDT contract wallet balance.
        Returns Decimal('0') if client missing or request fails.
        """
        if not self.client:
            return D("0")
        try:
            acct = self.client.get_account_v3()
            for w in (acct.get("contractWallets") or []):
                if (w.get("token") or "").upper() == "USDT":
                    return D(w.get("balance", "0"))
        except Exception as e:
            # Keep it quiet; this is best-effort.
            print(f"[capital_manager] ⚠️ read_contract_wallet_usdt failed: {e}")
        return D("0")

    def get_wallet_snapshot(self) -> Decimal:
        conn = self._db()
        try:
            row = conn.execute("SELECT v FROM meta WHERE k='wallet_usdt'").fetchone()
            return D(row[0]) if row else D("0")
        finally:
            conn.close()

    def set_wallet_snapshot(self, amount: Decimal):
        conn = self._db()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO meta(k,v) VALUES('wallet_usdt', ?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (str(D(amount)),)
            )
            conn.commit()
        finally:
            conn.close()

    # ---------- Configure & allocation ----------
    def configure_bots(self, weights_dict: Dict[str, Any]) -> bool:
        """
        Replace bot set with given weights. Example:
            {"BTC": "0.5", "SOL": "0.5"}
        """
        if not weights_dict:
            return False

        with self._lock:
            conn = self._db()
            try:
                conn.execute("BEGIN IMMEDIATE")
                now_ts = int(time.time())
                ids = tuple(weights_dict.keys())

                for bot_id, w in weights_dict.items():
                    w = D(w)
                    if w < D("0"):
                        w = D("0")
                    conn.execute(
                        """INSERT INTO bots(bot_id, weight, allocated_usdt, reserved_usdt, updated_ts)
                           VALUES(?,?,?,?,?)
                           ON CONFLICT(bot_id) DO UPDATE SET weight=excluded.weight""",
                        (str(bot_id).upper(), str(w), "0", "0", now_ts)
                    )

                # Remove any not in the new set
                if ids:
                    q = "DELETE FROM bots WHERE bot_id NOT IN (%s)" % ",".join("?" * len(ids))
                    conn.execute(q, tuple(str(x).upper() for x in ids))

                conn.commit()
            except Exception:
                try: conn.rollback()
                except Exception: pass
                return False
            finally:
                conn.close()

        return self.recalc_allocations()

    def recalc_allocations(self) -> bool:
        """Normalize weights against current snapshot and update allocations."""
        with self._lock:
            conn = self._db()
            try:
                conn.execute("BEGIN IMMEDIATE")
                wallet = self.get_wallet_snapshot()
                rows = conn.execute("SELECT bot_id, weight, reserved_usdt FROM bots").fetchall()
                bots = [{"id": r[0], "w": D(r[1]), "res": D(r[2])} for r in rows]
                total_w = sum(b["w"] for b in bots)
                if total_w <= 0:
                    conn.rollback()
                    return False

                now_ts = int(time.time())
                for b in bots:
                    share = (b["w"] / total_w)
                    alloc = (wallet * share).quantize(Decimal("0.0001"))
                    # Keep reservation within allocation
                    res = b["res"] if b["res"] <= alloc else alloc
                    conn.execute(
                        "UPDATE bots SET allocated_usdt=?, reserved_usdt=?, updated_ts=? WHERE bot_id=?",
                        (str(alloc), str(res), now_ts, b["id"])
                    )
                conn.commit()
                return True
            except Exception:
                try: conn.rollback()
                except Exception: pass
                return False
            finally:
                conn.close()

    # ---------- Reservations ----------
    def reserve(self, bot_id: str, trade_pct: str) -> Dict[str, Any]:
        """
        Reserve margin from bot's allocation.
        Returns {ok, approved_margin_usdt, allocated_usdt, reserved_usdt, available_usdt}
        trade_pct is a fraction (e.g., "0.05" for 5% of the bot’s allocation).
        """
        bot = str(bot_id).upper()
        t_pct = D(trade_pct)
        if t_pct < D("0"):
            t_pct = D("0")

        with self._lock:
            conn = self._db()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT allocated_usdt, reserved_usdt FROM bots WHERE bot_id=?",
                    (bot,)
                ).fetchone()
                if not row:
                    conn.rollback()
                    return {"ok": False, "error": "unknown bot"}

                allocated = D(row[0])
                reserved  = D(row[1])
                available = allocated - reserved
                if available < D("0"):
                    available = D("0")

                requested = (allocated * t_pct).quantize(Decimal("0.0001"))
                approved  = requested if requested <= available else available

                if approved > D("0"):
                    now_ts = int(time.time())
                    new_reserved = reserved + approved
                    conn.execute(
                        "UPDATE bots SET reserved_usdt=?, updated_ts=? WHERE bot_id=?",
                        (str(new_reserved), now_ts, bot)
                    )
                    conn.commit()
                else:
                    conn.rollback()
                    new_reserved = reserved

                return {
                    "ok": True,
                    "approved_margin_usdt": str(approved),
                    "allocated_usdt": str(allocated),
                    "reserved_usdt": str(new_reserved),
                    "available_usdt": str(available),
                }
            except Exception as e:
                try: conn.rollback()
                except Exception: pass
                return {"ok": False, "error": str(e)}
            finally:
                conn.close()

    def release(self, bot_id: str, amount_usdt: str) -> Dict[str, Any]:
        """Release a portion of the bot’s reserved USDT back to available."""
        bot = str(bot_id).upper()
        amt = D(amount_usdt)
        if amt <= D("0"):
            return {"ok": False, "error": "bad amount"}

        with self._lock:
            conn = self._db()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT reserved_usdt FROM bots WHERE bot_id=?",
                    (bot,)
                ).fetchone()
                cur = D(row[0]) if row else D("0")
                new_res = cur - amt
                if new_res < D("0"):
                    new_res = D("0")
                now_ts = int(time.time())
                conn.execute(
                    "UPDATE bots SET reserved_usdt=?, updated_ts=? WHERE bot_id=?",
                    (str(new_res), now_ts, bot)
                )
                conn.commit()
                return {"ok": True, "reserved_usdt": str(new_res)}
            except Exception as e:
                try: conn.rollback()
                except Exception: pass
                return {"ok": False, "error": str(e)}
            finally:
                conn.close()

    def on_close(self, bot_id: str) -> Dict[str, Any]:
        """
        After a bot closes a trade:
          • refresh live wallet snapshot
          • zero this bot's reservation
          • re-divvy allocations
        """
        bot = str(bot_id).upper()
        live = self.read_contract_wallet_usdt()
        self.set_wallet_snapshot(live)

        with self._lock:
            conn = self._db()
            try:
                conn.execute("BEGIN IMMEDIATE")
                now_ts = int(time.time())
                conn.execute(
                    "UPDATE bots SET reserved_usdt=?, updated_ts=? WHERE bot_id=?",
                    ("0", now_ts, bot)
                )
                conn.commit()
            finally:
                conn.close()

        self.recalc_allocations()
        return {"ok": True, "wallet_usdt": str(live)}

    # ---------- Health ----------
    def health(self) -> Dict[str, Any]:
        conn = self._db()
        try:
            rows = conn.execute(
                "SELECT bot_id, weight, allocated_usdt, reserved_usdt, updated_ts FROM bots"
            ).fetchall()
            bots = [{
                "bot_id": r[0],
                "weight": r[1],
                "allocated_usdt": r[2],
                "reserved_usdt": r[3],
                "updated_ts": r[4],
            } for r in rows]
            return {
                "ok": True,
                "wallet_usdt": str(self.get_wallet_snapshot()),
                "bots": bots
            }
        finally:
            conn.close()


# ---------- Self-test (optional; safe to keep during bring-up) ----------
if __name__ == "__main__":
    cm = CapitalManager()
    print("OK: CapitalManager constructed")
    print("Initial health:", cm.health())

    # Example: configure two bots 50/50
    ok = cm.configure_bots({"BTC": "0.5", "SOL": "0.5"})
    print("configure_bots:", ok, cm.health())

    # Reserve 5% for BTC
    print("reserve BTC 5%:", cm.reserve("BTC", "0.05"))

    # Release 1.0 USDT from BTC
    print("release BTC 1.0:", cm.release("BTC", "1.0"))

    # Simulate close -> refresh wallet snapshot + zero reservation + re-alloc
    print("on_close BTC:", cm.on_close("BTC"))
    print("final health:", cm.health())


