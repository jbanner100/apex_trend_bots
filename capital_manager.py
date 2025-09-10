# capital_manager.py — v1.4a (engine; no envs)
import time
import sqlite3
import threading
from decimal import Decimal, getcontext

# ====== CREDENTIALS — FILL THESE IN ======
api_creds = {
    "key": "3e965beb-41e2-f125-a7c5-569f45bfba21",
    "secret": "NXtuAyq4hS9G4fVlytQtQGn9Qk5LUukXGAYg8SBj",
    "passphrase": "GEy6yNGaZ5_0fuX4VBJ3"
}
zk_seeds = "0xd00ec9396facbafc423b5d92a289ea49adfdb0b918d3d5db26edbb978893ed5d0bd48c3fbe4309de2a09a3514cbec6d2c4012df85653a1421aca1cf599acda491c"
zk_l2Key = "0xd6094a658c50dccf9be8f85cde1804e92a74a0482788766c9b8744cbc6fe8501"

# =========================================

# ====== DB LOCATION (no envs) ======
DB_PATH_DEFAULT = "./capalloc.db"
# ===================================

# ApeX SDK
from apexomni.constants import APEX_OMNI_HTTP_MAIN, NETWORKID_OMNI_MAIN_ARB
from apexomni.http_private_sign import HttpPrivateSign

getcontext().prec = 28
D = lambda x: x if isinstance(x, Decimal) else Decimal(str(x))


def _connect(db_path: str):
    conn = sqlite3.connect(
        db_path,
        timeout=5.0,
        isolation_level=None,   # we control transactions manually
        check_same_thread=False,
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


class CapitalManager:
    """
    Manages:
      • wallet snapshot (contract USDT)
      • per-bot weights & normalized allocations
      • atomic reservation/release for each bot
    SQLite (WAL) + BEGIN IMMEDIATE for multi-proc safety.
    """
    def __init__(self, db_path: str = DB_PATH_DEFAULT):
        self.db_path = db_path
        self._lock = threading.Lock()

        # ApeX client (no envs; from constants above)
        self.client = HttpPrivateSign(
            APEX_OMNI_HTTP_MAIN,
            network_id=NETWORKID_OMNI_MAIN_ARB,
            api_key_credentials=API_CREDS,
            zk_seeds=ZK_SEEDS,
            zk_l2Key=ZK_L2KEY,
        )
        self.client.configs_v3()

        self._init_db()

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
                # best-effort snapshot now
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
        try:
            acct = self.client.get_account_v3()
            for w in (acct.get("contractWallets") or []):
                if w.get("token") == "USDT":
                    return D(w.get("balance", "0"))
        except Exception:
            pass
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
                (str(amount),)
            )
            conn.commit()
        finally:
            conn.close()

    # ---------- Configure & allocation ----------
    def configure_bots(self, weights_dict: dict) -> bool:
        """
        Replace bot set with given weights.
        Example: {"BTC": "0.5", "SOL": "0.5"}
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
                    conn.execute(
                        """INSERT INTO bots(bot_id, weight, allocated_usdt, reserved_usdt, updated_ts)
                           VALUES(?,?,?,?,?)
                           ON CONFLICT(bot_id) DO UPDATE SET weight=excluded.weight""",
                        (str(bot_id).upper(), str(w), "0", "0", now_ts)
                    )
                # remove any that aren't in the new set
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
    def reserve(self, bot_id: str, trade_pct: str) -> dict:
        """
        Reserve margin from bot's allocation.
        Returns {ok, approved_margin_usdt, allocated_usdt, reserved_usdt, available_usdt}
        """
        bot = str(bot_id).upper()
        t_pct = D(trade_pct)

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

                requested = allocated * t_pct
                approved  = requested if requested <= available else available

                if approved > D("0"):
                    now_ts = int(time.time())
                    conn.execute(
                        "UPDATE bots SET reserved_usdt=?, updated_ts=? WHERE bot_id=?",
                        (str(reserved + approved), now_ts, bot)
                    )
                    conn.commit()
                    new_reserved = reserved + approved
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

    def release(self, bot_id: str, amount_usdt: str) -> dict:
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

    def on_close(self, bot_id: str) -> dict:
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
    def health(self) -> dict:
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
