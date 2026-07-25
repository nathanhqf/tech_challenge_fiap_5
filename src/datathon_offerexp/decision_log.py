"""Log auditável de decisões — persistência em SQLite."""
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.datathon_offerexp.contracts import DecisionLog

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/decisions.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id   TEXT PRIMARY KEY,
    event_id      TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    selected_arm  TEXT NOT NULL,
    mode          TEXT NOT NULL,
    exploration   INTEGER NOT NULL,
    reason_codes  TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rewards (
    decision_id TEXT PRIMARY KEY,
    reward      REAL NOT NULL,
    recorded_at TEXT NOT NULL
);
"""


class DecisionLogger:
    """Log auditável de decisões e recompensas em SQLite."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def log_decision(self, log: DecisionLog) -> None:
        """Persiste registro de decisão."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?,?,?)",
                (
                    log.decision_id,
                    log.event_id,
                    log.policy_version,
                    log.selected_arm,
                    log.mode,
                    int(log.exploration),
                    json.dumps(list(log.reason_codes)),
                    log.created_at,
                ),
            )
        logger.debug("Decisão registrada: %s → %s", log.decision_id, log.selected_arm)

    def log_reward(self, decision_id: str, reward: float) -> None:
        """Registra recompensa observada para uma decisão."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO rewards VALUES (?,?,?)",
                (decision_id, reward, now),
            )
        logger.debug("Recompensa registrada: %s = %.3f", decision_id, reward)

    def get_arm_stats(self) -> dict[str, dict]:
        """Retorna estatísticas agregadas por braço."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT d.selected_arm,
                       COUNT(*) as n_decisions,
                       SUM(CASE WHEN r.reward IS NOT NULL THEN 1 ELSE 0 END) as n_rewarded,
                       AVG(r.reward) as mean_reward,
                       SUM(d.exploration) as n_explorations
                FROM decisions d
                LEFT JOIN rewards r ON d.decision_id = r.decision_id
                GROUP BY d.selected_arm
            """).fetchall()

        return {
            row[0]: {
                "n_decisions": row[1],
                "n_rewarded": row[2],
                "mean_reward": round(row[3] or 0.0, 4),
                "exploration_rate": round(row[4] / row[1], 3) if row[1] > 0 else 0.0,
            }
            for row in rows
        }

    def get_decision(self, decision_id: str) -> dict | None:
        """Recupera decisão por ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM decisions WHERE decision_id=?", (decision_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "decision_id": row[0],
            "event_id": row[1],
            "policy_version": row[2],
            "selected_arm": row[3],
            "mode": row[4],
            "exploration": bool(row[5]),
            "reason_codes": json.loads(row[6]),
            "created_at": row[7],
        }

    def count_decisions(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
