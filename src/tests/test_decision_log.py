"""Testes do log auditável de decisões."""
import pytest

from src.datathon_offerexp.contracts import DecisionLog
from src.datathon_offerexp.decision_log import DecisionLogger


def _make_log(n: int = 1) -> DecisionLog:
    return DecisionLog(
        decision_id=f"dec-{n:03d}",
        event_id=f"evt-{n:03d}",
        policy_version="1.0.0",
        selected_arm="deposito_prazo_premium",
        mode="thompson_sampling",
        exploration=False,
        reason_codes=("thompson_sampling_sample",),
        created_at="2024-01-15T10:00:00+00:00",
    )


def test_log_and_retrieve_decision(tmp_db):
    logger = DecisionLogger(tmp_db)
    log = _make_log(1)
    logger.log_decision(log)
    result = logger.get_decision("dec-001")
    assert result is not None
    assert result["selected_arm"] == "deposito_prazo_premium"
    assert result["mode"] == "thompson_sampling"


def test_log_reward(tmp_db):
    logger = DecisionLogger(tmp_db)
    logger.log_decision(_make_log(1))
    logger.log_reward("dec-001", 1.0)
    stats = logger.get_arm_stats()
    assert "deposito_prazo_premium" in stats
    assert stats["deposito_prazo_premium"]["mean_reward"] == 1.0


def test_count_decisions(tmp_db):
    logger = DecisionLogger(tmp_db)
    assert logger.count_decisions() == 0
    logger.log_decision(_make_log(1))
    logger.log_decision(_make_log(2))
    assert logger.count_decisions() == 2


def test_get_nonexistent_decision(tmp_db):
    logger = DecisionLogger(tmp_db)
    assert logger.get_decision("nao-existe") is None


def test_arm_stats_empty(tmp_db):
    logger = DecisionLogger(tmp_db)
    assert logger.get_arm_stats() == {}


def test_multiple_decisions_arm_stats(tmp_db):
    logger = DecisionLogger(tmp_db)
    for i in range(3):
        log = DecisionLog(
            decision_id=f"d{i}",
            event_id=f"e{i}",
            policy_version="1.0.0",
            selected_arm="sem_oferta",
            mode="baseline",
            exploration=False,
            reason_codes=("maior_recompensa_media",),
            created_at="2024-01-15T10:00:00+00:00",
        )
        logger.log_decision(log)
        logger.log_reward(f"d{i}", float(i % 2))

    stats = logger.get_arm_stats()
    assert stats["sem_oferta"]["n_decisions"] == 3
