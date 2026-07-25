"""Testes dos contratos de dados."""
import pytest

from src.datathon_offerexp.contracts import ALL_ARMS, ArmStats, DecisionLog, SyntheticOfferEvent


def test_arm_stats_initial_mean():
    stats = ArmStats(arm="sem_oferta")
    assert stats.mean_reward == 0.0


def test_arm_stats_update_reward():
    stats = ArmStats(arm="deposito_prazo_premium")
    stats.update(1.0)
    stats.update(0.0)
    assert stats.n_pulls == 2
    assert stats.mean_reward == 0.5


def test_arm_stats_update_multiple():
    stats = ArmStats(arm="educacao_financeira")
    for r in [1.0, 1.0, 0.0, 1.0]:
        stats.update(r)
    assert stats.n_pulls == 4
    assert abs(stats.mean_reward - 0.75) < 1e-9


def test_synthetic_offer_event_immutable(sample_event):
    with pytest.raises((AttributeError, TypeError)):
        sample_event.event_id = "outro"  # type: ignore


def test_synthetic_offer_event_to_dict(sample_event):
    d = sample_event.to_dict()
    assert d["subject_key"] == "c001"
    assert d["segment"] == "recorrente"
    assert isinstance(d["available_arms"], list)


def test_decision_log_to_dict():
    log = DecisionLog(
        decision_id="d001",
        event_id="e001",
        policy_version="1.0.0",
        selected_arm="deposito_prazo_premium",
        mode="thompson_sampling",
        exploration=False,
        reason_codes=("thompson_sampling_sample",),
        created_at="2024-01-15T10:00:00+00:00",
    )
    d = log.to_dict()
    assert d["selected_arm"] == "deposito_prazo_premium"
    assert d["exploration"] is False


def test_all_arms_has_four():
    assert len(ALL_ARMS) == 4
    assert "sem_oferta" in ALL_ARMS
    assert "deposito_prazo_premium" in ALL_ARMS
