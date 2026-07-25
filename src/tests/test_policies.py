"""Testes das políticas de decisão."""
import pytest

from src.datathon_offerexp.contracts import ALL_ARMS
from src.datathon_offerexp.policies import BaselinePolicy, NilosUCBPolicy, ThompsonSamplingPolicy


# ── Baseline ────────────────────────────────────────────────────────────────

def test_baseline_cold_start(sample_event):
    policy = BaselinePolicy(arms=ALL_ARMS)
    arm, is_exploration, codes = policy.select(sample_event)
    assert arm in ALL_ARMS
    assert is_exploration is True
    assert "cold_start" in codes


def test_baseline_chooses_best_arm(sample_event):
    policy = BaselinePolicy(arms=ALL_ARMS)
    # Treina com recompensas favorecendo deposito_prazo_premium
    for arm in ALL_ARMS:
        policy.stats[arm].update(0.0)
    policy.stats["deposito_prazo_premium"].update(1.0)
    policy.stats["deposito_prazo_premium"].update(1.0)

    arm, is_exploration, _ = policy.select(sample_event)
    assert arm == "deposito_prazo_premium"
    assert is_exploration is False


def test_baseline_returns_arm_in_available(sample_event):
    policy = BaselinePolicy(arms=ALL_ARMS)
    arm, _, _ = policy.select(sample_event)
    assert arm in sample_event.available_arms


# ── Thompson Sampling ────────────────────────────────────────────────────────

def test_thompson_cold_start_uses_uniform_prior(sample_event):
    import numpy as np
    policy = ThompsonSamplingPolicy(arms=ALL_ARMS, rng=np.random.default_rng(0))
    arm, _, codes = policy.select(sample_event)
    assert arm in ALL_ARMS
    assert "thompson_sampling_sample" in codes


def test_thompson_update_shifts_stats(sample_event):
    import numpy as np
    policy = ThompsonSamplingPolicy(arms=ALL_ARMS, rng=np.random.default_rng(42))
    policy.update("deposito_prazo_premium", 1.0)
    s = policy.stats["deposito_prazo_premium"]
    assert s.successes == 1.0
    assert s.failures == 0.0
    assert s.n_pulls == 1


def test_thompson_converges_to_best_arm():
    """Com muitas amostras, Thompson deve escolher o melhor braço na maioria."""
    import numpy as np
    from src.datathon_offerexp.contracts import SyntheticOfferEvent

    policy = ThompsonSamplingPolicy(arms=ALL_ARMS, rng=np.random.default_rng(99))
    # Simula: premium tem recompensa alta, outros baixa
    for _ in range(200):
        policy.update("deposito_prazo_premium", 1.0)
        policy.update("sem_oferta", 0.0)
        policy.update("deposito_prazo_basico", 0.1)
        policy.update("educacao_financeira", 0.05)

    event = SyntheticOfferEvent(
        event_id="e", occurred_at="2024-01-01T00:00:00Z",
        subject_key="x", channel="app", segment="novo",
        available_arms=ALL_ARMS, context={},
    )
    choices = [policy.select(event)[0] for _ in range(50)]
    assert choices.count("deposito_prazo_premium") > 30


# ── Nilos-UCB ────────────────────────────────────────────────────────────────

def test_ucb_cold_start(sample_event):
    policy = NilosUCBPolicy(arms=ALL_ARMS)
    arm, is_exploration, codes = policy.select(sample_event)
    assert arm in ALL_ARMS
    assert is_exploration is True
    assert "cold_start_ucb" in codes


def test_ucb_explores_untried_arms_first(sample_event):
    policy = NilosUCBPolicy(arms=ALL_ARMS)
    seen_arms = set()
    for _ in range(len(ALL_ARMS)):
        arm, _, _ = policy.select(sample_event)
        policy.update(arm, 0.0)
        seen_arms.add(arm)
    assert seen_arms == set(ALL_ARMS)


def test_ucb_update_increments_total_pulls(sample_event):
    policy = NilosUCBPolicy(arms=ALL_ARMS)
    policy.update("sem_oferta", 0.0)
    policy.update("deposito_prazo_basico", 1.0)
    assert policy._total_pulls == 2


# ── Geral ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("PolicyClass", [BaselinePolicy, ThompsonSamplingPolicy, NilosUCBPolicy])
def test_policy_raises_on_empty_arms(PolicyClass):
    from src.datathon_offerexp.contracts import SyntheticOfferEvent
    policy = PolicyClass(arms=ALL_ARMS)
    empty_event = SyntheticOfferEvent(
        event_id="e", occurred_at="2024-01-01T00:00:00Z",
        subject_key="x", channel="app", segment="novo",
        available_arms=(), context={},
    )
    with pytest.raises(ValueError):
        policy.select(empty_event)
