"""Testes da avaliação offline (simulação, regret, fairness, comparação)."""
import numpy as np

from src.datathon_offerexp.contracts import ALL_ARMS, SyntheticOfferEvent
from src.datathon_offerexp.evaluation import (
    SimulationResult,
    compare_policies,
    compute_fairness,
    compute_regret,
    replay_simulation,
    simulate_reward,
)
from src.datathon_offerexp.policies import BaselinePolicy, ThompsonSamplingPolicy


def _event(segment: str = "novo", arms: tuple = ALL_ARMS) -> SyntheticOfferEvent:
    return SyntheticOfferEvent(
        event_id="e",
        occurred_at="2024-01-01T00:00:00Z",
        subject_key="s",
        channel="app",
        segment=segment,
        available_arms=arms,
        context={},
    )


def test_simulate_reward_is_binary():
    rng = np.random.default_rng(0)
    vals = {simulate_reward("deposito_prazo_premium", _event(), rng) for _ in range(50)}
    assert vals <= {0.0, 1.0}


def test_simulate_reward_sem_oferta_is_zero():
    rng = np.random.default_rng(0)
    assert simulate_reward("sem_oferta", _event(), rng) == 0.0


def test_replay_simulation_runs():
    events = [_event() for _ in range(100)]
    result = replay_simulation(events, ThompsonSamplingPolicy(arms=ALL_ARMS))
    assert result.n_episodes == 100
    assert len(result.arm_choices) == 100
    assert 0.0 <= result.exploration_rate <= 1.0
    assert result.total_reward >= 0.0


def test_compute_regret_non_negative():
    events = [_event() for _ in range(50)]
    result = replay_simulation(events, BaselinePolicy(arms=ALL_ARMS))
    assert compute_regret(result) >= 0.0


def test_compute_fairness_ratio_ge_one():
    events = [_event(segment=s) for s in ["novo", "recorrente", "reativado"] * 10]
    result = replay_simulation(events, ThompsonSamplingPolicy(arms=ALL_ARMS))
    fairness = compute_fairness(result)
    assert fairness["max_min_ratio"] >= 1.0


def test_compare_policies_has_expected_columns():
    events = [_event() for _ in range(80)]
    results = [
        replay_simulation(events[:], BaselinePolicy(arms=ALL_ARMS)),
        replay_simulation(events[:], ThompsonSamplingPolicy(arms=ALL_ARMS)),
    ]
    df = compare_policies(results)
    assert {"policy", "mean_reward", "regret"} <= set(df.columns)
    assert len(df) == 2


def test_simulation_result_empty_defaults():
    r = SimulationResult(policy_name="x", n_episodes=0)
    assert r.mean_reward == 0.0
    assert r.exploration_rate == 0.0
    assert r.total_reward == 0.0
