"""Avaliação offline: recompensa acumulada, regret e fairness."""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.datathon_offerexp.contracts import Arm, Segment, SyntheticOfferEvent

logger = logging.getLogger(__name__)

# Recompensa esperada por braço (simulação baseada em dados Bank Marketing)
# Fonte: arm_reward_multiplier × P(y=1) do dataset
ARM_BASE_REWARD: dict[Arm, float] = {
    "sem_oferta": 0.00,
    "deposito_prazo_basico": 0.08,
    "deposito_prazo_premium": 0.13,
    "educacao_financeira": 0.05,
}
OPTIMAL_ARM: Arm = "deposito_prazo_premium"


@dataclass
class SimulationResult:
    """Resultado de uma simulação offline de política."""

    policy_name: str
    n_episodes: int
    cumulative_rewards: list[float] = field(default_factory=list)
    arm_choices: list[Arm] = field(default_factory=list)
    exploration_flags: list[bool] = field(default_factory=list)
    segment_choices: list[Segment] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return sum(self.cumulative_rewards)

    @property
    def mean_reward(self) -> float:
        return np.mean(self.cumulative_rewards) if self.cumulative_rewards else 0.0

    @property
    def exploration_rate(self) -> float:
        if not self.exploration_flags:
            return 0.0
        return sum(self.exploration_flags) / len(self.exploration_flags)


def simulate_reward(arm: Arm, event: SyntheticOfferEvent, rng: np.random.Generator) -> float:
    """Simula recompensa Bernoulli para um braço dado o contexto do evento.

    Recompensa depende do braço e do segmento (proxies de propensão).
    """
    base = ARM_BASE_REWARD.get(arm, 0.0)
    # Segmentos com maior propensão
    segment_boost = {"recorrente": 1.5, "reativado": 0.8, "novo": 1.0}.get(
        event.segment, 1.0
    )
    p = min(base * segment_boost, 1.0)
    return float(rng.random() < p)


def replay_simulation(
    events: list[SyntheticOfferEvent],
    policy,
    seed: int = 42,
) -> SimulationResult:
    """Simula política sobre lista de eventos em ordem cronológica.

    Args:
        events: Eventos sintéticos a processar.
        policy: Política com métodos select(event) e update(arm, reward).
        seed: Semente para reprodutibilidade.

    Returns:
        SimulationResult com histórico completo.
    """
    rng = np.random.default_rng(seed)
    result = SimulationResult(
        policy_name=policy.mode,
        n_episodes=len(events),
    )

    for event in events:
        arm, is_exploration, _ = policy.select(event)
        reward = simulate_reward(arm, event, rng)
        policy.update(arm, reward)

        result.cumulative_rewards.append(reward)
        result.arm_choices.append(arm)
        result.exploration_flags.append(is_exploration)
        result.segment_choices.append(event.segment)

    logger.info(
        "Simulação '%s': reward=%.4f, exploração=%.1f%%",
        policy.mode,
        result.mean_reward,
        result.exploration_rate * 100,
    )
    return result


def compute_regret(result: SimulationResult, optimal_reward: float | None = None) -> float:
    """Calcula regret acumulado vs braço ótimo.

    Regret = Σ (E[r*] - E[rₜ]) onde r* = recompensa do braço ótimo.
    """
    opt = optimal_reward or ARM_BASE_REWARD[OPTIMAL_ARM]
    return max(0.0, opt * result.n_episodes - result.total_reward)


def compute_fairness(result: SimulationResult) -> dict[str, float]:
    """Calcula equilíbrio de exposição entre segmentos.

    Métrica: razão entre exposição máxima e mínima por segmento.
    Valor ideal = 1.0 (exposição perfeitamente equilibrada).
    """
    if not result.segment_choices:
        return {}

    from collections import Counter

    counts = Counter(result.segment_choices)
    total = len(result.segment_choices)
    proportions = {seg: n / total for seg, n in counts.items()}
    values = list(proportions.values())

    return {
        "exposure_by_segment": proportions,
        "max_min_ratio": max(values) / min(values) if min(values) > 0 else float("inf"),
    }


def compare_policies(results: list[SimulationResult]) -> pd.DataFrame:
    """Gera tabela comparativa entre políticas.

    Args:
        results: Lista de SimulationResult de diferentes políticas.

    Returns:
        DataFrame com métricas por política.
    """
    rows = []
    for r in results:
        rows.append({
            "policy": r.policy_name,
            "mean_reward": round(r.mean_reward, 4),
            "total_reward": round(r.total_reward, 2),
            "regret": round(compute_regret(r), 2),
            "exploration_rate": round(r.exploration_rate, 3),
            "n_episodes": r.n_episodes,
        })
    df = pd.DataFrame(rows).sort_values("mean_reward", ascending=False)
    return df
