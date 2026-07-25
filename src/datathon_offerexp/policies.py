"""Políticas de decisão adaptativa: Baseline, Thompson Sampling e Nilos-UCB.

Referências:
  Thompson (1933) — On the likelihood that one unknown probability exceeds another.
  Auer et al. (2002) — Finite-time Analysis of the Multiarmed Bandit Problem (UCB1).
"""
import logging
import math
import random

import numpy as np

from src.datathon_offerexp.contracts import Arm, ArmStats, PolicyMode, SyntheticOfferEvent

logger = logging.getLogger(__name__)

POLICY_VERSION = "1.0.0"


def _init_stats(arms: tuple[Arm, ...]) -> dict[Arm, ArmStats]:
    return {arm: ArmStats(arm=arm) for arm in arms}


class BaselinePolicy:
    """Política determinística: escolhe o braço com maior recompensa média histórica.

    Serve como controle para comparação com políticas adaptativas.
    Cold-start: braço com mais pulls ou primeiro disponível se empate.
    """

    mode: PolicyMode = "baseline"

    def __init__(self, arms: tuple[Arm, ...], rng: random.Random | None = None):
        self.stats: dict[Arm, ArmStats] = _init_stats(arms)
        self.rng = rng or random.Random(42)

    def select(self, event: SyntheticOfferEvent) -> tuple[Arm, bool, tuple[str, ...]]:
        """Seleciona braço com maior recompensa média.

        Returns:
            Tupla (arm, exploration_flag, reason_codes).
        """
        available = [a for a in event.available_arms if a in self.stats]
        if not available:
            raise ValueError(f"Nenhum braço disponível: {event.available_arms}")

        # Cold-start: pull counts iguais → escolhe aleatório
        untried = [a for a in available if self.stats[a].n_pulls == 0]
        if untried:
            arm = self.rng.choice(untried)
            return arm, True, ("cold_start",)

        best = max(available, key=lambda a: self.stats[a].mean_reward)
        return best, False, ("maior_recompensa_media",)

    def update(self, arm: Arm, reward: float) -> None:
        """Atualiza estatísticas do braço após observar recompensa."""
        self.stats[arm].update(reward)


class ThompsonSamplingPolicy:
    """Thompson Sampling com prior Beta(1,1) — exploração bayesiana.

    Para cada braço, amostra θ ~ Beta(α, β) e escolhe argmax(θ).
    Trata-se de um estimador Bayesiano natural para recompensas Bernoulli.
    Cold-start tratado automaticamente pelo prior uniforme Beta(1,1).
    """

    mode: PolicyMode = "thompson_sampling"

    def __init__(
        self,
        arms: tuple[Arm, ...],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.stats: dict[Arm, ArmStats] = _init_stats(arms)
        self.rng = rng or np.random.default_rng(42)

    def select(self, event: SyntheticOfferEvent) -> tuple[Arm, bool, tuple[str, ...]]:
        """Amostra de Beta(α, β) por braço e retorna argmax.

        Returns:
            Tupla (arm, exploration_flag, reason_codes).
        """
        available = [a for a in event.available_arms if a in self.stats]
        if not available:
            raise ValueError(f"Nenhum braço disponível: {event.available_arms}")

        samples: dict[Arm, float] = {}
        for arm in available:
            s = self.stats[arm]
            alpha = self.prior_alpha + s.successes
            b = self.prior_beta + s.failures
            samples[arm] = self.rng.beta(alpha, b)

        best = max(available, key=lambda a: samples[a])
        # É exploração se o braço com maior sample não é o de maior média empírica
        empirical_best = max(available, key=lambda a: self.stats[a].mean_reward)
        is_exploration = best != empirical_best

        logger.debug("TS samples=%s → %s", {a: f"{v:.3f}" for a, v in samples.items()}, best)
        return best, is_exploration, ("thompson_sampling_sample",)

    def update(self, arm: Arm, reward: float) -> None:
        self.stats[arm].update(reward)


class NilosUCBPolicy:
    """Nilos-UCB — variante UCB1 para o desafio.

    Fórmula: score(arm) = x̄ᵢ + C × √(ln(N) / nᵢ)

    Onde:
      x̄ᵢ = recompensa média do braço i
      N   = total de pulls
      nᵢ  = pulls do braço i
      C   = constante de exploração (padrão √2)

    Referência: Auer et al. (2002), UCB1.
    """

    mode: PolicyMode = "nilos_ucb"

    def __init__(
        self,
        arms: tuple[Arm, ...],
        exploration_constant: float = math.sqrt(2),
    ):
        self.C = exploration_constant
        self.stats: dict[Arm, ArmStats] = _init_stats(arms)
        self._total_pulls: int = 0

    def select(self, event: SyntheticOfferEvent) -> tuple[Arm, bool, tuple[str, ...]]:
        """Calcula UCB score por braço e retorna argmax.

        Returns:
            Tupla (arm, exploration_flag, reason_codes).
        """
        available = [a for a in event.available_arms if a in self.stats]
        if not available:
            raise ValueError(f"Nenhum braço disponível: {event.available_arms}")

        # Cold-start: pulls==0 → UCB = +∞, explorar primeiro
        untried = [a for a in available if self.stats[a].n_pulls == 0]
        if untried:
            arm = untried[0]
            return arm, True, ("cold_start_ucb",)

        def ucb_score(arm: Arm) -> float:
            s = self.stats[arm]
            return s.mean_reward + self.C * math.sqrt(math.log(self._total_pulls) / s.n_pulls)

        scores = {a: ucb_score(a) for a in available}
        best = max(available, key=lambda a: scores[a])
        empirical_best = max(available, key=lambda a: self.stats[a].mean_reward)
        is_exploration = best != empirical_best

        logger.debug("UCB scores=%s → %s", {a: f"{v:.3f}" for a, v in scores.items()}, best)
        return best, is_exploration, ("nilos_ucb_score",)

    def update(self, arm: Arm, reward: float) -> None:
        self.stats[arm].update(reward)
        self._total_pulls += 1
