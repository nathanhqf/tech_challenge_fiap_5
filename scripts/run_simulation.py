"""Simulação offline: compara Baseline, Thompson Sampling e Nilos-UCB.

Gera eventos simples em memória (sem dados sintéticos complexos) e registra
os experimentos no MLflow (Etapa 7). Salva o resumo em data/policy_comparison.csv.
"""
import logging
import os
import sys
from pathlib import Path

# Garante que a raiz do projeto esteja no path ao rodar `python scripts/run_simulation.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.datathon_offerexp.contracts import ALL_ARMS, SyntheticOfferEvent  # noqa: E402
from src.datathon_offerexp.evaluation import compare_policies, compute_fairness, compute_regret, replay_simulation  # noqa: E402
from src.datathon_offerexp.policies import BaselinePolicy, NilosUCBPolicy, ThompsonSamplingPolicy  # noqa: E402

logger = logging.getLogger(__name__)

MLFLOW_TAGS = {
    "model_type": "bandit",
    "framework": "custom",
    "owner": "datathon-7mlet-grupo",
    "phase": "datathon-fase05",
    "risk_level": "low",
    "fairness_checked": "true",
    "training_data_version": "bank-marketing-v1+synthetic",
}


def load_events(n: int = 2000, seed: int = 7) -> list[SyntheticOfferEvent]:
    """Gera n eventos simples em memória (segmento/canal aleatórios)."""
    rng = np.random.default_rng(seed)
    segments = ["novo", "recorrente", "reativado"]
    channels = ["app", "web", "telemarketing"]
    events = []
    for i in range(n):
        events.append(SyntheticOfferEvent(
            event_id=f"ev-{i:05d}",
            occurred_at="2024-01-01T00:00:00Z",
            subject_key=f"subj-{i}",
            channel=str(rng.choice(channels)),
            segment=str(rng.choice(segments)),
            available_arms=ALL_ARMS,
            context={"age_group": "30-40"},
        ))
    return events


def run(n_episodes: int = 2000) -> pd.DataFrame:
    """Executa simulação comparativa e loga no MLflow."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "offerexp-bandit"))

    events = load_events(n_episodes)
    policies = [
        BaselinePolicy(arms=ALL_ARMS),
        ThompsonSamplingPolicy(arms=ALL_ARMS),
        NilosUCBPolicy(arms=ALL_ARMS),
    ]

    results = []
    for policy in policies:
        with mlflow.start_run(run_name=policy.mode):
            mlflow.log_params({
                "policy": policy.mode,
                "n_episodes": n_episodes,
                "n_arms": len(ALL_ARMS),
            })
            for k, v in MLFLOW_TAGS.items():
                mlflow.set_tag(k, v)

            result = replay_simulation(events[:], policy)
            regret = compute_regret(result)
            fairness = compute_fairness(result)

            mlflow.log_metrics({
                "mean_reward": result.mean_reward,
                "total_reward": result.total_reward,
                "regret": regret,
                "exploration_rate": result.exploration_rate,
                "fairness_max_min_ratio": fairness.get("max_min_ratio", 0.0),
            })
            logger.info("%s → reward=%.4f regret=%.2f expl=%.1f%%",
                        policy.mode, result.mean_reward, regret, result.exploration_rate * 100)
            results.append(result)

    comparison = compare_policies(results)
    Path("data").mkdir(exist_ok=True)
    comparison.to_csv("data/policy_comparison.csv", index=False)
    logger.info("\n%s", comparison.to_string())
    return comparison


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = run()
    print("\n=== Comparação de Políticas ===")
    print(df.to_string(index=False))
