"""Contratos de dados da plataforma de experimentação adaptativa."""
from dataclasses import asdict, dataclass
from typing import Literal

Arm = Literal[
    "sem_oferta",
    "deposito_prazo_basico",
    "deposito_prazo_premium",
    "educacao_financeira",
]
Channel = Literal["app", "web", "telemarketing"]
Segment = Literal["novo", "recorrente", "reativado"]
PolicyMode = Literal["baseline", "thompson_sampling", "nilos_ucb"]

ALL_ARMS: tuple[Arm, ...] = (
    "sem_oferta",
    "deposito_prazo_basico",
    "deposito_prazo_premium",
    "educacao_financeira",
)


@dataclass(frozen=True, slots=True)
class SyntheticOfferEvent:
    """Evento sintético de oferta — representa uma oportunidade de decisão."""

    event_id: str
    occurred_at: str
    subject_key: str
    channel: Channel
    segment: Segment
    available_arms: tuple[Arm, ...]
    context: dict[str, str | float | bool]
    chosen_arm: Arm | None = None
    reward: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["available_arms"] = list(self.available_arms)
        return d


@dataclass(frozen=True, slots=True)
class DecisionLog:
    """Registro auditável de uma decisão da política."""

    decision_id: str
    event_id: str
    policy_version: str
    selected_arm: Arm
    mode: PolicyMode
    exploration: bool
    reason_codes: tuple[str, ...]
    created_at: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reason_codes"] = list(self.reason_codes)
        return d


@dataclass
class ArmStats:
    """Estatísticas de um braço: sucessos, falhas e pulls."""

    arm: Arm
    successes: float = 0.0
    failures: float = 0.0
    n_pulls: int = 0

    @property
    def mean_reward(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.0

    def update(self, reward: float) -> None:
        """Atualiza estatísticas com nova recompensa observada."""
        self.n_pulls += 1
        self.successes += reward
        self.failures += 1.0 - reward
