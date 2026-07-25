"""API FastAPI — serviço de decisão adaptativa."""
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.datathon_offerexp.contracts import ALL_ARMS, Arm, Channel, DecisionLog, PolicyMode, Segment, SyntheticOfferEvent
from src.datathon_offerexp.decision_log import DecisionLogger
from src.datathon_offerexp.policies import BaselinePolicy, NilosUCBPolicy, ThompsonSamplingPolicy

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

# Estado global da política (em produção: carregar do MLflow)
_policies: dict[PolicyMode, object] = {}
_log: DecisionLogger | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _policies, _log
    _policies = {
        "baseline": BaselinePolicy(arms=ALL_ARMS),
        "thompson_sampling": ThompsonSamplingPolicy(arms=ALL_ARMS),
        "nilos_ucb": NilosUCBPolicy(arms=ALL_ARMS),
    }
    _log = DecisionLogger(os.environ.get("DECISION_LOG_PATH", "data/decisions.db"))
    logger.info("Plataforma iniciada. Políticas: %s", list(_policies.keys()))
    yield


app = FastAPI(
    title="OfferExp — Plataforma de Experimentação Adaptativa",
    description="Seleciona braço de oferta via Multi-Armed Bandit e registra recompensas.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class DecideRequest(BaseModel):
    subject_key: str = Field(..., description="Identificador anonimizado do cliente")
    channel: Channel = "app"
    segment: Segment = "novo"
    context: dict[str, str | float | bool] = Field(default_factory=dict)
    available_arms: list[Arm] | None = None
    mode: PolicyMode = "thompson_sampling"


class DecideResponse(BaseModel):
    decision_id: str
    selected_arm: Arm
    mode: PolicyMode
    exploration: bool
    reason_codes: list[str]
    policy_version: str


class RewardRequest(BaseModel):
    decision_id: str
    reward: float = Field(..., ge=0.0, le=1.0)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/decide", response_model=DecideResponse)
def decide(req: DecideRequest) -> DecideResponse:
    """Seleciona braço de oferta para o evento recebido."""
    policy = _policies.get(req.mode)
    if policy is None:
        raise HTTPException(400, f"Modo inválido: {req.mode}")

    arms = tuple(req.available_arms) if req.available_arms else ALL_ARMS
    event = SyntheticOfferEvent(
        event_id=str(uuid.uuid4()),
        occurred_at=datetime.now(timezone.utc).isoformat(),
        subject_key=req.subject_key,
        channel=req.channel,
        segment=req.segment,
        available_arms=arms,
        context=req.context,
    )

    arm, is_exploration, reason_codes = policy.select(event)  # type: ignore[union-attr]
    decision_id = str(uuid.uuid4())

    log_entry = DecisionLog(
        decision_id=decision_id,
        event_id=event.event_id,
        policy_version="1.0.0",
        selected_arm=arm,
        mode=req.mode,
        exploration=is_exploration,
        reason_codes=reason_codes,
        created_at=event.occurred_at,
    )
    _log.log_decision(log_entry)  # type: ignore[union-attr]

    return DecideResponse(
        decision_id=decision_id,
        selected_arm=arm,
        mode=req.mode,
        exploration=is_exploration,
        reason_codes=list(reason_codes),
        policy_version="1.0.0",
    )


@app.post("/reward", status_code=204)
def register_reward(req: RewardRequest) -> None:
    """Registra recompensa observada para uma decisão e atualiza a política."""
    decision = _log.get_decision(req.decision_id)  # type: ignore[union-attr]
    if not decision:
        raise HTTPException(404, f"decision_id não encontrado: {req.decision_id}")

    _log.log_reward(req.decision_id, req.reward)  # type: ignore[union-attr]

    # Atualiza política correspondente
    mode: PolicyMode = decision["mode"]
    arm: Arm = decision["selected_arm"]
    if mode in _policies:
        _policies[mode].update(arm, req.reward)  # type: ignore[union-attr]


@app.get("/status")
def status() -> dict:
    """Retorna estatísticas atuais dos braços e total de decisões."""
    return {
        "total_decisions": _log.count_decisions(),  # type: ignore[union-attr]
        "arm_stats": _log.get_arm_stats(),  # type: ignore[union-attr]
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "decisions": _log.count_decisions() if _log else 0}  # type: ignore[union-attr]
