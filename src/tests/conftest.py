"""Fixtures compartilhadas para os testes."""
import pytest

from src.datathon_offerexp.contracts import ALL_ARMS, SyntheticOfferEvent


@pytest.fixture
def sample_event() -> SyntheticOfferEvent:
    return SyntheticOfferEvent(
        event_id="evt-001",
        occurred_at="2024-01-15T10:00:00+00:00",
        subject_key="c001",
        channel="app",
        segment="recorrente",
        available_arms=ALL_ARMS,
        context={"age_group": "35-50", "job_group": "management"},
    )


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_decisions.db")
