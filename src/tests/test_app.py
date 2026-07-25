"""Testes da API FastAPI (Etapa 5 — serviço demonstrável)."""
import pytest
from fastapi.testclient import TestClient

from src.datathon_offerexp.app import app
from src.datathon_offerexp.contracts import ALL_ARMS


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DECISION_LOG_PATH", str(tmp_path / "decisions.db"))
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_decide_returns_valid_arm(client):
    r = client.post("/decide", json={
        "subject_key": "c1",
        "channel": "app",
        "segment": "novo",
        "context": {"age_group": "30-40"},
        "mode": "thompson_sampling",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["selected_arm"] in ALL_ARMS
    assert "decision_id" in body
    assert body["mode"] == "thompson_sampling"


def test_decide_reward_status_flow(client):
    decision = client.post("/decide", json={"subject_key": "c2", "mode": "baseline"}).json()
    r = client.post("/reward", json={"decision_id": decision["decision_id"], "reward": 1.0})
    assert r.status_code == 204
    status = client.get("/status").json()
    assert status["total_decisions"] >= 1
    assert "arm_stats" in status


def test_reward_unknown_decision_returns_404(client):
    r = client.post("/reward", json={"decision_id": "nao-existe", "reward": 1.0})
    assert r.status_code == 404


def test_decide_invalid_mode_returns_422(client):
    r = client.post("/decide", json={"subject_key": "c3", "mode": "modo_invalido"})
    assert r.status_code == 422


def test_reward_out_of_range_returns_422(client):
    decision = client.post("/decide", json={"subject_key": "c4", "mode": "baseline"}).json()
    r = client.post("/reward", json={"decision_id": decision["decision_id"], "reward": 2.5})
    assert r.status_code == 422
