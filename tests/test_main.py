import groq
import pytest
from fastapi.testclient import TestClient

from app import audit_log, repository
from app.db import get_connection
from app.fallback import FallbackExtractor
from app.main import app, get_service
from app.models import ExtractionResult
from app.service import TicketService


@pytest.fixture
def conn():
    # TestClient runs sync endpoints in a worker thread, so this connection
    # (unlike the shared conftest one) must tolerate cross-thread use.
    connection = get_connection(":memory:", check_same_thread=False)
    repository.init_schema(connection)
    audit_log.init_schema(connection)
    yield connection
    connection.close()


class ScriptedProvider:
    name = "groq"

    def __init__(self, script):
        self._script = list(script)

    def extract(self, message: str) -> ExtractionResult:
        outcome = self._script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_extraction(**overrides) -> ExtractionResult:
    defaults = dict(category="other", urgency="low", sentiment="calm", summary="s", confidence=0.9)
    defaults.update(overrides)
    return ExtractionResult(**defaults)


def with_script(conn, script):
    """Build a TestClient whose service is wired to a scripted provider."""
    service = TicketService(conn=conn, extractor=FallbackExtractor(providers=[ScriptedProvider(script)]))
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


class TestSubmitTicket:
    def test_submit_returns_created_ticket(self, conn):
        client = with_script(conn, [make_extraction(category="safety")])
        response = client.post("/tickets", json={"rider_id": "rider-1", "message": "help, threatened"})
        assert response.status_code == 200
        body = response.json()
        assert body["state"] == "ESCALATED_URGENT"
        assert body["decision"]["rule_id"] == "SAFETY_ALWAYS_URGENT"
        app.dependency_overrides.clear()

    def test_all_providers_failing_returns_503(self, conn):
        client = with_script(conn, [groq.APITimeoutError(request=object())])
        response = client.post("/tickets", json={"rider_id": "rider-1", "message": "hello"})
        assert response.status_code == 503
        app.dependency_overrides.clear()


class TestGetTicket:
    def test_get_returns_submitted_ticket(self, conn):
        client = with_script(conn, [make_extraction(category="other")])
        created = client.post("/tickets", json={"rider_id": "rider-1", "message": "hi"}).json()

        response = client.get(f"/tickets/{created['id']}")
        assert response.status_code == 200
        assert response.json()["id"] == created["id"]
        app.dependency_overrides.clear()

    def test_get_unknown_ticket_returns_404(self, conn):
        client = with_script(conn, [])
        response = client.get("/tickets/does-not-exist")
        assert response.status_code == 404
        app.dependency_overrides.clear()


class TestAddMessage:
    def test_followup_escalates_the_case(self, conn):
        client = with_script(
            conn, [make_extraction(category="customer_dispute"), make_extraction(category="safety")]
        )
        created = client.post("/tickets", json={"rider_id": "rider-1", "message": "no payment"}).json()
        assert created["state"] == "ESCALATED_ROUTINE"

        response = client.post(
            f"/tickets/{created['id']}/messages",
            json={"rider_id": "rider-1", "message": "also threatened me"},
        )
        assert response.status_code == 200
        assert response.json()["state"] == "ESCALATED_URGENT"
        app.dependency_overrides.clear()

    def test_unknown_ticket_returns_404(self, conn):
        client = with_script(conn, [])
        response = client.post(
            "/tickets/does-not-exist/messages", json={"rider_id": "rider-1", "message": "hi"}
        )
        assert response.status_code == 404
        app.dependency_overrides.clear()

    def test_wrong_rider_returns_404(self, conn):
        client = with_script(conn, [make_extraction(category="other")])
        created = client.post("/tickets", json={"rider_id": "rider-1", "message": "hi"}).json()

        response = client.post(
            f"/tickets/{created['id']}/messages",
            json={"rider_id": "someone-else", "message": "not mine"},
        )
        assert response.status_code == 404
        app.dependency_overrides.clear()


class TestAuditLog:
    def test_returns_entries_for_ticket(self, conn):
        client = with_script(conn, [make_extraction(category="safety")])
        created = client.post("/tickets", json={"rider_id": "rider-1", "message": "help"}).json()

        response = client.get(f"/tickets/{created['id']}/audit-log")
        assert response.status_code == 200
        actors = [e["actor"] for e in response.json()]
        assert actors == ["llm_extraction", "policy_engine", "state_machine"]
        app.dependency_overrides.clear()

    def test_unknown_ticket_returns_404(self, conn):
        client = with_script(conn, [])
        response = client.get("/tickets/does-not-exist/audit-log")
        assert response.status_code == 404
        app.dependency_overrides.clear()


class TestResolveTicket:
    def test_resolves_an_escalated_ticket(self, conn):
        client = with_script(conn, [make_extraction(category="safety")])
        created = client.post("/tickets", json={"rider_id": "rider-1", "message": "help"}).json()

        response = client.post(f"/tickets/{created['id']}/resolve")
        assert response.status_code == 200
        assert response.json()["state"] == "CLOSED"
        app.dependency_overrides.clear()

    def test_unknown_ticket_returns_404(self, conn):
        client = with_script(conn, [])
        response = client.post("/tickets/does-not-exist/resolve")
        assert response.status_code == 404
        app.dependency_overrides.clear()

    def test_double_resolve_returns_409(self, conn):
        client = with_script(conn, [make_extraction(category="safety")])
        created = client.post("/tickets", json={"rider_id": "rider-1", "message": "help"}).json()
        client.post(f"/tickets/{created['id']}/resolve")

        response = client.post(f"/tickets/{created['id']}/resolve")
        assert response.status_code == 409
        app.dependency_overrides.clear()
