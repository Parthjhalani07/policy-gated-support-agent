import pytest

from app import audit_log, repository
from app.db import get_connection


@pytest.fixture
def conn():
    connection = get_connection(":memory:")
    repository.init_schema(connection)
    audit_log.init_schema(connection)
    yield connection
    connection.close()
