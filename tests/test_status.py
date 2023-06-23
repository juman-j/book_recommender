from src.main import app
from src.settings import settings
from fastapi.testclient import TestClient

def test_answer():
    client = TestClient(app)
    result = client.get(settings.main_url)
    assert result.status_code == 200