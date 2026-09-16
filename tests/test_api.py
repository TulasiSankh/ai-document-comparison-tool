import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.compare import ChunkDiff

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("app.main.compare_documents")
def test_compare_text_endpoint(mock_compare):
    mock_compare.return_value = [
        ChunkDiff(status="unchanged", doc_a_text="A", doc_b_text="A", similarity=1.0)
    ]
    
    response = client.post("/compare-text", json={"text_a": "A", "text_b": "A"})
    assert response.status_code == 200
    json_data = response.json()
    assert "diffs" in json_data
    assert len(json_data["diffs"]) == 1
    assert json_data["diffs"][0]["status"] == "unchanged"
    assert json_data["diffs"][0]["doc_a_text"] == "A"

@patch("app.main.compare_documents")
def test_compare_upload_endpoint(mock_compare):
    mock_compare.return_value = []
    
    file_a = ("docA.txt", b"Hello", "text/plain")
    file_b = ("docB.txt", b"World", "text/plain")
    
    response = client.post(
        "/compare",
        files={"file_a": file_a, "file_b": file_b}
    )
    assert response.status_code == 200
    assert "diffs" in response.json()
