import pytest
import time
import requests
import numpy as np
from unittest.mock import patch
from app.embeddings import get_embeddings_batch, get_embedding

def test_get_embeddings_batch_order_and_concurrency():
    calls = []
    
    def mock_post(url, json=None, timeout=None):
        prompt = json["prompt"]
        calls.append(f"start:{prompt}")
        time.sleep(0.1)
        calls.append(f"end:{prompt}")
        class MockResponse:
            def raise_for_status(self): pass
            def json(self): return {"embedding": [len(prompt)] * 768}
        return MockResponse()
        
    with patch('requests.post', side_effect=mock_post):
        start_time = time.time()
        res = get_embeddings_batch(["A", "BB", "CCC", "DDDD", "EEEEE"])
        duration = time.time() - start_time
        
        # Max execution time should be largely parallel due to max_workers=5
        assert duration < 0.3 # If seq it would be 0.5s
        
        # Verify ordering is robust
        assert res.shape == (5, 768)
        assert res[0][0] == 1
        assert res[1][0] == 2
        
def test_get_embeddings_empty():
    res = get_embeddings_batch([])
    assert res.shape == (0,)

def test_get_embeddings_exception_handled():
    def mock_post_fail(url, json=None, timeout=None):
        if json["prompt"] == "FAIL":
            raise requests.exceptions.ConnectionError("Mocked failure")
        class MockResponse:
            def raise_for_status(self): pass
            def json(self): return {"embedding": [0] * 768}
        return MockResponse()

    with patch('requests.post', side_effect=mock_post_fail):
        with pytest.raises(RuntimeError) as exc:
            get_embeddings_batch(["OK", "FAIL"])
        assert "Failed to fetch embeddings concurrently" in str(exc.value)
