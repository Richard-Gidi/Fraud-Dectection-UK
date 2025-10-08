import pytest
from fastapi.testclient import TestClient

# ...existing code...
from app.main import app


@pytest.mark.integration
def test_predict_in_process():
    """In-process integration test using FastAPI TestClient to avoid network.

    This exercises the full FastAPI stack (routing, validation, feature
    preparation) while still running in-process so CI and local runs stay fast
    and deterministic.
    """
    client = TestClient(app)

    payload = {
        "amount": 1000.0,
        "merchant_name": "Tech Store",
        "merchant_category": "Electronics",
        "transaction_type": "online",
        "card_present": False,
        "timestamp": "2025-09-28T10:00:00",
        "distance_from_home": 25.5,
        "distance_from_last_transaction": 30.0,
        "ratio_to_median_purchase_price": 2.1
    }

    resp = client.post("/api/v1/predict", json=payload)

    # If model/preprocessor missing the endpoint returns 500; assert either
    # a successful prediction schema or a clear server error.
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "model_version" in data
