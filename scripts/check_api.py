from fastapi.testclient import TestClient
from app.main import app


def main():
    client = TestClient(app)

    print("GET / ->")
    r = client.get("/")
    print(r.status_code)
    try:
        print(r.json())
    except Exception:
        print(r.text)

    print("\nPOST /api/v1/predict ->")
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
    r = client.post("/api/v1/predict", json=payload)
    print(r.status_code)
    try:
        print(r.json())
    except Exception:
        print(r.text)


if __name__ == "__main__":
    main()
