import requests
import json
import pytest


def setup_module():
    # This file used to contain integration tests that require a running API server.
    # To keep pytest runs deterministic, skip these here. Run integration checks
    # separately against a live server if needed.
    pytest.skip("Skipping integration test that requires a running API server")


def test_root():
    response = requests.get("http://localhost:8000/")
    print("\nTesting root endpoint:")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")


def test_prediction():
    url = "http://localhost:8000/api/v1/predict"
    data = {
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

    headers = {
        "Content-Type": "application/json"
    }

    try:
        print("\nTesting prediction endpoint:")
        print(f"Sending request to {url}")
        print(f"Request data: {json.dumps(data, indent=2)}")

        response = requests.post(url, json=data, headers=headers)

        print(f"\nStatus code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response body: {response.text}")

        if response.status_code == 200:
            result = response.json()
            print("\nPrediction results:")
            print(f"Is Fraudulent: {result.get('is_fraud')}")
            print(f"Fraud Probability: {result.get('fraud_probability'):.2%}")
            print(f"Model Version: {result.get('model_version')}")
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the API server")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    print("Testing Fraud Detection API...")
    test_root()
    test_prediction()