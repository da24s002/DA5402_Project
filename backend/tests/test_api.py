from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app
from unittest.mock import patch
import json

import base64
import numpy as np

client = TestClient(app)


def test_test_invocations_error():
    response = client.get("/test-invocations-error")
    assert response.status_code == 400
    assert response.json() == {"detail": "Testing invocations error"}

def test_test_save_image_error():
    response = client.get("/test-save-image-error")
    assert response.status_code == 400
    assert response.json() == {"detail": "Testing save image error"}


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "model_prediction_requests_total" in response.text


def test_save_image_success(tmp_path, monkeypatch):
    # Create a dummy image: 28x28 grayscale
    image_array = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    image_bytes = image_array.tobytes()
    image_b64 = base64.b64encode(image_bytes).decode()

    # Patch the image save directory to a temp path to avoid writing to real disk
    monkeypatch.setenv("IMG_DATA_LOCATION", str(tmp_path))

    data = {
        "image": image_b64,
        "category": "test_category"
    }
    response = client.post("/save-image/", json=data)
    assert response.status_code == 200
    assert response.json()["message"] == "Image saved successfully"


def test_save_image_invalid_length():
    # Not 784 bytes
    image_b64 = base64.b64encode(b"short").decode()
    data = {
        "image": image_b64,
        "category": "test_category"
    }
    response = client.post("/save-image/", json=data)
    assert response.status_code == 400
    assert response.json()["detail"] == "Image must be 28x28 pixels (784 values)"

def test_infer_using_mlflow_hosted_model():
    dummy_response = {"prediction": "some_result"}
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = json.dumps(dummy_response)
        data = {"inputs": "dummy_base64_image_string"}
        response = client.post("/invocations/", json=data)
        assert response.status_code == 200
        assert response.json() == dummy_response
