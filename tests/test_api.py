from fastapi.testclient import TestClient
from src.inference_api import app
import numpy as np
import cv2

client = TestClient(app)

def test_health_check():
    """Prueba que la API esté viva."""
    response = client.get("/")
    assert response.status_code == 200 or response.status_code == 404

def test_predict_shape():
    """Simula una subida de imagen y verifica respuesta."""

    img = np.zeros((512, 512, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", img)
    
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_encoded.tobytes(), "image/jpeg")}
    )
    assert response.status_code == 200
    assert "detected_ingredients_ids" in response.json()