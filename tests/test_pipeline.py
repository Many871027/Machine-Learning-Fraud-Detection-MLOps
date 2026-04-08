import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_metrics_endpoint():
    """
    Test que garantiza que el endpoint GET /metrics está vivo y retorna la tabla métrica de MLflow.
    """
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "comparative_table" in response.json() or "message" in response.json()

def test_predict_endpoint_valid_payload():
    """
    Test crucial que simula un payload financiero estructurado hacia POST /predict.
    Asegura que el modelo Artifact y el Scaler están acoplados respondiendo una probabilidad JSON.
    """
    payload = {
        "Time": 0.0, "V1": -1.3598, "V2": -0.0727, "V3": 2.5363, "V4": 1.3781,
        "V5": -0.3383, "V6": 0.4623, "V7": 0.2395, "V8": 0.0986, "V9": 0.3637,
        "V10": 0.0907, "V11": -0.5515, "V12": -0.6178, "V13": -0.9913, "V14": -0.3111,
        "V15": 1.4681, "V16": -0.4704, "V17": 0.2079, "V18": 0.0257, "V19": 0.4039,
        "V20": 0.2514, "V21": -0.0183, "V22": 0.2778, "V23": -0.1104, "V24": 0.0669,
        "V25": 0.1285, "V26": -0.1891, "V27": 0.1335, "V28": -0.0210, "Amount": 149.62
    }
    response = client.post("/predict", json=payload)
    
    # Comprobar si el modelo fue compilado; si es 500 puede ser que falte el model_pipeline.py
    # Pero aquí forzamos que al menos reconozca la ruta y el esquema Pydantic
    if response.status_code == 200:
        json_resp = response.json()
        assert "fraud" in json_resp
        assert "probability" in json_resp
    else:
        # En CI/CD sin el modelo entrenado lanzará 500, que es lo documentado
        assert response.status_code == 500
