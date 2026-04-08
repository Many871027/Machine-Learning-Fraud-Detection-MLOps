from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import json
import os
import joblib

app = FastAPI(
    title="Credit Card Fraud Prediction API",
    description="Inference API to calculate probability of a transaction being fraudulent.",
    version="1.0.0"
)

from src.data_pipeline import preprocess_new_data

class TransactionFeatures(BaseModel):
    Time: float = 0.0
    V1: float = -1.3598
    V2: float = -0.0727
    V3: float = 2.5363
    V4: float = 1.3781
    V5: float = -0.3383
    V6: float = 0.4623
    V7: float = 0.2395
    V8: float = 0.0986
    V9: float = 0.3637
    V10: float = 0.0907
    V11: float = -0.5515
    V12: float = -0.6178
    V13: float = -0.9913
    V14: float = -0.3111
    V15: float = 1.4681
    V16: float = -0.4704
    V17: float = 0.2079
    V18: float = 0.0257
    V19: float = 0.4039
    V20: float = 0.2514
    V21: float = -0.0183
    V22: float = 0.2778
    V23: float = -0.1104
    V24: float = 0.0669
    V25: float = 0.1285
    V26: float = -0.1891
    V27: float = 0.1335
    V28: float = -0.0210
    Amount: float = 149.62

@app.post("/predict")
def predict_fraud(transaction: TransactionFeatures):
    if not os.path.exists("artifacts/model"):
        raise HTTPException(status_code=500, detail="Model artifact missing. Run training pipeline first (`python -m src.model_pipeline`).")
    if not os.path.exists("artifacts/scaler.pkl"):
        raise HTTPException(status_code=500, detail="Scaler artifact missing.")
        
    try:
        # Preprocess
        input_df = preprocess_new_data(transaction.dict())
        
        # Scale
        scaler = joblib.load("artifacts/scaler.pkl")
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
        
        # Predict 
        model = mlflow.pyfunc.load_model("artifacts/model")
        
        # Attempt to get probability
        if hasattr(model._model_impl, "predict_proba"):
            proba = model._model_impl.predict_proba(input_scaled)[0, 1]
        else:
            proba = model.predict(input_scaled)[0]
            
        fraud_pred = 1 if proba >= 0.5 else 0
        return {"fraud": fraud_pred, "probability": float(proba), "warning": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    """Reads all experimental runs from MLflow tracking server and returns a comparative table."""
    try:
        from mlflow.tracking import MlflowClient
        
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tracking_uri = "file:///" + os.path.join(root_dir, "mlruns").replace("\\", "/")
        mlflow.set_tracking_uri(tracking_uri)
        
        client = MlflowClient(tracking_uri=tracking_uri)
        
        experiment = client.get_experiment_by_name("Fraud_Prediction_Experiments")
        if not experiment:
            return {"message": "No experiments found. Ensure you ran the `run_experiments.py` first!"}
            
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        
        table = []
        for r in runs:
            data = r.data.metrics
            table.append({
                "Model": r.data.params.get("model_type", "Unknown Model"),
                "Accuracy": round(data.get("accuracy", 0), 4),
                "AUC_ROC": round(data.get("auc_roc", 0), 4),
                "F1_Score": round(data.get("f1_score", 0), 4),
                "Precision": round(data.get("precision", 0), 4),
                "Recall": round(data.get("recall", 0), 4)
            })
            
        # Sort by AUC
        table = sorted(table, key=lambda x: x["AUC_ROC"], reverse=True)
        return {"comparative_table": table}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
