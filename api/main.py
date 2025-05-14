from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()
MODEL_PATH = os.getenv("MODEL_PATH", "model/fraud_model.pkl")
model = joblib.load(MODEL_PATH)

class Transaction(BaseModel):
    features: list[float]

# placeholder for the model prediction
@app.post("/predict")
def predict(data: Transaction):
    try:
        X = np.array(data.features).reshape(1, -1)
        prediction = model.predict(X)
        return {"fraud": bool(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))