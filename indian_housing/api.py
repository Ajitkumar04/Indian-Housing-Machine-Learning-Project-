from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
PREPROCESSOR_PATH = ROOT_DIR / "models" / "preprocessor.joblib"
FALLBACK_MODEL_PATH = ROOT_DIR / "models" / "baseline_best.pkl"


class PropertyPredictionRequest(BaseModel):
    state: str
    city: str
    locality: str
    property_type: str
    bhk: int
    size_in_sqft: float
    furnished_status: str
    amenities: str
    facing: str
    owner_type: str
    availability_status: str
    public_transport_accessibility: str
    parking_space: str
    security: str
    floor_no: int
    total_floors: int
    age_of_property: int
    nearby_schools: int
    nearby_hospitals: int


class PredictionResponse(BaseModel):
    predicted_price_in_lakhs: float
    model_used: str


app = FastAPI(
    title="Indian Housing Price Prediction API",
    description="Serve housing price predictions using a saved preprocessing pipeline and model.",
    version="0.1.0",
)


def load_artifacts() -> tuple[object, object, str]:
    if MODEL_PATH.exists():
        model_path = MODEL_PATH
        model_name = "best_model"
    elif FALLBACK_MODEL_PATH.exists():
        model_path = FALLBACK_MODEL_PATH
        model_name = "baseline_best"
    else:
        raise FileNotFoundError(
            "No saved model found. Expected one of: best_model.pkl, baseline_best.pkl"
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError("Saved preprocessor not found at models/preprocessor.joblib")

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(model_path)
    return preprocessor, model, model_name


try:
    PREPROCESSOR, MODEL, MODEL_NAME = load_artifacts()
except Exception as exc:
    PREPROCESSOR = None
    MODEL = None
    MODEL_NAME = "unloaded"
    load_error = str(exc)
else:
    load_error = ""


@app.get("/health")
def health_check() -> dict[str, str]:
    status = "ready" if PREPROCESSOR is not None and MODEL is not None else "error"
    return {
        "status": status,
        "model_name": MODEL_NAME,
        "error": load_error,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PropertyPredictionRequest) -> PredictionResponse:
    if PREPROCESSOR is None or MODEL is None:
        raise HTTPException(status_code=500, detail=f"Model artifacts failed to load: {load_error}")

    try:
        raw_data = pd.DataFrame([payload.dict()])
        prediction_features = PREPROCESSOR.transform(raw_data)
        prediction = MODEL.predict(prediction_features)
        if hasattr(prediction, "tolist"):
            prediction = prediction.tolist()[0]
        return PredictionResponse(predicted_price_in_lakhs=float(prediction), model_used=MODEL_NAME)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
