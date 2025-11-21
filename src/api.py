# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import traceback

# Try relative import for the featurizer used at training time
try:
    # when running as module (uvicorn src.api:app), relative import works
    from .features import DateTimeFeaturizer
except Exception:
    # fallback (if running differently)
    try:
        from src.features import DateTimeFeaturizer
    except Exception:
        DateTimeFeaturizer = None  # we'll handle missing import gracefully

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "pipeline.joblib"

app = FastAPI(title="Accident Severity Predictor", version="1.0")

# ---- Pydantic request model (aliases use original CSV names where needed)
class AccidentRecord(BaseModel):
    Start_Time: Optional[str] = None
    End_Time: Optional[str] = None

    Temperature_F: Optional[float] = Field(None, alias="Temperature(F)")
    Visibility_mi: Optional[float] = Field(None, alias="Visibility(mi)")
    Wind_Speed_mph: Optional[float] = Field(None, alias="Wind_Speed(mph)")
    Precipitation_in: Optional[float] = Field(None, alias="Precipitation(in)")
    Start_Lat: Optional[float] = None
    Start_Lng: Optional[float] = None

    Weather_Condition: Optional[str] = None
    Sunrise_Sunset: Optional[str] = None
    City: Optional[str] = None
    County: Optional[str] = None
    State: Optional[str] = None
    Wind_Direction: Optional[str] = None

    class Config:
        # Pydantic V1 -> V2 compatibility will warn but works; keep alias population enabled
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "Start_Time": "2019-02-08 01:12:00",
                "End_Time": "2019-02-08 01:35:00",
                "Temperature(F)": 48.5,
                "Visibility(mi)": 10.0,
                "Wind_Speed(mph)": 5.0,
                "Precipitation(in)": 0.0,
                "Start_Lat": 37.7749,
                "Start_Lng": -122.4194,
                "Weather_Condition": "Clear",
                "Sunrise_Sunset": "Night",
                "City": "San Francisco",
                "County": "San Francisco",
                "State": "CA",
                "Wind_Direction": "NW"
            }
        }

class PredictRequest(BaseModel):
    records: List[AccidentRecord]

# Globals populated at startup
model = None
REQUIRED_COLS: List[str] = []

# ---- Helper: inspect the loaded pipeline and return expected input columns
def _inspect_required_columns(pipeline) -> List[str]:
    cols = []
    try:
        # If this is an sklearn Pipeline / imblearn Pipeline, find the preprocessing step named "prep"
        prep = None
        if hasattr(pipeline, "named_steps"):
            # try direct 'prep'
            if "prep" in pipeline.named_steps:
                prep = pipeline.named_steps["prep"]
            else:
                # find first ColumnTransformer in pipeline steps
                for step in pipeline.named_steps.values():
                    if hasattr(step, "transformers_"):
                        prep = step
                        break
        else:
            # pipeline might directly be a ColumnTransformer
            if hasattr(pipeline, "transformers_"):
                prep = pipeline

        # If we have a ColumnTransformer-like object, its transformers_ contain (name, transformer, cols)
        if prep is not None and hasattr(prep, "transformers_"):
            for name, transformer, colspec in prep.transformers_:
                # If colspec is a list/tuple/Index/ndarray, extend
                if isinstance(colspec, (list, tuple, np.ndarray, pd.Index)):
                    cols.extend([c for c in list(colspec) if isinstance(c, str)])
                # if colspec is a string (single column), add it
                elif isinstance(colspec, str):
                    cols.append(colspec)
                # else: ignore callables or slices (hard to enumerate here)
        # deduplicate preserving order
        seen = set()
        out = []
        for c in cols:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out
    except Exception:
        # In case of any unexpected structure, return empty list (caller will handle)
        return []


# ---- Load model at startup
@app.on_event("startup")
def load_model():
    global model, REQUIRED_COLS
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Train the model first.")
    model = joblib.load(MODEL_PATH)
    # inspect expected columns (used to pad missing columns at predict time)
    REQUIRED_COLS = _inspect_required_columns(model)


@app.get("/")
def read_root():
    return {"status": "ok", "msg": "Accident Severity Predictor API"}


@app.get("/schema")
def schema():
    """Debug endpoint: returns the list of columns the saved pipeline expects (best-effort)."""
    return {"required_columns": REQUIRED_COLS}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded.")

        # Build DataFrame with original column names (by_alias=True ensures names like "Temperature(F)")
        records = [r.dict(by_alias=True) for r in req.records]
        df = pd.DataFrame(records)

        # 1) Create datetime features if DateTimeFeaturizer is available (same as training)
        if DateTimeFeaturizer is not None:
            dt_fe = DateTimeFeaturizer(start_col="Start_Time", end_col="End_Time")
            df = dt_fe.transform(df)
        else:
            # If featurizer not importable, try to produce minimal time columns safely
            if "Start_Time" in df.columns:
                try:
                    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
                    df["start_hour"] = df["Start_Time"].dt.hour
                    df["start_dayofweek"] = df["Start_Time"].dt.dayofweek
                    df["start_month"] = df["Start_Time"].dt.month
                    df["is_weekend"] = df["Start_Time"].dt.dayofweek.isin([5, 6]).astype(int)
                except Exception:
                    pass
            # duration_min left as NaN if End_Time missing or featurizer unavailable
            if "duration_min" not in df.columns:
                df["duration_min"] = np.nan

        # 2) Ensure required columns exist (add as NaN). REQUIRED_COLS is best-effort from loaded pipeline
        for col in REQUIRED_COLS:
            if col not in df.columns:
                df[col] = np.nan

        # 3) Predict
        preds = model.predict(df)
        probs = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(df).tolist()
            except Exception:
                # some wrappers may not support predict_proba; ignore if fails
                probs = None

        return {"predictions": preds.tolist(), "probabilities": probs}
    except HTTPException:
        raise
    except Exception as e:
        # include traceback in logs but return compact message to client
        tb = traceback.format_exc()
        # log to console for debugging
        print(tb)
        raise HTTPException(status_code=500, detail=str(e))
