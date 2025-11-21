# streamlit_app.py
import streamlit as st
import requests
from pathlib import Path
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Accident Severity - RF vs XGB", layout="wide")
st.title("Accident Severity Predictor — RandomForest vs XGBoost")

# Model selector: Both / RF only / XGB only
model_choice = st.selectbox("Select model to show", ["Both", "RandomForest", "XGBoost"])

# Input form
with st.form("input_form"):
    st.subheader("Accident information")
    start_time = st.text_input("Start_Time (YYYY-MM-DD HH:MM:SS)", "2019-02-08 01:12:00")
    end_time = st.text_input("End_Time (optional)", "2019-02-08 01:35:00")
    temp = st.number_input("Temperature (F)", value=48.5)
    vis = st.number_input("Visibility (mi)", value=10.0)
    wind = st.number_input("Wind Speed (mph)", value=5.0)
    precip = st.number_input("Precipitation (in)", value=0.0)
    lat = st.number_input("Start_Lat", value=37.7749, format="%.6f")
    lng = st.number_input("Start_Lng", value=-122.4194, format="%.6f")
    weather = st.text_input("Weather_Condition", "Clear")
    sunrise = st.selectbox("Sunrise_Sunset", ["Day", "Night"])
    city = st.text_input("City", "San Francisco")
    county = st.text_input("County", "San Francisco")
    state = st.text_input("State", "CA")
    wind_dir = st.text_input("Wind_Direction", "NW")
    submitted = st.form_submit_button("Predict")

# human-readable labels for classes
LABEL_MAP = {1: "Minor", 2: "Moderate", 3: "Serious", 4: "Severe"}

def pretty_probs(probs, label_map=LABEL_MAP):
    """Return a DataFrame with class labels and probabilities for plotting."""
    if probs is None:
        return None
    labels = [label_map.get(i+1, f"Class {i+1}") for i in range(len(probs))]
    return pd.DataFrame({"class": labels, "prob": probs})

def _normalize_api_response(r):
    """
    Accept dict or list-of-dict and return a dict.
    """
    if isinstance(r, dict):
        return r
    if isinstance(r, list):
        if len(r) > 0 and isinstance(r[0], dict):
            return r[0]
        return {}
    return {}

def _extract_predictions_and_probs(out):
    """
    Accept normalized dict 'out' (the API response).
    Return tuple of:
      - rf_pred_list (or None),
      - rf_probs_list (or None),
      - xgb_pred_list (or None),
      - xgb_probs_list (or None)
    Handles both per-model and single-model formats.
    """
    rf_pred = None; rf_probs = None; xgb_pred = None; xgb_probs = None

    # raw blocks could be dict (per-model) or list (single-model)
    raw_preds = out.get("predictions", None)
    raw_probs = out.get("probabilities", None)

    # Case 1: per-model dict
    if isinstance(raw_preds, dict):
        rf_pred = raw_preds.get("random_forest")
        xgb_pred = raw_preds.get("xgboost")
    # Case 2: single-model list (e.g. [2])
    elif isinstance(raw_preds, list):
        # interpret as single-pipeline prediction for the submitted record(s)
        # show this same single prediction for whichever model the UI asked for
        # (or both if model_choice == "Both")
        rf_pred = raw_preds
        xgb_pred = raw_preds

    # Probabilities: similar logic
    if isinstance(raw_probs, dict):
        rf_probs = raw_probs.get("random_forest")
        xgb_probs = raw_probs.get("xgboost")
    elif isinstance(raw_probs, list):
        # e.g. [[0.17,0.41,0.27,0.15]]
        rf_probs = raw_probs
        xgb_probs = raw_probs

    return rf_pred, rf_probs, xgb_pred, xgb_probs

if submitted:
    record = {
        "Start_Time": start_time,
        "End_Time": end_time,
        "Temperature(F)": temp,
        "Visibility(mi)": vis,
        "Wind_Speed(mph)": wind,
        "Precipitation(in)": precip,
        "Start_Lat": lat,
        "Start_Lng": lng,
        "Weather_Condition": weather,
        "Sunrise_Sunset": sunrise,
        "City": city,
        "County": county,
        "State": state,
        "Wind_Direction": wind_dir
    }

    st.write("### Input record")
    st.json(record)

    # Call API
    try:
        resp = requests.post(API_URL, json={"records": [record]}, timeout=10)
        resp.raise_for_status()
        try:
            out_raw = resp.json()
        except Exception as e:
            st.error(f"Failed to parse API response as JSON: {e}")
            st.write("Status code:", resp.status_code)
            st.write("Raw response text:", resp.text)
            out_raw = None
    except Exception as e:
        st.error(f"API request failed: {e}")
        out_raw = None

    if out_raw is None:
        st.warning("No output returned from API.")
    else:
        # Normalize and debug display
        out = _normalize_api_response(out_raw)
        with st.expander("Raw API response (debug)"):
            st.write("type:", type(out_raw))
            st.write(out_raw)

        # Extract predictions and probabilities in a robust way
        rf_pred_list, rf_prob_list, xgb_pred_list, xgb_prob_list = _extract_predictions_and_probs(out)

        # Layout
        cols = st.columns(2) if model_choice == "Both" else st.columns(1)

        # Display RandomForest (if requested)
        if model_choice in ("Both", "RandomForest"):
            with cols[0]:
                st.subheader("RandomForest")
                if rf_pred_list:
                    try:
                        pred = int(rf_pred_list[0])
                        st.success(f"Predicted class: {pred} → {LABEL_MAP.get(pred, 'Class '+str(pred))}")
                    except Exception:
                        st.success(f"Predicted (raw): {rf_pred_list}")
                    # probabilities
                    if rf_prob_list and len(rf_prob_list) > 0:
                        prob_vec = rf_prob_list[0] if isinstance(rf_prob_list[0], (list, tuple)) else rf_prob_list
                        dfp = pretty_probs(prob_vec)
                        st.bar_chart(dfp.set_index("class"))
                else:
                    st.info("RandomForest prediction not available.")

        # Display XGBoost (if requested)
        if model_choice in ("Both", "XGBoost"):
            idx = 1 if model_choice == "Both" else 0
            with cols[idx]:
                st.subheader("XGBoost")
                if xgb_pred_list:
                    try:
                        pred = int(xgb_pred_list[0])
                        st.success(f"Predicted class: {pred} → {LABEL_MAP.get(pred, 'Class '+str(pred))}")
                    except Exception:
                        st.success(f"Predicted (raw): {xgb_pred_list}")
                    if xgb_prob_list and len(xgb_prob_list) > 0:
                        prob_vec = xgb_prob_list[0] if isinstance(xgb_prob_list[0], (list, tuple)) else xgb_prob_list
                        dfp = pretty_probs(prob_vec)
                        st.bar_chart(dfp.set_index("class"))
                else:
                    st.info("XGBoost prediction not available.")
