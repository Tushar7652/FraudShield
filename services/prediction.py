import pandas as pd
import streamlit as st
import xgboost as xgb
from config import FEATURES, MODEL_PATH


@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


def predict_transaction(model, values):
    transaction = pd.DataFrame([values], columns=FEATURES)
    probability = float(model.predict_proba(transaction)[0, 1])
    return {
        "Fraud Probability": probability,
        "Prediction": "Potential Fraud" if probability >= 0.65 else "Likely Legitimate",
    }
