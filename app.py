import pandas as pd
import streamlit as st
from config import CSV_PATH, FEATURES, THRESHOLD, apply_page_style
from database.history import init_history_db, load_history_from_db, save_history
from services.prediction import load_model, predict_transaction
from components.charts import render_analytics
from components.history_panel import render_history_panel, clear_history_button


apply_page_style(st)
init_history_db()
if "history" not in st.session_state:
    st.session_state["history"] = load_history_from_db()

st.title("🛡️ FraudShield")
st.subheader("Intelligent Transaction Fraud Detection")
st.write("Analyze transactions using your trained XGBoost machine-learning model.")

try:
    model = load_model()
    if model.n_features_in_ != len(FEATURES):
        st.error(
            f"Model expects {model.n_features_in_} features, but the dashboard defines {len(FEATURES)}.")
        st.stop()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Detection Model", "XGBoost")
col2.metric("Fraud Threshold", f"{THRESHOLD:.0%}")
col3.metric("Model Features", str(len(FEATURES)))
st.divider()
st.header("Transaction Testing")
method = st.radio("Choose testing method", [
                  "Sample Transactions", "Manual Entry"], horizontal=True)


def load_csv(uploaded_file, csv_path):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    elif csv_path.strip():
        data = pd.read_csv(csv_path.strip())
    else:
        raise ValueError("Please upload a CSV file or enter a valid CSV path.")

    if "Hour" not in data.columns and "Time" in data.columns:
        data["Hour"] = ((data["Time"] // 3600).astype(int) % 24)
    missing = set(FEATURES + ["Class"]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    return data


if method == "Sample Transactions":
    st.info("Test actual rows from your transaction dataset. Upload or select a CSV file available to this app.")
    csv_path = st.text_input("Dataset CSV path", value=str(
        CSV_PATH) if CSV_PATH.exists() else "", key="sample_csv_path")
    uploaded_csv = st.file_uploader("Or upload your transaction dataset (CSV)", type=[
                                    "csv"], key="sample_csv_upload")
    if st.button("Load sample transactions"):
        try:
            st.session_state["sample_data"] = load_csv(uploaded_csv, csv_path)
            st.success(
                f"Loaded {len(st.session_state['sample_data'])} transactions.")
        except Exception as exc:
            st.error(f"Could not load CSV: {exc}")

    if "sample_data" in st.session_state:
        data = st.session_state["sample_data"]
        sample_type = st.selectbox(
            "Choose a sample", ["Legitimate example", "Fraud example"], key="sample_type")
        class_value = 0 if sample_type == "Legitimate example" else 1
        subset = data[data["Class"] == class_value]
        if subset.empty:
            st.warning("No transactions found for this class.")
        else:
            row_index = st.selectbox("Transaction row", subset.index.tolist(
            ), format_func=lambda x: f"Row {x + 1}", key="sample_row")
            if st.button("Analyze transaction"):
                row = subset.loc[row_index]
                result = predict_transaction(
                    model, [float(row[name]) for name in FEATURES])
                record = {"Source": "Sample Transactions",
                          "Actual Class": int(row["Class"]), **result}
                st.session_state["last_result"] = record
                save_history(record)

elif method == "Manual Entry":
    st.write("Load a real example from your CSV or enter values manually.")
    if "manual_values" not in st.session_state:
        st.session_state["manual_values"] = {
            name: (1.0 if name == "Amount" else 0.0) for name in FEATURES}
    for name in FEATURES:
        key = f"manual_input_{name}"
        if key not in st.session_state:
            st.session_state[key] = float(
                st.session_state["manual_values"].get(name, 0.0))

    if CSV_PATH.exists():
        try:
            example_df = pd.read_csv(CSV_PATH)
            if "Hour" not in example_df.columns and "Time" in example_df.columns:
                example_df["Hour"] = (
                    (example_df["Time"] // 3600).astype(int) % 24)
            missing = set(FEATURES + ["Class"]) - set(example_df.columns)
            if missing:
                st.warning(f"CSV is missing columns: {sorted(missing)}")
            else:
                left, right = st.columns(2)
                with left:
                    st.caption("LEGITIMATE EXAMPLE")
                    if st.button("Load Legitimate Example"):
                        rows = example_df[example_df["Class"] == 0]
                        if not rows.empty:
                            row = rows.iloc[0]
                            for name in FEATURES:
                                st.session_state[f"manual_input_{name}"] = float(
                                    row[name])
                            st.session_state["manual_values"] = {
                                name: float(row[name]) for name in FEATURES}
                            st.rerun()
                        else:
                            st.warning("No legitimate examples found.")
                with right:
                    st.caption("FRAUD EXAMPLE")
                    if st.button("Load Fraud Example"):
                        rows = example_df[example_df["Class"] == 1]
                        if not rows.empty:
                            row = rows.iloc[0]
                            for name in FEATURES:
                                st.session_state[f"manual_input_{name}"] = float(
                                    row[name])
                            st.session_state["manual_values"] = {
                                name: float(row[name]) for name in FEATURES}
                            st.rerun()
                        else:
                            st.warning("No fraud examples found.")
        except Exception as exc:
            st.error(f"Could not read sample CSV: {exc}")
    else:
        st.warning(f"Sample CSV not found: {CSV_PATH}")

    with st.form("manual_transaction"):
        values = {}
        col1, col2 = st.columns(2)
        for index, name in enumerate(FEATURES):
            with (col1 if index < 15 else col2):
                values[name] = st.number_input(
                    name, format="%.8f", key=f"manual_input_{name}")
        submitted = st.form_submit_button("Analyze transaction")
    if submitted:
        result = predict_transaction(
            model, [values[name] for name in FEATURES])
        record = {"Source": "Manual Entry",
                  "Actual Class": "Unknown", **result}
        st.session_state["last_result"] = record
        save_history(record)

result = st.session_state.get("last_result")
if result:
    st.divider()
    st.subheader("Analysis Result")
    probability = result["Fraud Probability"]
    left, middle, right = st.columns(3)
    left.metric("Fraud Probability", f"{probability:.2%}")
    middle.metric("Decision Threshold", f"{THRESHOLD:.0%}")
    right.metric("Prediction", result["Prediction"])
    st.progress(min(max(probability, 0.0), 1.0))
    if probability >= THRESHOLD:
        st.error("⚠️ Potential Fraud — flag for further review.")
    else:
        st.success("✅ Likely Legitimate — below the configured threshold.")
    st.caption("This is a model prediction, not a confirmed finding of fraud.")

render_analytics(st.session_state.get("history", []), THRESHOLD)
render_history_panel()
clear_history_button()
