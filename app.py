import plotly.express as px
import sqlite3
import json
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

# ---------------- CONFIGURATION ----------------
st.set_page_config(
    page_title="FraudShield | Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)


# ---------------- UI STYLING ----------------
st.markdown("""
<style>
/* Main application background */
.stApp {
    background: linear-gradient(135deg, #080d19 0%, #101b30 100%);
    color: #f1f5f9;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Headings */
h1, h2, h3 {
    color: #f8fafc !important;
    font-weight: 700 !important;
}


/* Metric cards */
[data-testid="stHorizontalBlock"] [data-testid="stMetric"] {
    background: rgba(24, 42, 68, 0.85);
    border: 1px solid #294568;
    padding: 16px;
    border-radius: 14px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.18);
    min-width: 0;
    height: 100%;
    box-sizing: border-box;
}


/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #b8c9df !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-weight: 700;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    border-radius: 9px;
    border: 1px solid #2684d9;
    transition: all 0.2s ease;
    font-weight: 600;
}

/* Button hover */
.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #38bdf8;
    color: #38bdf8;
    transform: translateY(-1px);
}

/* Input fields and dropdowns */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div,
[data-testid="stFileUploader"] {
    border-radius: 8px;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #294568;
    border-radius: 10px;
    overflow: hidden;
}

/* Horizontal separators */
hr {
    border-color: #2b3a50;
}

/* Alerts and status messages */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Sidebar, if used */
[data-testid="stSidebar"] {
    background-color: #0b1322;
}

/* Responsive layout */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------- DASHBOARD THEME ----------------
st.markdown("""
<style>
    /* Main application background */
    .stApp {
        background: linear-gradient(135deg, #080d18 0%, #0d1628 55%, #101d32 100%);
        color: #e8f0ff;
    }

    /* Main content spacing */
.block-container {
    padding: 2rem 3rem 3rem 3rem;
    max-width: 100%;
    width: 100%;
}


    /* Headings */
    h1, h2, h3 {
        color: #edf5ff !important;
        letter-spacing: -0.4px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #121f35, #172842);
        border: 1px solid #263b59;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.18);
    }

    div[data-testid="stMetricLabel"] {
        color: #a9bfdc !important;
    }

    div[data-testid="stMetricValue"] {
        color: #54c8ff !important;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(90deg, #0879e8, #13b8d8);
        color: white;
        border: none;
        border-radius: 9px;
        padding: 0.55rem 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border: 1px solid #74e2ff;
        color: white;
        filter: brightness(1.12);
    }

    /* Input fields and dropdowns */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #121e30;
        border-color: #304663;
        border-radius: 8px;
    }

    /* Data tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #263b59;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Status messages */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Horizontal separators */
    hr {
        border-color: #263b59;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0a1220;
        border-right: 1px solid #263b59;
    }
</style>
""", unsafe_allow_html=True)


MODEL_PATH = Path(__file__).parent / "fraudshield_xgb_model.json"
THRESHOLD = 0.65


FEATURES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount", "Hour"]
)


# -------------- PERMANENT HISTORY DATABASE --------------

DB_PATH = Path(__file__).resolve().parent / "fraudshield_history.db"


def init_history_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transaction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                record_json TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS investigation_reviews (
                transaction_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'Pending Review',
                notes TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)


def load_history_from_db():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT record_json
            FROM transaction_history
            ORDER BY id ASC
        """).fetchall()

    history = []

    for (record_json,) in rows:
        try:
            history.append(json.loads(record_json))
        except (json.JSONDecodeError, TypeError):
            continue

    return history


def save_history(record):
    # Save to SQLite first, then update the current session.
    record_json = json.dumps(
        record,
        default=lambda value: (
            value.item() if hasattr(value, "item") else str(value)
        )
    )

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO transaction_history (record_json) VALUES (?)",
            (record_json,)
        )

    st.session_state.setdefault("history", []).append(record)


def load_investigation_review(transaction_id):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT status, notes
            FROM investigation_reviews
            WHERE transaction_id = ?
            """,
            (str(transaction_id),)
        ).fetchone()

    if row:
        return row[0], row[1]

    return "Pending Review", ""


def save_investigation_review(transaction_id, status, notes):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO investigation_reviews
                (transaction_id, status, notes, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(transaction_id) DO UPDATE SET
                status = excluded.status,
                notes = excluded.notes,
                updated_at = CURRENT_TIMESTAMP
            """,
            (str(transaction_id), status, notes)
        )


def clear_saved_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM transaction_history")


# Create the database and restore previous records.
init_history_db()

if "history" not in st.session_state:
    st.session_state["history"] = load_history_from_db()


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


st.title("🛡️ FraudShield")
st.subheader("Intelligent Transaction Fraud Detection")
st.write(
    "Analyze transactions using your trained XGBoost machine-learning model."
)

try:
    model = load_model()
    if model.n_features_in_ != len(FEATURES):
        st.error(
            f"Model expects {model.n_features_in_} features, "
            f"but the dashboard defines {len(FEATURES)}."
        )
        st.stop()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Detection Model", "XGBoost")
col2.metric("Fraud Threshold", f"{THRESHOLD:.0%}")
col3.metric("Model Features", "31")

st.divider()
st.header("Transaction Testing")

method = st.radio(
    "Choose testing method",
    ["Sample Transactions", "Manual Entry"],
    horizontal=True
)


def predict_transaction(values):
    transaction = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    probability = float(model.predict_proba(transaction)[0, 1])
    is_fraud = probability >= THRESHOLD

    return {
        "Fraud Probability": probability,
        "Prediction": (
            "Potential Fraud" if is_fraud
            else "Likely Legitimate"
        )
    }


# ---------------- TRANSACTION TESTING ----------------


csv_default = Path(__file__).parent / "fraudshield_sample.csv"

if method == "Sample Transactions":

    st.info(
        "Test actual rows from your transaction dataset. "
        "Upload or select a CSV file available to this app."
    )

    csv_path = st.text_input(
        "Dataset CSV path",
        value=str(csv_default) if csv_default.exists() else "",
        key="sample_csv_path"
    )

    uploaded_csv = st.file_uploader(
        "Or upload your transaction dataset (CSV)",
        type=["csv"],
        key="sample_csv_upload"
    )

    if st.button("Load sample transactions"):
        try:
            if uploaded_csv is not None:
                data = pd.read_csv(uploaded_csv)
            elif csv_path.strip():
                data = pd.read_csv(csv_path.strip())
            else:
                st.error("Please upload a CSV file or enter a valid CSV path.")
                st.stop()

            if "Hour" not in data.columns and "Time" in data.columns:
                data["Hour"] = (
                    (data["Time"] // 3600).astype(int) % 24
                )

            required = FEATURES + ["Class"]
            missing = set(required) - set(data.columns)

            if missing:
                st.error(
                    f"Missing CSV columns: {sorted(missing)}"
                )
            else:
                st.session_state["sample_data"] = data
                st.success(f"Loaded {len(data)} transactions.")

        except Exception as e:
            st.error(f"Could not load CSV: {e}")

    if "sample_data" in st.session_state:
        data = st.session_state["sample_data"]

        sample_type = st.selectbox(
            "Choose a sample",
            ["Legitimate example", "Fraud example"],
            key="sample_type"
        )

        class_value = (
            0 if sample_type == "Legitimate example" else 1
        )

        subset = data[data["Class"] == class_value]

        if subset.empty:
            st.warning("No transactions found for this class.")
        else:
            row_index = st.selectbox(
                "Transaction row",
                subset.index.tolist(),
                format_func=lambda x: f"Row {x + 1}",
                key="sample_row"
            )

            if st.button("Analyze transaction"):
                row = subset.loc[row_index]
                values = [float(row[name]) for name in FEATURES]

                result = predict_transaction(values)

                st.session_state["last_result"] = {
                    "Source": "Sample Transactions",
                    "Actual Class": int(row["Class"]),
                    **result
                }

                save_history(st.session_state["last_result"])


# ---------------- MANUAL ENTRY ----------------

elif method == "Manual Entry":

    st.write(
        "Load a real example from your CSV or enter values manually."
    )

    # Initialize all widget values before displaying the widgets.
    if "manual_values" not in st.session_state:
        st.session_state["manual_values"] = {
            name: (1.0 if name == "Amount" else 0.0)
            for name in FEATURES
        }

    for name in FEATURES:
        key = f"manual_input_{name}"

        if key not in st.session_state:
            st.session_state[key] = float(
                st.session_state["manual_values"].get(name, 0.0)
            )

    # Load examples from the CSV.
    try:
        if csv_default.exists():
            example_df = pd.read_csv(csv_default)

            if (
                "Hour" not in example_df.columns
                and "Time" in example_df.columns
            ):
                example_df["Hour"] = (
                    (example_df["Time"] // 3600).astype(int) % 24
                )

            missing = set(FEATURES + ["Class"]) - set(
                example_df.columns
            )

            if missing:
                st.warning(
                    f"CSV is missing columns: {sorted(missing)}"
                )
            else:
                left, right = st.columns(2)

                with left:
                    st.caption("LEGITIMATE EXAMPLE")

                    if st.button("Load Legitimate Example"):
                        rows = example_df[
                            example_df["Class"] == 0
                        ]

                        if not rows.empty:
                            row = rows.iloc[0]

                            for name in FEATURES:
                                st.session_state[
                                    f"manual_input_{name}"
                                ] = float(row[name])

                            st.session_state["manual_values"] = {
                                name: float(row[name])
                                for name in FEATURES
                            }

                            st.rerun()
                        else:
                            st.warning(
                                "No legitimate examples found."
                            )

                with right:
                    st.caption("FRAUD EXAMPLE")

                    if st.button("Load Fraud Example"):
                        rows = example_df[
                            example_df["Class"] == 1
                        ]

                        if not rows.empty:
                            row = rows.iloc[0]

                            for name in FEATURES:
                                st.session_state[
                                    f"manual_input_{name}"
                                ] = float(row[name])

                            st.session_state["manual_values"] = {
                                name: float(row[name])
                                for name in FEATURES
                            }

                            st.rerun()
                        else:
                            st.warning("No fraud examples found.")

        else:
            st.warning(
                f"Sample CSV not found: {csv_default}"
            )

    except Exception as exc:
        st.error(f"Could not read sample CSV: {exc}")

    # Manual inputs: these keys match the example-loading buttons.
    with st.form("manual_transaction"):
        values = {}
        col1, col2 = st.columns(2)

        for index, name in enumerate(FEATURES):
            with (col1 if index < 15 else col2):
                values[name] = st.number_input(
                    name,
                    format="%.8f",
                    key=f"manual_input_{name}"
                )

        submitted = st.form_submit_button(
            "Analyze transaction"
        )

    if submitted:
        result = predict_transaction(
            [values[name] for name in FEATURES]
        )

        st.session_state["last_result"] = {
            "Source": "Manual Entry",
            "Actual Class": "Unknown",
            **result
        }

        save_history(st.session_state["last_result"])


# ---------------- DISPLAY RESULT ----------------
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

    st.caption(
        "This is a model prediction, not a confirmed finding of fraud."
    )


# -------------- DASHBOARD ANALYTICS --------------
st.divider()
st.subheader("FraudShield Analytics Dashboard")

history = st.session_state.get("history", [])

if history:
    analytics_df = pd.DataFrame(history)
    total_transactions = len(analytics_df)

    if "Prediction" in analytics_df.columns:
        predictions = analytics_df["Prediction"].astype(str)

        fraud_count = predictions.str.contains(
            "Fraud", case=False, na=False
        ).sum()

        legitimate_count = predictions.str.contains(
            "Legitimate", case=False, na=False
        ).sum()
    else:
        fraud_count = 0
        legitimate_count = 0

      # Dashboard KPI metrics

    # Dashboard KPI metrics — horizontal layout
    col1, col2, col3 = st.columns(3, gap="medium")

    fraud_rate = (
        (fraud_count / total_transactions) * 100
        if total_transactions > 0 else 0
    )

    legitimate_rate = (
        (legitimate_count / total_transactions) * 100
        if total_transactions > 0 else 0
    )

    with col1:
        st.metric(
            label="Transactions Analyzed",
            value=f"{total_transactions:,}",
            help="Total transactions recorded in the current history."
        )

    with col2:
        st.metric(
            label="Potential Fraud Alerts",
            value=f"{int(fraud_count):,}",
            delta=f"{fraud_rate:.2f}% of analyzed transactions",
            delta_color="inverse",
            help="Transactions classified as potential fraud by the model."
        )

    with col3:
        st.metric(
            label="Likely Legitimate",
            value=f"{int(legitimate_count):,}",
            delta=f"{legitimate_rate:.2f}% of analyzed transactions",
            help="Transactions classified as likely legitimate by the model."
        )

    if "Prediction" in analytics_df.columns:
        st.write("### Prediction Distribution")
        distribution = analytics_df["Prediction"].value_counts()

        if len(distribution) > 0:
            chart_data = distribution.rename_axis(
                "Prediction").reset_index(name="Count")

            st.bar_chart(
                chart_data,
                x="Prediction",
                y="Count",
                height=200,
                use_container_width=True
            )

        else:
            st.info("No predictions available yet.")

    # Fraud vs Legitimate Transactions Chart
    st.write("### Fraud vs. Legitimate Transactions")

    if "Prediction" in analytics_df.columns:
        chart_counts = (
            analytics_df["Prediction"]
            .astype(str)
            .value_counts()
            .rename_axis("Prediction")
            .reset_index(name="Count")
        )

        if not chart_counts.empty:
            fig_donut = px.pie(
                chart_counts,
                names="Prediction",
                values="Count",
                hole=0.55,
                title="Transaction Classification Breakdown",
                color="Prediction",
                color_discrete_map={
                    "Likely Fraud": "#FF4B4B",
                    "Fraud": "#FF4B4B",
                    "Likely Legitimate": "#38BDF8",
                    "Legitimate": "#38BDF8"
                }
            )

            fig_donut.update_traces(
                textposition="inside",
                textinfo="label+percent",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Transactions: %{value}<br>"
                    "Share: %{percent}"
                    "<extra></extra>"
                )
            )

            fig_donut.update_layout(
                template="plotly_dark",
                height=360,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,15,25,0.8)",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(l=20, r=20, t=70, b=20)
            )

            st.plotly_chart(
                fig_donut,
                use_container_width=True,
                key="fraud_legitimate_donut"
            )
        else:
            st.info("No prediction data available yet.")

    if total_transactions >= 2:
        st.write("### Fraud Probability Trend (%)")

        if "Fraud Probability" in analytics_df.columns:
            trend_df = pd.DataFrame({
                "Transaction": range(1, total_transactions + 1),
                "Fraud Probability": pd.to_numeric(
                    analytics_df["Fraud Probability"],
                    errors="coerce"
                )
            }).dropna(subset=["Fraud Probability"])

            if len(trend_df) >= 2:
                fig = px.line(
                    trend_df,
                    x="Transaction",
                    y="Fraud Probability",
                    markers=True,
                    title="Transaction Risk Over Time",
                    labels={
                        "Transaction": "Transaction Number",
                        "Fraud Probability": "Fraud Probability (%)"
                    },
                    template="plotly_dark",
                    range_y=[0, 100]
                )

                fig.update_traces(
                    line=dict(width=3, color="#38BDF8"),
                    marker=dict(
                        size=9,
                        color="#38BDF8",
                        line=dict(width=2, color="#FFFFFF")
                    ),
                    hovertemplate=(
                        "<b>Transaction %{x}</b><br>"
                        "Fraud Probability: %{y:.4f}%"
                        "<extra></extra>"
                    )
                )

                fig.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=60, b=40),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10,15,25,0.8)",
                    xaxis=dict(
                        showgrid=False,
                        dtick=1
                    ),

                    yaxis=dict(
                        title="Fraud Probability (%)",
                        range=[0, 100],
                        ticksuffix="%",
                        tickformat=".1f",
                        showgrid=True,
                        gridcolor="rgba(148, 163, 184, 0.15)"
                    ),

                    showlegend=False
                )

                fig.add_hline(
                    y=50,
                    line_dash="dash",
                    line_color="#FF4B4B",
                    line_width=2,
                    annotation_text="High Risk Threshold (50%)",
                    annotation_position="top right",
                    annotation_font_color="#FF4B4B"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="fraud_probability_trend"
                )
            else:
                st.info("Analyze more transactions to display the trend.")
        else:
            st.info("Fraud probability data is not available.")
    else:
        st.info(
            "Analyze at least two transactions to display the fraud probability trend.")


# ---------------- HISTORY ----------------
st.divider()
st.subheader("Transaction Testing History")

history = st.session_state.get("history", [])

if history:

    history_df = pd.DataFrame(history)

    # Normalize mixed values for Streamlit/PyArrow display
    if "Actual Class" in history_df.columns:
        history_df["Actual Class"] = history_df["Actual Class"].map(
            lambda value: str(value) if pd.notna(value) else "Unknown"
        )

    # Format fraud probability for display
    display_df = history_df.copy()
    display_df["Fraud Probability"] = pd.to_numeric(
        display_df["Fraud Probability"], errors="coerce"
    ).map(lambda p: f"{p:.2%}" if pd.notna(p) else "N/A")

    # Classify transaction risk for display
    probabilities = pd.to_numeric(
        history_df["Fraud Probability"], errors="coerce"
    )

    display_df["Risk Level"] = probabilities.apply(
        lambda p: (
            "High Risk" if pd.notna(p) and p >= THRESHOLD
            else "Medium Risk" if pd.notna(p) and p >= THRESHOLD * 0.4
            else "Low Risk" if pd.notna(p)
            else "Unknown"
        )
    )

    # Professional transaction history table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=min(400, 60 + len(display_df) * 38),
        column_config={
            "Source": st.column_config.TextColumn("Transaction Source"),
            "Actual Class": st.column_config.TextColumn("Actual Class"),
            "Fraud Probability": st.column_config.TextColumn(
                "Fraud Probability",
                help="Estimated probability that the transaction is fraudulent."
            ),

            "Prediction": st.column_config.TextColumn(
                "Model Prediction",
                help="The classification returned by the model."
            ),
            "Risk Level": st.column_config.TextColumn(
                "Risk Level",
                help="Based on the estimated fraud probability."
            ),

        },
    )

    # Export transaction testing history
    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download History as CSV",
        data=csv_data,
        file_name="fraudshield_transaction_history.csv",
        mime="text/csv",
        key="download_history_csv",
    )

    # ---------------- FRAUD INVESTIGATION PANEL ----------------
    st.divider()
    st.subheader("Fraud Investigation Panel")

    probabilities = pd.to_numeric(
        history_df["Fraud Probability"], errors="coerce"
    )

    if "Prediction" in history_df.columns:
        suspicious_mask = (
            history_df["Prediction"]
            .astype(str)
            .str.contains("Fraud", case=False, na=False)
        )
    else:
        suspicious_mask = probabilities >= THRESHOLD

    suspicious_df = history_df.loc[suspicious_mask].copy()

    st.metric("Flagged Transactions", len(suspicious_df))

    if not suspicious_df.empty:
        st.warning(
            "Review these transactions before making a final decision."
        )

        suspicious_display = suspicious_df.copy()
        suspicious_display["Fraud Probability"] = probabilities.loc[
            suspicious_df.index
        ].map(lambda p: f"{p:.2%}" if pd.notna(p) else "N/A")

        st.dataframe(
            suspicious_display,
            width="stretch",
            hide_index=True
        )
    else:
        st.info("No flagged transactions to investigate right now.")

    # Transaction investigation and review controls
    if not suspicious_df.empty:
        st.markdown("### Investigate Flagged Transactions")

        for idx, transaction in suspicious_df.iterrows():
            transaction_id = str(idx)

        with st.expander(f"Review Transaction: {transaction_id}"):
            probability = pd.to_numeric(
                transaction.get("Fraud Probability", 0),
                errors="coerce"
            )

        if pd.notna(probability) and probability <= 1:
            probability_percent = probability * 100
        else:
            probability_percent = probability

        st.write(
            f"**Fraud Probability:** "
            f"{probability_percent:.2f}%"
            if pd.notna(probability_percent)
            else "**Fraud Probability:** Unavailable"
        )

        st.write(
            f"**Prediction:** "
            f"{transaction.get('Prediction', 'Potential Fraud')}"
        )

        status_key = f"review_status_{transaction_id}"
        notes_key = f"review_notes_{transaction_id}"

        saved_status, saved_notes = load_investigation_review(transaction_id)

        if status_key not in st.session_state:
            st.session_state[status_key] = saved_status

        if notes_key not in st.session_state:
            st.session_state[notes_key] = saved_notes

        def save_current_review():
            save_investigation_review(
                transaction_id,
                st.session_state[status_key],
                st.session_state[notes_key]
            )

        st.selectbox(
            "Investigation Status",
            [
                "Pending Review",
                "Under Investigation",
                "Reviewed - Escalated",
                "Reviewed - Cleared",
            ],
            key=status_key,
            on_change=save_current_review,
        )

        st.text_area(
            "Analyst Notes",
            key=notes_key,
            placeholder="Record the reason for your decision...",
            on_change=save_current_review,
        )

        suspicious_csv = suspicious_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Flagged Transactions CSV",
            data=suspicious_csv,
            file_name="fraudshield_flagged_transactions.csv",
            mime="text/csv",
            key="download_flagged_transactions"
        )
    else:
        st.success("No transactions are currently flagged by the model.")


if st.button("Clear testing history"):
    clear_saved_history()
    st.session_state["history"] = []
    st.session_state.pop("last_result", None)
    st.rerun()
else:
    st.info("No transactions analyzed yet.")
