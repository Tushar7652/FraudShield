from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fraudshield_xgb_model.json"
DB_PATH = BASE_DIR / "fraudshield_history.db"
CSV_PATH = BASE_DIR / "fraudshield_sample.csv"
THRESHOLD = 0.65
FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Hour"]


def apply_page_style(st):
    st.set_page_config(
        page_title="FraudShield | Fraud Detection",
        page_icon="🛡️",
        layout="wide",
    )
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #080d18 0%, #0d1628 55%, #101d32 100%); color: #e8f0ff; }
    .block-container { padding: 2rem 3rem 3rem 3rem; max-width: 100%; width: 100%; }
    h1, h2, h3 { color: #edf5ff !important; letter-spacing: -0.4px; }
    div[data-testid="stMetric"] { background: linear-gradient(145deg, #121f35, #172842); border: 1px solid #263b59; padding: 20px; border-radius: 14px; box-shadow: 0 4px 18px rgba(0,0,0,.18); }
    div[data-testid="stMetricLabel"] { color: #a9bfdc !important; }
    div[data-testid="stMetricValue"] { color: #54c8ff !important; }
    .stButton > button, .stDownloadButton > button { background: linear-gradient(90deg, #0879e8, #13b8d8); color: white; border: none; border-radius: 9px; padding: .55rem 1rem; font-weight: 600; transition: all .2s ease; }
    .stButton > button:hover, .stDownloadButton > button:hover { border: 1px solid #74e2ff; color: white; filter: brightness(1.12); }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background-color: #121e30; border-color: #304663; border-radius: 8px; }
    div[data-testid="stDataFrame"] { border: 1px solid #263b59; border-radius: 10px; overflow: hidden; }
    div[data-testid="stAlert"] { border-radius: 10px; }
    hr { border-color: #263b59; }
    section[data-testid="stSidebar"] { background: #0a1220; border-right: 1px solid #263b59; }
    @media (max-width: 768px) { .block-container { padding-left: 1rem; padding-right: 1rem; } }
    </style>
    """, unsafe_allow_html=True)
