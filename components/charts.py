import pandas as pd
import plotly.express as px
import streamlit as st


def render_analytics(history, threshold):
    st.divider()
    st.subheader("FraudShield Analytics Dashboard")
    if not history:
        st.info("No transactions analyzed yet.")
        return

    df = pd.DataFrame(history)
    total = len(df)
    predictions = df.get("Prediction", pd.Series(dtype=str)).astype(str)
    fraud_count = int(predictions.str.contains("Fraud", case=False, na=False).sum())
    legitimate_count = int(predictions.str.contains("Legitimate", case=False, na=False).sum())

    col1, col2, col3 = st.columns(3, gap="medium")
    fraud_rate = fraud_count / total * 100 if total else 0
    legitimate_rate = legitimate_count / total * 100 if total else 0
    col1.metric("Transactions Analyzed", f"{total:,}", help="Total transactions recorded in history.")
    col2.metric("Potential Fraud Alerts", f"{fraud_count:,}", delta=f"{fraud_rate:.2f}% of analyzed transactions", delta_color="inverse")
    col3.metric("Likely Legitimate", f"{legitimate_count:,}", delta=f"{legitimate_rate:.2f}% of analyzed transactions")

    if "Prediction" in df:
        st.write("### Prediction Distribution")
        distribution = df["Prediction"].value_counts().rename_axis("Prediction").reset_index(name="Count")
        st.bar_chart(distribution, x="Prediction", y="Count", height=200, use_container_width=True)

        st.write("### Fraud vs. Legitimate Transactions")
        chart_counts = df["Prediction"].astype(str).value_counts().rename_axis("Prediction").reset_index(name="Count")
        if not chart_counts.empty:
            fig = px.pie(
                chart_counts, names="Prediction", values="Count", hole=0.55,
                title="Transaction Classification Breakdown", template="plotly_dark",
                color="Prediction",
                color_discrete_map={
                    "Potential Fraud": "#FF4B4B", "Likely Fraud": "#FF4B4B",
                    "Fraud": "#FF4B4B", "Likely Legitimate": "#38BDF8",
                    "Legitimate": "#38BDF8",
                },
            )
            fig.update_traces(textposition="inside", textinfo="label+percent")
            fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,25,0.8)", margin=dict(l=20, r=20, t=70, b=20))
            st.plotly_chart(fig, use_container_width=True, key="fraud_legitimate_donut")

    if total >= 2 and "Fraud Probability" in df:
        st.write("### Fraud Probability Trend (%)")
        trend = pd.to_numeric(df["Fraud Probability"], errors="coerce").dropna().reset_index(drop=True)
        if len(trend) >= 2:
            trend_df = pd.DataFrame({"Transaction": range(1, len(trend) + 1), "Fraud Probability": trend * 100})
            fig = px.line(trend_df, x="Transaction", y="Fraud Probability", markers=True, title="Transaction Risk Over Time", template="plotly_dark", range_y=[0, 100])
            fig.update_traces(line=dict(width=3, color="#38BDF8"), marker=dict(size=9, color="#38BDF8", line=dict(width=2, color="#FFFFFF")))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=40), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,25,0.8)", yaxis=dict(title="Fraud Probability (%)", range=[0, 100], ticksuffix="%", tickformat=".1f", gridcolor="rgba(148, 163, 184, 0.15)"), showlegend=False)
            fig.add_hline(y=threshold * 100, line_dash="dash", line_color="#FF4B4B", line_width=2, annotation_text=f"Configured Threshold ({threshold:.0%})", annotation_position="top right", annotation_font_color="#FF4B4B")
            st.plotly_chart(fig, use_container_width=True, key="fraud_probability_trend")
        else:
            st.info("Analyze more transactions to display the trend.")
