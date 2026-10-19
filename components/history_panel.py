import pandas as pd
import streamlit as st
from database.history import (
    clear_saved_history, load_investigation_review, save_investigation_review,
)
from config import THRESHOLD


def render_history_panel():
    st.divider()
    st.subheader("Transaction Testing History")
    history = st.session_state.get("history", [])
    if not history:
        st.info("No transactions analyzed yet.")
        return

    history_df = pd.DataFrame(history)
    if "Actual Class" in history_df.columns:
        history_df["Actual Class"] = history_df["Actual Class"].map(
            lambda value: str(value) if pd.notna(value) else "Unknown"
        )

    display_df = history_df.copy()
    probabilities = pd.to_numeric(history_df["Fraud Probability"], errors="coerce")
    display_df["Fraud Probability"] = probabilities.map(lambda p: f"{p:.2%}" if pd.notna(p) else "N/A")
    display_df["Risk Level"] = probabilities.apply(
        lambda p: "High Risk" if pd.notna(p) and p >= THRESHOLD
        else "Medium Risk" if pd.notna(p) and p >= THRESHOLD * 0.4
        else "Low Risk" if pd.notna(p) else "Unknown"
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=min(400, 60 + len(display_df) * 38))
    st.download_button("Download History as CSV", history_df.to_csv(index=False).encode("utf-8"), "fraudshield_transaction_history.csv", "text/csv", key="download_history_csv")

    st.divider()
    st.subheader("Fraud Investigation Panel")
    if "Prediction" in history_df:
        mask = history_df["Prediction"].astype(str).str.contains("Fraud", case=False, na=False)
    else:
        mask = probabilities >= THRESHOLD
    suspicious_df = history_df.loc[mask].copy()
    st.metric("Flagged Transactions", len(suspicious_df))
    if suspicious_df.empty:
        st.success("No transactions are currently flagged by the model.")
    else:
        st.warning("Review these transactions before making a final decision.")
        suspicious_display = suspicious_df.copy()
        suspicious_display["Fraud Probability"] = probabilities.loc[suspicious_df.index].map(lambda p: f"{p:.2%}" if pd.notna(p) else "N/A")
        st.dataframe(suspicious_display, use_container_width=True, hide_index=True)

        st.markdown("### Investigate Flagged Transactions")
        # Use the history row index as the stable identifier for this local history.
        for idx, transaction in suspicious_df.iterrows():
            transaction_id = str(idx)
            status_key = f"review_status_{transaction_id}"
            notes_key = f"review_notes_{transaction_id}"
            with st.expander(f"Review Transaction: {transaction_id}"):
                probability = pd.to_numeric(transaction.get("Fraud Probability", 0), errors="coerce")
                probability_percent = probability * 100 if pd.notna(probability) else None
                st.write(f"**Fraud Probability:** {probability_percent:.2f}%" if probability_percent is not None else "**Fraud Probability:** Unavailable")
                st.write(f"**Prediction:** {transaction.get('Prediction', 'Potential Fraud')}")
                saved_status, saved_notes = load_investigation_review(transaction_id)
                if status_key not in st.session_state:
                    st.session_state[status_key] = saved_status
                if notes_key not in st.session_state:
                    st.session_state[notes_key] = saved_notes

                def save_current_review(tid=transaction_id, sk=status_key, nk=notes_key):
                    save_investigation_review(tid, st.session_state[sk], st.session_state[nk])

                st.selectbox("Investigation Status", ["Pending Review", "Under Investigation", "Reviewed - Escalated", "Reviewed - Cleared"], key=status_key, on_change=save_current_review)
                st.text_area("Analyst Notes", key=notes_key, placeholder="Record the reason for your decision...", on_change=save_current_review)

        st.download_button("Download Flagged Transactions CSV", suspicious_df.to_csv(index=False).encode("utf-8"), "fraudshield_flagged_transactions.csv", "text/csv", key="download_flagged_transactions")


def clear_history_button():
    if st.button("Clear testing history"):
        clear_saved_history()
        st.session_state["history"] = []
        st.session_state.pop("last_result", None)
        st.rerun()
