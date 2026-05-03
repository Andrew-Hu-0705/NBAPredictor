"""
app.py
------
Streamlit UI for the NBA game outcome predictor.

Run with:
    streamlit run app/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from predict import predict_game

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Game Predictor",
    page_icon="🏀",
    layout="centered",
)

# ── Team list ──────────────────────────────────────────────────────────────────
NBA_TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🏀 NBA Game Predictor")
st.caption("XGBoost + SHAP · Trained on current season rolling stats")

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("🏠 Home Team")
    home_team = st.selectbox("Select home team", NBA_TEAMS, index=NBA_TEAMS.index("BOS"))
with col2:
    st.subheader("✈️ Away Team")
    away_team = st.selectbox("Select away team", NBA_TEAMS, index=NBA_TEAMS.index("MIA"))

st.divider()

if home_team == away_team:
    st.warning("Please select two different teams.")
else:
    predict_btn = st.button("Predict Outcome", type="primary", use_container_width=True)

    if predict_btn:
        with st.spinner("Running prediction..."):
            try:
                result = predict_game(home_team, away_team)

                # ── Winner banner ──────────────────────────────────────────────
                winner = result["predicted_winner"]
                conf = result["confidence"]
                home_prob = result["home_win_prob"]
                away_prob = result["away_win_prob"]

                if winner == home_team:
                    st.success(f"### 🏆 Predicted Winner: **{home_team}** (Home)")
                else:
                    st.success(f"### 🏆 Predicted Winner: **{away_team}** (Away)")

                # ── Win probability bar ────────────────────────────────────────
                st.subheader("Win Probability")
                prob_df = pd.DataFrame({
                    "Team": [f"{home_team} (Home)", f"{away_team} (Away)"],
                    "Win Probability": [home_prob, away_prob],
                })
                st.bar_chart(prob_df.set_index("Team"), color="#f7941d")

                # ── Metrics ────────────────────────────────────────────────────
                m1, m2, m3 = st.columns(3)
                m1.metric(f"{home_team} Win Prob", f"{home_prob:.1%}")
                m2.metric(f"{away_team} Win Prob", f"{away_prob:.1%}")
                m3.metric("Model Confidence", f"{conf:.1%}")

                # ── SHAP waterfall ─────────────────────────────────────────────
                st.subheader("Why this prediction? (SHAP)")
                st.caption(
                    "Positive values push toward a **home win**. "
                    "Negative values push toward an **away win**."
                )

                shap_vals = result["shap_values"]
                feat_names = result["feature_cols"]
                feat_vals = result["feature_values"]

                # Top 10 most impactful features
                pairs = sorted(
                    zip(shap_vals, feat_names, feat_vals),
                    key=lambda x: abs(x[0]),
                    reverse=True
                )[:10]

                shap_df = pd.DataFrame(pairs, columns=["SHAP Value", "Feature", "Raw Value"])

                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ["#f7941d" if v > 0 else "#1d6ef7" for v in shap_df["SHAP Value"]]
                ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
                ax.axvline(0, color="gray", linewidth=0.8)
                ax.set_xlabel("SHAP Value (impact on home win probability)")
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # ── Raw feature table ──────────────────────────────────────────
                with st.expander("Show raw feature values"):
                    st.dataframe(
                        shap_df[["Feature", "Raw Value", "SHAP Value"]].style.format(
                            {"Raw Value": "{:.3f}", "SHAP Value": "{:+.4f}"}
                        ),
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info(
                    "Make sure you've run `data/fetch_data.py` and `model/train.py` first, "
                    "and that both teams appear in the dataset."
                )