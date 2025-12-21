import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AeroSeg | Customer Intelligence",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "dark" not in st.session_state:
    st.session_state.dark = False
if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "df" not in st.session_state:
    st.session_state.df = None
if "result" not in st.session_state:
    st.session_state.result = None

# ---------------- GLOBAL THEME ----------------
def load_css(dark=False):
    if dark:
        bg = "#021117"
        fg = "#B15F99"
        card = "#E9EAEE"
        nav = "linear-gradient(90deg,#0F172A,#020617)"
    else:
        bg = "#1C515B"
        fg = "#BB4580"
        card = "#DBE9ED"
        nav = "linear-gradient(90deg,#2563EB,#06B6D4)"

    st.markdown(f"""
    <style>
    html, body, [data-testid="stApp"] {{
        background-color: {bg};
        color: {fg};
    }}
    .block-container {{ background-color: {bg}; }}
    .nav-bar {{
        background: {nav};
        padding: 14px;
        border-radius: 12px;
        margin-bottom: 20px;
    }}
    .card {{
        background: {card};
        color: #000;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.25);
        margin-bottom: 20px;
    }}
    </style>
    """, unsafe_allow_html=True)

load_css(st.session_state.dark)

# ---------------- NAVIGATION ----------------
with st.container():
    st.markdown('<div class="nav-bar">', unsafe_allow_html=True)
    cols = st.columns([2,1,1,1,1,1])
    cols[0].markdown("### ✈️ **AeroSeg**")
    if cols[1].button("Overview"): st.session_state.page="Overview"
    if cols[2].button("Upload"): st.session_state.page="Upload"
    if cols[3].button("Segmentation"): st.session_state.page="Segmentation"
    if cols[4].button("Recommendations"): st.session_state.page="Recommendations"
    if cols[5].button("🌙 Dark"):
        st.session_state.dark = not st.session_state.dark
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- REQUIRED TEMPLATE ----------------
TEMPLATE_COLUMNS = [
    "LOAD_TIME",
    "FFP_DATE",
    "LAST_TO_END",
    "FLIGHT_COUNT",
    "SUM_YR_1",
    "SUM_YR_2",
    "avg_discount"
]

def download_template():
    df = pd.DataFrame(columns=TEMPLATE_COLUMNS)
    return df.to_csv(index=False).encode("utf-8")

# ---------------- OVERVIEW ----------------
if st.session_state.page == "Overview":
    st.markdown("""
    <div class="card">
    <h1>✈️ AeroSeg – Airline Customer Intelligence</h1>
    <p>Upload airline customer data → Run ML segmentation → Get business insights</p>
    <ul>
      <li>📊 K-Means customer clustering</li>
      <li>🧠 Business-ready recommendations</li>
      <li>⬇️ Downloadable segmented results</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------- UPLOAD ----------------
elif st.session_state.page == "Upload":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📂 Upload Airline Dataset")

    st.download_button(
        "⬇️ Download CSV Template",
        download_template(),
        "aeroseg_template.csv",
        "text/csv"
    )

    file = st.file_uploader("Upload filled template CSV", type="csv")

    if file:
        df = pd.read_csv(file)
        missing = [c for c in TEMPLATE_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"❌ Invalid file. Missing columns: {missing}")
        else:
            st.session_state.df = df
            st.success("✅ Dataset uploaded successfully")
            st.dataframe(df.head())

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SEGMENTATION ----------------
elif st.session_state.page == "Segmentation":
    if st.session_state.df is None:
        st.warning("Upload dataset first.")
    else:
        df = st.session_state.df.copy()

        # ==== CORE LOGIC (UNCHANGED) ====
        df["LOYALTY_DAYS"] = (
            pd.to_datetime(df["LOAD_TIME"]) -
            pd.to_datetime(df["FFP_DATE"])
        ).dt.days

        df["MONETARY"] = df["SUM_YR_1"].fillna(0) + df["SUM_YR_2"].fillna(0)

        features = df[
            ["LOYALTY_DAYS","LAST_TO_END","FLIGHT_COUNT","MONETARY","avg_discount"]
        ]

        imputer = SimpleImputer(strategy="median")
        features = imputer.fit_transform(features)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=5, random_state=42)
        df["Cluster"] = kmeans.fit_predict(features_scaled)

        segment_map = {
            0:"Potential Loyalists",
            1:"Regular Customers",
            2:"Loyal High-Value Customers",
            3:"At-Risk Customers",
            4:"Inactive Customers"
        }

        df["Segment"] = df["Cluster"].map(segment_map)
        st.session_state.result = df
        # ===============================

        st.success("🎯 Segmentation completed")

        dist = df["Segment"].value_counts().reset_index()
        dist.columns = ["Segment","Customers"]

        fig = px.bar(dist, x="Segment", y="Customers", color="Segment")
        st.plotly_chart(fig, use_container_width=True)

        # ---------- CLUSTER PROFILE TABLE ----------
        profile_cols = [
            "LOYALTY_DAYS",
            "LAST_TO_END",
            "FLIGHT_COUNT",
            "MONETARY",
            "avg_discount"
        ]

        cluster_profile = (
            df.groupby("Segment")[profile_cols]
              .mean()
              .round(2)
              .reset_index()
        )

        st.markdown("## 📊 Cluster Profile Summary")

        def highlight_max(s):
            is_max = s == s.max()
            return ["background-color: #fde68a; font-weight: bold" if v else "" for v in is_max]

        styled_profile = (
            cluster_profile
            .set_index("Segment")
            .style
            .apply(highlight_max, axis=0)
        )

        st.dataframe(styled_profile, use_container_width=True)

        # ---------- AUTO-GENERATED INSIGHTS ----------
        st.markdown("## 🧠 Segment Insights")

        segment_explanations = {
            "Loyal High-Value Customers":
                "These customers exhibit the highest loyalty, flight frequency, and monetary value. They are the airline’s most valuable segment and should be prioritized for premium rewards and retention strategies.",

            "Regular Customers":
                "These customers show consistent engagement with moderate loyalty and spending. Targeted upselling and engagement campaigns can increase their lifetime value.",

            "Potential Loyalists":
                "This segment shows promising activity and engagement but has not yet reached peak loyalty. Personalized promotions can help convert them into loyal customers.",

            "At-Risk Customers":
                "Customers in this segment display declining activity despite previous engagement. Proactive win-back campaigns can help reduce churn.",

            "Inactive Customers":
                "These customers have very low recent activity and minimal engagement. Cost-effective reactivation strategies may be applied if profitable."
        }

        for seg in cluster_profile["Segment"]:
            st.markdown(f"""
            <div class="card">
                <h4>{seg}</h4>
                <p>{segment_explanations.get(seg, "No insight available.")}</p>
            </div>
            """, unsafe_allow_html=True)

        # ---------- DOWNLOAD CLUSTER PROFILE ----------
        st.markdown("## 📥 Download Cluster Profile")

        csv_profile = cluster_profile.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Cluster Profile CSV",
            csv_profile,
            "cluster_profile.csv",
            "text/csv"
        )

# ---------------- RECOMMENDATIONS ----------------
elif st.session_state.page == "Recommendations":
    if st.session_state.result is None:
        st.warning("Run segmentation first.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("🧠 Business Recommendations")

        recs = {
            "Loyal High-Value Customers":"🎁 Premium rewards & retention",
            "Regular Customers":"📈 Upsell & engagement",
            "Potential Loyalists":"🎯 Targeted promotions",
            "At-Risk Customers":"🚨 Win-back campaigns",
            "Inactive Customers":"💤 Low-cost reactivation"
        }

        for k,v in recs.items():
            st.info(f"**{k}** → {v}")

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- DOWNLOAD ----------------
if st.session_state.result is not None:
    csv = st.session_state.result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Segmentation Results",
        csv,
        "aeroseg_results.csv",
        "text/csv"
    )