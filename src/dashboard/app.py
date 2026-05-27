"""
Mining Rehabilitation & Environmental Compliance Verification Project
Phase 7: Production-Grade Frontend Streamlit Dashboard
Author: Senior Geospatial Data Scientist Portfolio Blueprint
"""

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 1. PAGE CONFIGURATION & THEME
st.set_page_config(
    page_title="ESG Mining Rehabilitation Dashboard",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Remote Sensing & ML Mining Rehabilitation Verification")
st.markdown("""
**Case Study:** Ranger Uranium Mine, Northern Territory, Australia  
This system operates as an independent, data-driven ESG compliance auditor. It ingests harmonized Landsat time-series data processed via Google Earth Engine and utilizes an optimized Machine Learning Classifier to track canopy recovery.
""")

# 2. SAFE DATA LOADING PIPELINE
DATA_PATH = os.path.join("data", "processed", "final_dashboard_metrics.csv")

@st.cache_data
def load_dashboard_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    return df

df = load_dashboard_data()

if df is None:
    st.error(f"❌ Missing required production data matrix at `{DATA_PATH}`. Please execute `train_model.py` first.")
else:
    # 3. SIDEBAR CONTROLLER MATRIX
    st.sidebar.header("🕹️ Controller Matrix")
    st.sidebar.markdown("Filter temporal boundaries and tracking metrics:")
    
    # Year Slider
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    year_range = st.sidebar.slider("Select Operational Horizon", min_year, max_year, (min_year, max_year))
    
    # Filter dataset dynamically based on slider selection
    filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    # 4. HIGH-LEVEL KPI EXECUTIVE MATRIX
    st.subheader("📊 Operational Performance Metrics")
    kpi1, kpi2, kpi3 = st.columns(3)
    
    latest_rehab_index = filtered_df['rehab_index'].iloc[-1]
    avg_divergence = filtered_df['ecosystem_divergence'].mean()
    latest_ml_score = filtered_df['rehab_probability_score'].iloc[-1]
    
    with kpi1:
        st.metric(
            label="Current Rehabilitation Index (RI)",
            value=f"{latest_rehab_index:.2f}",
            delta=f"{(latest_rehab_index * 100):.1f}% Target Canopy Reached"
        )
    with kpi2:
        st.metric(
            label="Average Ecosystem Divergence Delta",
            value=f"{avg_divergence:.3f}",
            delta="Lower scores imply recovery trends",
            delta_color="inverse"
        )
    with kpi3:
        st.metric(
            label="ML Rehabilitation Verification Score",
            value=f"{latest_ml_score:.1f}%",
            delta="Confidence matching reference matrix"
        )
        
    st.markdown("---")
    
    # 5. SPLIT GRAPH WINDOW: TREND GRAPH AND FEATURES
    st.subheader("📈 Environmental Integrity Plots")
    col1, col2 = st.columns([2, 1])
    
    col_mapping = {col.lower(): col for col in filtered_df.columns}
    mine_col = col_mapping.get('mine_ndvi', 'mine_NDVI')
    ref_col = col_mapping.get('ref_ndvi', 'ref_NDVI')
    
    with col1:
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(filtered_df['year'], filtered_df[ref_col], color='green', marker='o', linewidth=2, label='Kakadu Reference Matrix')
        ax.plot(filtered_df['year'], filtered_df[mine_col], color='crimson', marker='s', linewidth=2, label='Active Mine Footprint')
        
        ax.set_title("Canopy Greenness (NDVI) Trajectory Over Selected Horizon", fontsize=11, fontweight='bold')
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel("NDVI Value", fontsize=9)
        ax.legend(loc="lower left")
        
        if year_range[0] <= 2021 <= year_range[1]:
            ax.axvspan(2021, min(2026, year_range[1]), color='yellow', alpha=0.15, label='Reclamation Frame')
            
        st.pyplot(fig)
        
    with col2:
        feat_img_path = os.path.join("data", "processed", "ml_feature_importances.png")
        if os.path.exists(feat_img_path):
            st.image(feat_img_path, caption="Live Weights: What drives the ML classifier decisions?", use_container_width=True)
            
    st.markdown("---")
    
    # 6. RAW DATA INSPECTION MATRIX
    st.subheader("🗂️ Audit Data Inspection Matrix")
    with st.expander("Click to expand raw verified engine metrics table"):
        st.dataframe(
            filtered_df[['year', mine_col, ref_col, 'rehab_index', 'ecosystem_divergence', 'rehab_probability_score']], 
            use_container_width=True
        )

st.sidebar.markdown("---")
st.sidebar.info("💡 **Developer Note:** This application demonstrates structural integration from cloud satellite ingestion down to model serving layers.")