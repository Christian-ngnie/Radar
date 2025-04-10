# -*- coding: utf-8 -*-
"""Gabon Election Threat Intelligence Dashboard"""

import asyncio
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Fix event loop policy before other imports
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
else:
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Configure page
st.set_page_config(
    page_title="Election Threat Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


from pipeline import (
    bertrend_analysis, calculate_trend_momentum,
    visualize_trends, generate_investigative_report,
    categorize_momentum
)

# Session state initialization
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'reports' not in st.session_state:
    st.session_state.reports = {}

# GitHub data configuration (REPLACE WITH YOUR ACTUAL URL)
GITHUB_DATA_URL = "Rd_gb_8.csv"

@st.cache_data
def load_github_data(url):
    """Load dataset from GitHub repository"""
    try:
        if url.endswith('.csv'):
            df = pd.read_csv(url)
        else:
            df = pd.read_excel(url)
            
        # Validate required columns
        required_columns = ['text', 'Timestamp', 'URL', 'Source']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            st.error(f"Missing required columns: {', '.join(missing)}")
            return None
            
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    except Exception as e:
        st.error(f"Error loading data from GitHub: {str(e)}")
        return None

def convert_df(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

def main():
    st.title("🇬🇦 Gabon Election Threat Intelligence Dashboard")
    st.markdown("### Real-time Narrative Monitoring & FIMI Detection")

    # Load data from GitHub
    with st.spinner("Loading election data from GitHub..."):
        df = load_github_data(GITHUB_DATA_URL)
    
    if df is None:
        st.error("""
            Failed to load data. Please check:
            1. GitHub URL is correct and publicly accessible
            2. Data contains required columns: text, Timestamp, URL, Source
            3. File format is CSV or Excel
        """)
        st.stop()

    # Analysis trigger
    if st.button("🚀 Analyze Data", help="Run full BERTrend analysis", type="primary"):
        with st.status("Processing data...", expanded=True) as status:
            try:
                st.write("🔍 Running temporal-semantic clustering...")
                clustered_df = bertrend_analysis(df)

                st.write("📈 Calculating narrative momentum...")
                emerging_trends, momentum_states = calculate_trend_momentum(clustered_df)

                st.write("🎨 Generating visualizations...")
                viz_path = visualize_trends(
                    clustered_df, 
                    momentum_states, 
                    save_path="./data/visualizations"
                )
                
                # Debug information
                st.write(f"🔧 Visualization saved to: {viz_path}")
                st.write(f"📂 Current working directory: {os.getcwd()}")
                st.write(f"📁 Directory contents: {os.listdir(os.path.dirname(viz_path))}")

                status.update(label="Analysis complete!", state="complete", expanded=False)
                
                # Store results in session state
                st.session_state.processed = True
                st.session_state.clustered_df = clustered_df
                st.session_state.momentum_states = momentum_states
                st.session_state.emerging_trends = emerging_trends
                st.session_state.viz_path = viz_path
                
                st.rerun()
                
            except Exception as e:
                status.update(label="Analysis failed!", state="error")
                st.error(f"Critical error during analysis: {str(e)}")
                st.stop()

    # Display results if processed
    if st.session_state.processed:
        clustered_df = st.session_state.clustered_df
        momentum_states = st.session_state.momentum_states
        emerging_trends = st.session_state.emerging_trends
        viz_path = st.session_state.viz_path

        # Create analysis tabs
        tab1, tab2, tab3 = st.tabs([
            "📊 Cluster Analytics",
            "📜 Threat Reports",
            "🚨 Threat Categorization"
        ])

        with tab1:
            col1, col2 = st.columns([2, 1])
            with col1:
                try:
                    if os.path.exists(viz_path):
                        st.image(viz_path, use_container_width=True)
                    else:
                        st.error(f"Visualization file not found at: {viz_path}")
                        st.write("Directory contents:", os.listdir(os.path.dirname(viz_path)))
                except Exception as e:
                    st.error(f"Error displaying visualization: {str(e)}")
                    st.code(f"""
                    Visualization path: {viz_path}
                    File exists: {os.path.exists(viz_path)}
                    Absolute path: {os.path.abspath(viz_path)}
                    Current directory: {os.getcwd()}
                    """)

            with col2:
                st.markdown("### Top Clusters by Momentum")
                momentum_df = pd.DataFrame([
                    {
                        "Cluster": cluster,
                        "Momentum": score,
                        "Sources": len(momentum_states[cluster]['sources']),
                        "Last Active": momentum_states[cluster]['last_update'].strftime('%Y-%m-%d %H:%M')
                    }
                    for cluster, score in emerging_trends
                ])
                
                st.dataframe(
                    momentum_df.sort_values('Momentum', ascending=False),
                    column_config={
                        "Momentum": st.column_config.ProgressColumn(
                            format="%.0f",
                            min_value=0,
                            max_value=momentum_df['Momentum'].max()
                        )
                    },
                    height=400,
                    use_container_width=True
                )

        with tab2:
            cluster_selector = st.selectbox(
                "Select Cluster for Detailed Analysis",
                [cluster for cluster, _ in emerging_trends],
                format_func=lambda x: f"Cluster {x}"
            )

            cluster_score = next((score for cluster, score in emerging_trends if cluster == cluster_selector), 0)
            category = categorize_momentum(cluster_score)
            color_map = {
                'Tier 1: Ambient Noise (Normal baseline activity)': '🟢',
                'Tier 2: Emerging Narrative (Potential story development)': '🟡',
                'Tier 3: Coordinated Activity (Organized group behavior)': '🟠',
                'Tier 4: Viral Emergency (Requires immediate response)': '🔴'
            }
            color = color_map.get(category, '⚪')
            st.markdown(f"**Threat Classification:** {color} `{category}`")

            if cluster_selector not in st.session_state.reports:
                with st.spinner("Generating intelligence report..."):
                    cluster_data = clustered_df[clustered_df['Cluster'] == cluster_selector].copy()
                    cluster_data['momentum_score'] = cluster_score
                    report = generate_investigative_report(
                        cluster_data,
                        momentum_states,
                        cluster_selector
                    )
                    st.session_state.reports[cluster_selector] = report

            report = st.session_state.reports[cluster_selector]

            with st.expander("📄 Full Intelligence Report", expanded=True):
                st.markdown(f"#### Cluster {cluster_selector} Analysis")
                st.markdown(report['report'])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Example Content")
                for i, text in enumerate(report['sample_texts'][:5]):
                    st.markdown(f"**Document {i+1}**")
                    st.info(text[:500] + "..." if len(text) > 500 else text)

            with col2:
                st.markdown("### Associated URLs")
                for url in report['sample_urls']:
                    st.markdown(f"- [{url[:50]}...]({url})")

        with tab3:
            st.markdown("### Threat Tier Classification")
            intel_df = pd.DataFrame([{
                "Cluster": cluster,
                "Momentum": score
            } for cluster, score in emerging_trends])

            st.bar_chart(
                intel_df.set_index('Cluster')['Momentum'],
                color="#FF4B4B",
                height=400,
                use_container_width=True
            )

            st.dataframe(
                intel_df,
                hide_index=True,
                use_container_width=True
            )

        # Download button
        st.sidebar.download_button(
            label="📥 Download Full Report",
            data=convert_df(intel_df),
            file_name=f"threat_report_{datetime.now().date()}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
