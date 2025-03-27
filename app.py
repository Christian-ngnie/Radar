# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from pipeline import (
    bertrend_analysis, calculate_trend_momentum,
    visualize_trends, generate_investigative_report,
    categorize_momentum
)

# Configure page
st.set_page_config(
    page_title="Election Threat Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state management
def init_session_state():
    return {
        'processed': False,
        'reports': {},
        'run_count': 0
    }

if 'session' not in st.session_state:
    st.session_state.session = init_session_state()

# Cache data loading with size limit
@st.cache_data(max_entries=3, ttl=3600, show_spinner="Loading data...")
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

# Cache resource-intensive components
@st.cache_resource
def init_analytics_pipeline():
    from pipeline import AnalysisPipeline
    return AnalysisPipeline()

def main():
    st.title("🇬🇦 Gabon Election Threat Intelligence Dashboard")
    st.markdown("### Real-time Narrative Monitoring & FIMI Detection")

    # File upload with size validation
    uploaded_file = st.file_uploader(
        "Upload Social Media Data (CSV/Excel)",
        type=["csv", "xlsx"],
        help="Max file size: 200MB"
    )
    
    if uploaded_file:
        if uploaded_file.size > 200 * 1024 * 1024:
            st.error("File size exceeds 200MB limit")
            return
            
        df = load_data(uploaded_file)
        pipeline = init_analytics_pipeline()

        if st.button("🚀 Analyze Data"):
            with st.status("Processing data...", expanded=True) as status:
                try:
                    results = pipeline.process(df)
                    st.session_state.session.update({
                        'processed': True,
                        'clustered_df': results['clustered_df'],
                        'emerging_trends': results['emerging_trends'],
                        'momentum_states': results['momentum_states'],
                        'viz_figure': results['viz_figure']
                    })
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
                    return
                
                # Periodic cleanup
                if st.session_state.session['run_count'] % 5 == 0:
                    gc.collect()
                    st.cache_data.clear()
                    
                st.session_state.session['run_count'] += 1
                status.update(label="Analysis complete!", state="complete")

    if st.session_state.session['processed']:
        display_results()

def display_results():
    clustered_df = st.session_state.session['clustered_df']
    momentum_states = st.session_state.session['momentum_states']
    emerging_trends = st.session_state.session['emerging_trends']
    viz_figure = st.session_state.session['viz_figure']

    tab1, tab2, tab3 = st.tabs([
        "📊 Cluster Analytics", 
        "📜 Threat Reports",
        "🚨 Threat Categorization"
    ])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            try:
                st.pyplot(viz_figure, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying visualization: {str(e)}")

        with col2:
            display_momentum_table(emerging_trends, momentum_states)

    with tab2:
        display_threat_reports(clustered_df, emerging_trends, momentum_states)

    with tab3:
        display_threat_categorization(emerging_trends)

def display_momentum_table(emerging_trends, momentum_states):
    st.markdown("### Top Clusters by Momentum")
    momentum_df = pd.DataFrame([
        {
            "Cluster": cluster,
            "Momentum": score,
            "Sources": len(momentum_states[cluster]['sources']),
            "Last Active": momentum_states[cluster]['last_update'].strftime('%Y-%m-%d %H:%M')
        } for cluster, score in emerging_trends
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
        height=400
    )

def display_threat_reports(clustered_df, emerging_trends, momentum_states):
    cluster_selector = st.selectbox(
        "Select Cluster for Detailed Analysis",
        [cluster for cluster, _ in emerging_trends],
        format_func=lambda x: f"Cluster {x}"
    )
    
    # Report generation with cache
    if cluster_selector not in st.session_state.session['reports']:
        with st.spinner("Generating intelligence report..."):
            report = generate_investigative_report(
                clustered_df[clustered_df['Cluster'] == cluster_selector],
                momentum_states,
                cluster_selector
            )
            st.session_state.session['reports'][cluster_selector] = report

    report = st.session_state.session['reports'][cluster_selector]
    display_report_content(report)

def display_report_content(report):
    with st.expander("📄 Full Intelligence Report", expanded=True):
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

def display_threat_categorization(emerging_trends):
    st.markdown("### Threat Tier Classification")
    intel_df = pd.DataFrame([{
        "Cluster": cluster,
        "Momentum": score,
        "Category": categorize_momentum(score)
    } for cluster, score in emerging_trends])

    st.bar_chart(
        intel_df.set_index('Cluster')['Momentum'],
        color="#FF4B4B",
        height=400
    )

    st.dataframe(
        intel_df,
        column_config={
            "Category": st.column_config.SelectboxColumn(
                help="Threat classification tiers"
            )
        },
        hide_index=True
    )

if __name__ == "__main__":
    import gc
    gc.enable()
    main()
