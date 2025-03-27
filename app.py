# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import gc
from pipeline import (
    AnalysisPipeline, 
    categorize_momentum,
    generate_investigative_report
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
        'run_count': 0,
        'current_file_hash': None,
        'clustered_df': None,
        'emerging_trends': None,
        'momentum_states': None,
        'viz_figure': None
    }

if 'session' not in st.session_state:
    st.session_state.session = init_session_state()

@st.cache_data(max_entries=3, ttl=3600, show_spinner="Loading data...")
def load_data(file):
    """Cache data loading with hash validation"""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

@st.cache_resource(show_spinner=False)
def init_pipeline():
    """Cache resource-heavy pipeline initialization"""
    return AnalysisPipeline()

def main():
    st.title("🇬🇦 Gabon Election Threat Intelligence Dashboard")
    st.markdown("### Real-time Narrative Monitoring & FIMI Detection")

    # File upload with hash-based cache invalidation
    uploaded_file = st.file_uploader(
        "Upload Social Media Data (CSV/Excel)",
        type=["csv", "xlsx"],
        help="Max file size: 200MB"
    )
    
    if uploaded_file:
        current_hash = hash(uploaded_file.getvalue())
        
        # Reset session if new file uploaded
        if st.session_state.session['current_file_hash'] != current_hash:
            st.session_state.session = init_session_state()
            st.session_state.session['current_file_hash'] = current_hash
            
        if uploaded_file.size > 200 * 1024 * 1024:
            st.error("File size exceeds 200MB limit")
            return
            
        df = load_data(uploaded_file)
        pipeline = init_pipeline()

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
                
                # Cache maintenance
                if st.session_state.session['run_count'] % 5 == 0:
                    gc.collect()
                    st.cache_data.clear()
                    
                st.session_state.session['run_count'] += 1
                status.update(label="Analysis complete!", state="complete")

    if st.session_state.session['processed']:
        display_results()

def display_results():
    session = st.session_state.session
    clustered_df = session['clustered_df']
    momentum_states = session['momentum_states']
    emerging_trends = session['emerging_trends']

    tab1, tab2, tab3 = st.tabs([
        "📊 Cluster Analytics", 
        "📜 Threat Reports",
        "🚨 Threat Categorization"
    ])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            if session['viz_figure']:
                try:
                    st.pyplot(session['viz_figure'], use_container_width=True)
                except Exception as e:
                    st.error(f"Visualization error: {str(e)}")
            else:
                st.warning("No visualization available")

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
    
    # Get the cluster's momentum score and category
    cluster_score = next((score for cluster, score in emerging_trends if cluster == cluster_selector), 0)
    category = categorize_momentum(cluster_score)
    
    # Display category with color coding
    color_map = {
        'Tier 1: Ambient Noise': '🟢',
        'Tier 2: Emerging Narrative': '🟡',
        'Tier 3: Coordinated Activity': '🟠',
        'Tier 4: Viral Emergency': '🔴'
    }
    st.markdown(f"""
    **Threat Classification:** {color_map.get(category, '⚪')} `{category}`
    """)
    
    # Rest of the report generation remains the same
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
        "Momentum": score
        #"Category": categorize_momentum(score)
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
    main()
