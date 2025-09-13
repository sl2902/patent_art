"""Metrics page"""
import streamlit as st
import sys
import os
from loguru import logger

# Add the parent directory to sys.path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your visualization functions
from src.kaggle.kaggle_patent_chart_metrics import (
    create_latency_visualization_streamlit,
    create_bigquery_visualization, 
    create_discoverability_visualization
)

st.set_page_config(
    page_title="Performance Metrics",
    page_icon="📈",
    layout="wide",
)

st.title("📊 BigQuery AI Performance Metrics")
st.markdown("Comprehensive analysis of semantic search performance, scalability, and optimization strategies.")

# Create tabs for different metrics
tab1, tab2, tab3 = st.tabs([
    "🚀 Latency Analysis", 
    "⚡ Partition Efficiency", 
    "🔍 Search Comparison"
])

with tab1:
    st.header("Query Latency Performance")
    st.markdown("""
    **Methodology:** Performance testing across 10 diverse technology queries, measuring core vector search 
    vs complete pipeline latency (including explainability). Testing validates BigQuery AI performance 
    characteristics between development and cloud environments.
    """)
    
    try:
        create_latency_visualization_streamlit()
        st.markdown("""
        **Key Findings:**
        - Cloud environment shows 33% faster vector search performance
        - Complete pipeline includes query embedding + vector search + explainability computation
        - Both environments successfully process semantic searches across 2.9M patents, with core vector search completing in under 4 seconds; 
          The complete pipeline including explainability features (sentence-level similarity analysis) takes 5.2 seconds in cloud environments
        """)
    except Exception as e:
        st.error(f"Error loading latency visualization: {e}")

with tab2:
    st.header("Partition Pruning Scalability")
    st.markdown("""
    **Methodology:** Cost optimization analysis measuring BigQuery date-based partitioning impact on 
    query performance and data processing costs across different temporal partition strategies.
    Testing was carried out across 10 diverse technology queries avoiding caching
    """)
    
    try:
        fig = create_bigquery_visualization()
        st.plotly_chart(fig, width='content')
        st.markdown("""
        **Key Findings:**
        - 1-month partitioning delivers 84% cost reduction with 12% performance improvement
        - Stable query performance (2.8-3.9 seconds) regardless of data volume
        - Production-ready cost optimization for enterprise deployment
        """)
    except Exception as e:
        st.error(f"Error loading partition efficiency visualization: {e}")

with tab3:
    st.header("Semantic vs Keyword Search Analysis")
    st.markdown("""
    **Methodology:** Comparative analysis across 10 diverse technology queries evaluating discovery 
    capabilities between BigQuery AI semantic search and traditional keyword search approaches. Each
    query returns 30 patents.
    """)
    
    try:
        fig = create_discoverability_visualization()
        st.plotly_chart(fig, width='content')
        st.markdown("""
        **Key Findings:**
        - 98.3% semantic search uniqueness demonstrates complementary approaches
        - Semantic search handles complex technical descriptions where keyword search fails
        - Minimal overlap (5 patents) shows fundamentally different relevance discovery
        """)
    except Exception as e:
        st.error(f"Error loading discovery comparison visualization: {e}")

# Add summary section
st.markdown("---")
st.header("📋 Performance Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Query Response Time",
        value="< 5 seconds",
        help="Consistent performance across 2.9M patents"
    )

with col2:
    st.metric(
        label="Cost Optimization",
        value="84%",
        help="Data processing reduction through partition pruning"
    )

with col3:
    st.metric(
        label="Performance Improvement",
        value="12%",
        help="Performance improvement through partition pruning"
    )

with col4:
    st.metric(
        label="Unique Discovery",
        value="98.3%",
        help="Semantic search results not found by keyword search"
    )