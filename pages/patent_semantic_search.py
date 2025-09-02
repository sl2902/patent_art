import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
import numpy as np
from run_patent_search_pipeline import run_semantic_search_pipeline

# Configure page
st.set_page_config(page_title="Patent Semantic Search", layout="wide")

# Title and description
st.title("🔍 Patent Semantic Search & Discovery")
st.markdown("Discover patents through semantic similarity with explainable results")

# Sidebar for search options
with st.sidebar:
    st.header("Search Configuration")
    
    # Search method
    search_method = st.radio(
        "Search method:",
        ["Text Query", "Similar Patents"]
    )
    
    # Input based on method
    if search_method == "Text Query":
        query_text = st.text_area(
            "Describe the technology:",
            placeholder="e.g., solar panel efficiency improvement, machine learning algorithms",
            height=100
        )
        query_patents = None
        
    elif search_method == "Similar Patents":
        patent_numbers_input = st.text_area(
            "Patent numbers (one per line):",
            placeholder="US1234567\nEP9876543\nCN555666",
            height=100
        )
        query_patents = patent_numbers_input.split('\n') if patent_numbers_input and "\n" in patent_numbers_input else None
        query_text = None
        
    # else:  # Hybrid
    #     query_text = st.text_input("Text query (optional):")
    #     query_patents = st.text_input("Patent numbers (comma-separated, optional):").split(',') if st.text_input("", key="hybrid_patents") else None
    
    st.divider()
    
    st.subheader("Filters")
    
    available_countries = ["US", "CN", "JP", "DE", "KR", "GB", "FR", "TW", "EP", "WO"]
    selected_countries = st.multiselect("Countries:", available_countries)
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = str(st.date_input("From:", value=date(2024, 1, 1)))
    with col2:
        end_date = str(st.date_input("To:", value=date(2024, 6, 30)))

    top_k = st.slider("Max results:", 1, 5, 1)
    
    # Search button
    search_clicked = st.button("🔍 Search Patents", type="primary", width='content')

# Main content area
if search_clicked and (query_text or query_patents):
    
    # Mock search function call (replace with your actual function)
    with st.spinner("Searching patents and generating explanations..."):
        results_df = run_semantic_search_pipeline(
            start_date,
            end_date,
            query_text=query_text,
            patent_ids=query_patents
        )
        
        # Mock data for visualization demo
        # results_df = pd.DataFrame({
        #     'publication_number': ['US11234567', 'CN987654321', 'JP2024123456', 'DE202412345'],
        #     'country_code': ['US', 'CN', 'JP', 'DE'],
        #     'title_en': [
        #         'High-Efficiency Solar Panel with Quantum Dot Enhancement',
        #         'Machine Learning Optimization for Photovoltaic Systems', 
        #         'Advanced Silicon Wafer Processing for Solar Applications',
        #         'AI-Driven Solar Panel Defect Detection System'
        #     ],
        #     'abstract_en': [
        #         'This invention relates to solar panels with improved efficiency using quantum dots for light absorption enhancement...',
        #         'A system for optimizing solar panel performance using machine learning algorithms to predict optimal positioning...',
        #         'Novel silicon processing techniques that improve solar cell efficiency through advanced wafer treatment methods...',
        #         'Automated defect detection in solar manufacturing using computer vision and neural network classification...'
        #     ],
        #     'pub_date': ['2024-03-15', '2024-02-28', '2024-04-10', '2024-01-20'],
        #     'cosine_score': [0.892, 0.847, 0.823, 0.798],
        #     'explanation': [
        #         'Most relevant: "improved efficiency using quantum dots for light absorption" (similarity: 0.891)',
        #         'Most relevant: "optimizing solar panel performance using machine learning" (similarity: 0.845)',
        #         'Most relevant: "improve solar cell efficiency through advanced wafer treatment" (similarity: 0.821)',
        #         'Most relevant: "solar manufacturing using computer vision and neural network" (similarity: 0.797)'
        #     ]
        # })
    
    # Display results
    st.success(f"Found {len(results_df)} relevant patents")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Results", "📊 Analytics", "🗺️ Geographic", "🕒 Timeline"])
    
    with tab1:
        st.subheader("Search Results with Explanations")
        
        for row in results_df.itertuples(index=False):
            with st.container():
                # Header with score
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{row.title_en}**")
                with col2:
                    st.metric("Similarity", f"{row.similarity_score:.3f}")
                with col3:
                    st.markdown(f"**{row.country_code}** | {row.pub_date}")
                
                # Abstract preview
                st.markdown(f"**Patent:** {row.publication_number}")
                with st.expander("Read abstract"):
                    st.write(row.abstract_en)
                
                # Explanation
                st.info(f"🧠 **Why this matched:** {row.explanation}")
                
                st.divider()
    
    with tab2:
        st.subheader("Search Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Similarity score distribution
            fig_hist = px.histogram(
                results_df, 
                x='similarity_score', 
                title="Similarity Score Distribution",
                labels={'similarity_score': 'Cosine Similarity', 'count': 'Number of Patents'}
            )
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Country breakdown
            country_counts = results_df['country_code'].value_counts()
            fig_pie = px.pie(
                values=country_counts.values, 
                names=country_counts.index,
                title="Results by Country"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab3:
        st.subheader("Geographic Distribution")
        
        # Country mapping for visualization
        country_data = results_df.groupby('country_code').agg({
            'similarity_score': ['mean', 'count']
        }).round(3)
        country_data.columns = ['avg_similarity', 'patent_count']
        country_data = country_data.reset_index()
        
        st.dataframe(
            country_data,
            column_config={
                'country_code': 'Country',
                'avg_similarity': st.column_config.NumberColumn('Avg Similarity', format="%.3f"),
                'patent_count': st.column_config.NumberColumn('Patent Count', format="%d")
            }
        )
        
        # Scatter plot: country vs similarity
        fig_scatter = px.scatter(
            results_df,
            x='country_code',
            y='similarity_score', 
            size=[1]*len(results_df),  # Uniform size
            title="Similarity Scores by Country",
            labels={'similarity_score': 'Cosine Similarity', 'country_code': 'Country'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab4:
        st.subheader("Timeline View")
        
        # Convert dates for plotting
        results_df['pub_date'] = pd.to_datetime(results_df['pub_date'])
        
        # Timeline scatter plot
        fig_timeline = px.scatter(
            results_df,
            x='pub_date',
            y='similarity_score',
            color='country_code',
            size=[20]*len(results_df),  # Uniform size
            hover_data=['title_en', 'publication_number'],
            title="Patent Similarity Over Time",
            labels={'pub_date': 'Publication Date', 'cosine_score': 'Cosine Similarity'}
        )
        fig_timeline.update_layout(height=500)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Monthly aggregation
        results_df['month'] = results_df['pub_date'].dt.to_period('M')
        monthly_stats = results_df.groupby('month').agg({
            'similarity_score': ['mean', 'count']
        }).round(3)
        monthly_stats.columns = ['avg_similarity', 'count']
        monthly_stats = monthly_stats.reset_index()
        monthly_stats['month'] = monthly_stats['month'].astype(str)
        
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Scatter(
            x=monthly_stats['month'],
            y=monthly_stats['avg_similarity'],
            mode='lines+markers',
            name='Avg Similarity',
            line=dict(color='blue')
        ))
        
        fig_monthly.add_trace(go.Scatter(
            x=monthly_stats['month'],
            y=monthly_stats['count']/max(monthly_stats['count']),  # Normalize for dual axis
            mode='lines+markers',
            name='Patent Count (normalized)',
            line=dict(color='red', dash='dash'),
            yaxis='y2'
        ))
        
        fig_monthly.update_layout(
            title="Monthly Trends",
            xaxis_title="Month",
            yaxis=dict(title="Average Similarity", side="left"),
            yaxis2=dict(title="Patent Count", side="right", overlaying="y"),
            height=400
        )
        
        st.plotly_chart(fig_monthly, use_container_width=True)

# Summary metrics at the bottom
if search_clicked and (query_text or query_patents):
    st.subheader("Search Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Results", len(results_df))
    with col2:
        st.metric("Avg Similarity", f"{results_df['similarity_score'].mean():.3f}")
    with col3:
        st.metric("Top Score", f"{results_df['similarity_score'].max():.3f}")
    with col4:
        st.metric("Countries", len(results_df['country_code'].unique()))

else:
    # Welcome screen
    st.info("👈 Configure your search in the sidebar and click 'Search Patents' to get started!")
    
    # Show some example queries
    st.subheader("Example Queries")
    examples = [
        "solar panel efficiency improvement",
        "machine learning optimization algorithms", 
        "quantum computing error correction",
        "battery energy storage systems",
        "autonomous vehicle navigation"
    ]
    
    for example in examples:
        if st.button(f"Try: '{example}'", key=f"example_{example}"):
            st.session_state.example_query = example
            st.rerun()