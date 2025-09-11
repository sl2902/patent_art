import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
import random
from typing import Any, List, Optional
from loguru import logger
import numpy as np
from run_patent_search_pipeline import (
    sanitize_input_query,
    run_semantic_search_pipeline,
    technology_selection,
    generate_random_patents
)

COUNTRY_LIST = ['US', 'KR', 'WO', 'EP', 'RU', 'CN', 'CA', 'JP', 'TW', 'AU']
TECH_AREAS = ['All', 'Human Necessities', 'Operations & Transport', 
                      'Chemistry & Metallurgy', 'Textiles', 'Construction',
                      'Mechanical Engineering', 'Physics', 'Electricity',
                      'Emerging Technologies']
PATENTS_TABLE_NAME = "patents_cpc_flat"

# Configure page
st.set_page_config(
    page_title="🔍 Semantic Sarch",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🔍 Patent Semantic Search & Discovery")
st.markdown("Discover patents through semantic similarity with explainable results")

@st.cache_data
def call_semantic_search_pipeline(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        query_text: Optional[str] = None,
        patent_ids: Optional[List[str]] = None,
        countries: Optional[List[str]] = None,
        top_k: Optional[int] = 1
) -> pd.DataFrame:
    return run_semantic_search_pipeline(
            start_date,
            end_date,
            query_text=query_text,
            patent_ids=patent_ids,
            countries=countries,
            top_k=top_k
        )

@st.cache_data
def generate_random_patent_numbers(
    src_table_name: str,
    cpc_code: str,
    top_k: int = 3
) -> pd.DataFrame:
    return generate_random_patents(
        src_table_name,
        cpc_code,
        top_k
    )
    
def format_best_explanation(explanations_list: List[str]):
    if not explanations_list:
        return "No explanation available"
    
    best = explanations_list[0]  # Highest similarity
    sentence = best['sentence'][:80] + "..." if len(best['sentence']) > 80 else best['sentence']
    
    return f"\"{sentence}\" (similarity: {best['similarity']:.3f})"

# Initialize session state
if 'selected_patents' not in st.session_state:
    st.session_state['selected_patents'] = []

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
            height=100,
            value= st.session_state.get('search_query', ''),
            key="semantic_search_query_box"
        )
        query_patents = None
        
    elif search_method == "Similar Patents":
        current_patents = st.session_state.get('selected_patents', [])
        patent_display = '\n'.join(current_patents)
        patent_numbers_input = st.text_area(
            "Patent numbers (one per line):",
            placeholder="US1234567\nEP9876543\nCN555666",
            value= patent_display,
            key='patent_numbers_input',
            height=100
        )
        if patent_numbers_input:
            manual_patents = [p.strip() for p in patent_numbers_input.split("\n") if p.strip()]
            # Remove duplicates while preserving order
            unique_patents = []
            for patent in manual_patents:
                if patent not in unique_patents:
                    unique_patents.append(patent)
            
            if unique_patents != st.session_state.get('selected_patents', []):
                st.session_state['selected_patents'] = unique_patents
            query_patents = unique_patents
        else:
            query_patents = None
        query_text = None
        
    # else:  # Hybrid
    #     query_text = st.text_input("Text query (optional):")
    #     query_patents = st.text_input("Patent numbers (comma-separated, optional):").split(',') if st.text_input("", key="hybrid_patents") else None
    
    st.divider()
    
    st.subheader("Filters")
    

    available_countries = ["All"] + COUNTRY_LIST
    use_country_filter = st.selectbox("Country filtering:", ["All countries", "Select specific"])

    if use_country_filter == "Select specific":
        selected_countries = st.multiselect("Countries:", COUNTRY_LIST)
    else:
        selected_countries = None
    
    # Date range
    col1, col2 = st.columns(2)
    MIN_DATE = date(2024, 1, 1)
    MAX_DATE = date(2024, 6, 30)

    col1, col2 = st.columns(2)
    with col1:
        start_date = str(st.date_input(
            "From:", 
            value=MIN_DATE,
            min_value=MIN_DATE,
            max_value=MAX_DATE
        ))
    with col2:
        end_date = str(st.date_input(
            "To:", 
            value=MAX_DATE,
            min_value=start_date,
            max_value=MAX_DATE
        ))

    top_k = st.slider("Max results:", 1, 5, 1)

    st.info(f"📅 Patent embeddings available: {MIN_DATE.strftime('%B %Y')} - {MAX_DATE.strftime('%B %Y')}")


    # Sanitize query text before passing it through the semantic search pipeline
    search_valid = True
    validation_error = None
    if query_text:
        logger.info("Sanitize the query text before performing semantic search")
        query_text, is_valid, error_msg = sanitize_input_query(query_text)
        if not is_valid:
            validation_error = error_msg
            search_valid = False

    col1, col2 = st.columns(2)

    with col1:
        search_clicked = st.button(
            "🔍 Search Patents", 
            type="primary", 
            use_container_width=True,
            disabled=not search_valid
        )
    with col2:
        reset_clicked = st.button("🔄 Reset Form", type="secondary", use_container_width='content')
    

    if reset_clicked:
        for key in ['search_query', 'semantic_search_query_box', 'selectbox_key', 'patent_numbers_input', 'selected_patents']:
            if key in st.session_state:
                del st.session_state[key]

        selected_category = "Select Technology"
        st.session_state["selectbox_key"] = selected_category
        st.session_state['selected_patents'] = []
        st.rerun()

if validation_error:
    st.error(validation_error)
    validation_error = None

# Main content area
if search_clicked and (query_text or (query_patents and len(query_patents) > 0)):
    
    # Mock search function call (replace with your actual function)
    with st.spinner("Searching patents and generating explanations..."):
        results_df = call_semantic_search_pipeline(
            start_date,
            end_date,
            query_text,
            query_patents,
            selected_countries,
            top_k
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
    if results_df.empty:
        st.warning("Found no relevant patents. Try one or more of the following suggestions")
        st.markdown("- Check whether patent number is valid or not")
        st.markdown("- Broaden your date range")
        st.markdown("- Remove country filters") 
        st.markdown("- Use different search terms or improve quality of search query")
    else:
        msg = "patents" if len(results_df) > 1 else "patent"
        if results_df["cosine_score"].max() < 0.5:
            st.error(f"Found {len(results_df)} low quality {msg}. Consider improving search query")
        elif results_df["cosine_score"].max() >= 0.5 and results_df["cosine_score"].max() < 0.65:
            st.warning(f"Found {len(results_df)} moderate quality {msg}. Tweak search query for excellent results")
        else:
            st.success(f"Found {len(results_df)} relevant {msg}")
    
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
                        st.metric("Document similarity", f"{row.cosine_score:.3f}")
                    with col3:
                        st.markdown(f"**{row.country_code}** | {row.pub_date}")
                    
                    # Abstract preview
                    st.markdown(f"**Patent:** {row.publication_number}")
                    with st.expander("Read abstract"):
                        st.write(row.abstract_en)
                    
                    # Explanation
                    st.info(f"🧠 **Primary Match:** {format_best_explanation(row.explanation)}")
                    with st.expander("View all explanations"):
                        for i, exp in enumerate(row.explanation, 1):
                            st.write(f"**{i}.** {exp['sentence']}")
                            st.caption(f"Similarity: {exp['similarity']:.3f}")
                            st.divider()
                            
                    
                    st.divider()
        
        with tab2:
            st.subheader("Search Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Similarity score distribution
                fig_hist = px.histogram(
                    results_df, 
                    x='cosine_score', 
                    title="Similarity Score Distribution",
                    labels={'cosine_score': 'Cosine Similarity', 'count': 'Number of Patents'}
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
                'cosine_score': ['mean', 'count']
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
                y='cosine_score', 
                size=[1]*len(results_df),  # Uniform size
                title="Similarity Scores by Country",
                labels={'cosine_score': 'Cosine Similarity', 'country_code': 'Country'}
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
                y='cosine_score',
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
                'cosine_score': ['mean', 'count']
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

    if not results_df.empty:
        st.subheader("Search Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Results", len(results_df))
        with col2:
            st.metric("Avg Similarity", f"{results_df['cosine_score'].mean():.3f}")
        with col3:
            st.metric("Top Score", f"{results_df['cosine_score'].max():.3f}")
        with col4:
            st.metric("Countries", len(results_df['country_code'].unique()))

else:
    # Welcome screen
    st.info("👈 Configure your search in the sidebar and click 'Search Patents' to get started!")

    st.subheader("Examples to experiment with both options")

    col1, col2 = st.columns(2)
    
    with col1:
        # Show some example queries
        st.subheader("Example Queries")
        examples = [
            "solar panel efficiency improvement",
            "machine learning optimization algorithms", 
            "quantum computing error correction",
            "battery energy storage systems",
            "autonomous vehicle navigation"
        ]
        
        # selected_category = "Select Technology"
        # st.session_state["selectbox_key"] = selected_category
        for example in examples:
            if st.button(f"Try: '{example}'", key=f"example_{example}"):
                st.session_state['search_query'] = example
                # st.code(example, language="text")
                st.rerun()
    
    with col2:
        tech_categories = technology_selection()
        st.subheader("Example Patents from BigQuery")
        selected_category = st.selectbox(
            "Explore by technology area:",
            ["Select Technology"] + list(tech_categories.keys()),
            key="selectbox_key",
        )
        
        if selected_category and selected_category != "Select Technology":
            category_info = tech_categories[selected_category]
            st.markdown(f"**{category_info['description']}**")
            if search_method == "Text Query":
                st.markdown("Try these example searches:")
                cols = st.columns(len(category_info['sample_queries']))
                for i, query in enumerate(category_info['sample_queries']):
                    with cols[i]:
                        if st.button(f"'{query}'", key=f"sample_{i}"):
                            st.session_state['search_query'] = query
                            st.rerun()
            elif search_method == "Similar Patents":
                st.markdown("Try these example patents:")
                
                if selected_category and selected_category != "Select Technology":
                    results_key = f'results_{selected_category.replace(" ", "_")}'
                    
                    if results_key not in st.session_state:
                        chosen_cpc = random.choice(category_info["cpc_codes"])
                        try:
                            patents = generate_random_patent_numbers(PATENTS_TABLE_NAME, chosen_cpc)
                            st.session_state[results_key] = patents
                        except Exception as e:
                            st.error(f"Error generating patents: {e}")
                            st.session_state[results_key] = pd.DataFrame()
                    
                    results = st.session_state[results_key]
                    
                    if not results.empty:
                        for i, patent_row in enumerate(results.itertuples(index=False)):
                            patent_num = patent_row.publication_number.strip("'")
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.write(f"**{patent_num}**")
                                if len(patent_row.title_en) > 100:
                                    st.caption(f"{patent_row.title_en[:100]}...")
                                else:
                                    st.caption(f"{patent_row.title_en}")
                            
                            with col2:
                                is_selected = patent_num in st.session_state.get('selected_patents', [])
                    
                                if is_selected:
                                    # Show Remove button for selected patents
                                    remove_key = f"remove_{patent_num}_{i}"
                                    if st.button("Remove", key=remove_key, type="secondary"):
                                        st.session_state['selected_patents'].remove(patent_num)
                                        st.rerun()
                                else:
                                    # Show Add button for unselected patents
                                    add_key = f"add_{patent_num}_{i}"
                                    if st.button("Add", key=add_key, type="primary"):
                                        if patent_num not in st.session_state['selected_patents']:
                                            st.session_state['selected_patents'].append(patent_num)
                                        st.rerun()
                                        
                    if reset_clicked:
                        for key in ['search_query', 'semantic_search_query_box', 'selectbox_key', 'patent_numbers_input', 'selected_patents']:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.session_state['selected_patents'] = []
                        st.rerun()