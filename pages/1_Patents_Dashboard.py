"""Streamlit Patent Art Dashboard"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from typing import Any, List, Optional
import numpy as np

from generate_patent_analysis import (
    dataset_size_table, country_wise_breakdown, top_country_each_month,
    yoy_lang_growth_rate, yoy_country_growth_rate, citations_top_countries,
    top_cpc, tech_area_cpc, tech_convergence, patent_flow
)

st.set_page_config(
    page_title="Patents Dashboard",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

def create_metric_cards(df, metrics_config):
    """Create styled metric cards from dataframe"""
    cols = st.columns(len(metrics_config))
    
    for i, (col_name, config) in enumerate(metrics_config.items()):
        with cols[i]:
            if col_name in df.columns:
                value = df[col_name].iloc[0]
                if config.get('format') == 'comma':
                    formatted_value = f"{value:,.0f}"
                elif config.get('format') == 'percentage':
                    formatted_value = f"{value:.1f}%"
                elif config.get('format') == 'decimal':
                    formatted_value = f"{value:.1f}"
                else:
                    formatted_value = str(value)
                
                st.metric(
                    label=config['label'],
                    value=formatted_value,
                    delta=config.get('delta')
                )

def display_metrics_html_streamlit(summary_df: pd.DataFrame):
    """
    Display summary statistics as styled HTML cards in Streamlit.
    """
    if summary_df.empty:
        return None

    row = summary_df.iloc[0]

    # Use Streamlit columns instead of CSS grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 16px; box-shadow: 0 4px 8px rgba(0,0,0,0.08); border: 1px solid #eee; margin-bottom: 16px;">
            <h4 style="color: #333; border-bottom: 2px solid #f0f0f0; padding-bottom: 6px;">Patent Portfolio Overview</h4>
            <div style="display: flex; justify-content: space-between; margin: 6px 0;">
                <span style="color: #666;">Total Patents</span>
                <strong style="color: #2c3e50;">{:,}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 6px 0;">
                <span style="color: #666;">Countries</span>
                <strong style="color: #2c3e50;">{:,}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 6px 0;">
                <span style="color: #666;">Patent Families</span>
                <strong style="color: #2c3e50;">{:,}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 6px 0;">
                <span style="color: #666;">Avg Title Length</span>
                <strong style="color: #2c3e50;">{:.1f}</strong>
            </div>
        </div>
        """.format(
            int(row['total_patents']),
            int(row['unique_countries']), 
            int(row['unique_families']),
            row['avg_title_length']
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 16px; box-shadow: 0 4px 8px rgba(0,0,0,0.08); border: 1px solid #eee; margin-bottom: 16px;">
            <h4 style="color: #333; border-bottom: 2px solid #f0f0f0; padding-bottom: 6px;">Data Quality Metrics</h4>
            <div style="display: flex; justify-content: space-between; margin: 6px 0;">
                <span style="color: #666;">Title Completeness</span>
                <strong style="color: #2c3e50;">{:.1f}%</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 6px 0;">
                <span style="color: #666;">Abstract Completeness</span>
                <strong style="color: #2c3e50;">{:.1f}%</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 6px 0;">
                <span style="color: #666;">Claims Completeness</span>
                <strong style="color: #2c3e50;">{:.1f}%</strong>
            </div>
        </div>
        """.format(
            row['title_completeness_pct'],
            row['abstract_completeness_pct'],
            row['claims_completeness_pct']
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 16px; box-shadow: 0 4px 8px rgba(0,0,0,0.08); border: 1px solid #eee; margin-bottom: 16px;">
            <h4 style="color: #333; border-bottom: 2px solid #f0f0f0; padding-bottom: 6px;">Content Analysis</h4>
            <div style="display: flex; justify-content: space-between; margin: 6px 0;">
                <span style="color: #666;">Avg Abstract Length</span>
                <strong style="color: #2c3e50;">{:.1f}</strong>
            </div>
        </div>
        """.format(row['avg_abstract_length']), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 16px; box-shadow: 0 4px 8px rgba(0,0,0,0.08); border: 1px solid #eee; margin-bottom: 16px;">
            <h4 style="color: #333; border-bottom: 2px solid #f0f0f0; padding-bottom: 6px;">Corporate Patent Classification (CPC)</h4>
            <div style="display: flex; justify-content: space-between; margin: 6px 0;">
                <span style="color: #666;">Total Patents with CPC</span>
                <strong style="color: #2c3e50;">{:,}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 6px 0;">
                <span style="color: #666;">Coverage</span>
                <strong style="color: #2c3e50;">{:.1f}%</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 6px 0;">
                <span style="color: #666;">Average Codes Per Patent</span>
                <strong style="color: #2c3e50;">{:.1f}</strong>
            </div>
        </div>
        """.format(
            int(row['patents_with_codes']),
            row['coverage_pct'],
            row['avg_codes_per_patent']
        ), unsafe_allow_html=True)

@st.cache_data
def load_summary_data():
    """Load actual summary statistics from BigQuery"""
    try:
        return dataset_size_table()
    except Exception as e:
        st.error(f"Error loading summary data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_country_data(top_n=10):
    """Load actual country-wise statistics"""
    try:
        return country_wise_breakdown(top_n)
    except Exception as e:
        st.error(f"Error loading country data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_citation_data(top_n=10):
    """Load citation patterns for top countries"""
    try:
        return citations_top_countries(top_n)
    except Exception as e:
        st.error(f"Error loading citation data: {e}")
        return pd.DataFrame()

@st.cache_data  
def load_cpc_data(top_n=5):
    """Load CPC classification data"""
    try:
        return top_cpc(top_n)
    except Exception as e:
        st.error(f"Error loading CPC data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_tech_area_data(top_n=10):
    """Load technology area analysis data"""
    try:
        return tech_area_cpc(top_n)
    except Exception as e:
        st.error(f"Error loading tech area data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_tech_convergence_data(top_n=5):
    """Load technology convergence anlaysis data"""
    try:
        return tech_convergence(top_n)
    except Exception as e:
        st.error(f"Error loading tech convergence data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_time_series_data():
    """Load YoY trend data"""
    try:
        return yoy_lang_growth_rate()
    except Exception as e:
        st.error(f"Error loading time series data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_country_trends_data(top_n=10, filter_clause: Optional[str] = None):
    """Load country-wise YoY trends"""
    try:
        return yoy_country_growth_rate(top_n, filter_clause=filter_clause)
    except Exception as e:
        st.error(f"Error loading country trends: {e}")
        return pd.DataFrame()

@st.cache_data
def load_patent_flow_data():
    """Load patent flow data for Sankey"""
    try:
        return patent_flow()
    except Exception as e:
        st.error(f"Error loading patent flow data: {e}")
        return pd.DataFrame()

def create_country_bar_chart(df):
    """Create country-wise distribution of publications"""
    if df.empty:
        st.error("No data available for country-wise bar chart")
        return None
    
    topn_countries_publication = df.sort_values('percentage', ascending=False)
    
    fig = go.Figure()

    fig = px.bar(
    topn_countries_publication,
    x="percentage",
    y="country_code",
    text="percentage",
    # title="Top 10 Countries by Publication Percentage",
    labels={"country_code": "Country", "percentage": "Publication %"},
    color_discrete_sequence=["#1f77b4"],
    orientation="h"
)

    fig.update_traces(
        texttemplate='%{text:.2f}%', 
        textposition='outside'
    )

    fig.update_layout(
        # Remove background and grid
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
        
        # Clean up axes
        yaxis=dict(
            tickangle=0,
            showgrid=False,  # Remove vertical grid lines
            zeroline=False,  # Remove zero line
            title="Country",
            autorange='reversed',
        ),
        xaxis=dict(
            title="Publication Percentage",
            showgrid=False,  # Remove horizontal grid lines
            zeroline=False   # Remove zero line
        ),
        
        # Remove legend and set dimensions
        showlegend=False,
        height=500,
        width=1100,
        
        title=dict(
            text="Top 10 Countries by Publication Percentage (2017 - 2025)",
            x=0.5,  # Center title
            font=dict(size=16)
        ),
        
        # Remove margins for cleaner look
        margin=dict(l=20, r=20, t=60, b=60)
    )

    # Make bars slightly transparent for modern look
    fig.update_traces(
        marker=dict(
            color='#1f77b4',
            opacity=0.8,
            line=dict(width=0)  # Remove bar borders
        )
    )

    return fig

def create_timeline_chart(df):
    """Create timeline chart for top countries by month"""
    if df.empty:
        st.error("No data available for timeline chart")
        return None
    
    # Process data for timeline
    df_timeline = df.sort_values('month_date').copy()
    df_timeline['start_date'] = df_timeline['month_date']
    df_timeline['end_date'] = df_timeline['month_date'] + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    
    # Create categorical y-axis based on unique countries
    unique_countries = df_timeline['country_code'].unique()
    country_positions = {country: i for i, country in enumerate(unique_countries)}
    df_timeline['y_position'] = df_timeline['country_code'].map(country_positions)
    
    # Create the timeline chart
    fig = go.Figure()
    
    # Add bars for each country's dominant periods
    colors = px.colors.qualitative.Dark2[:len(unique_countries)]
    country_colors = {country: colors[i % len(colors)] for i, country in enumerate(unique_countries)}
    
    for _, row in df_timeline.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['start_date'], row['end_date'], row['end_date'], row['start_date'], row['start_date']],
            y=[row['y_position']-0.4, row['y_position']-0.4, row['y_position']+0.4, row['y_position']+0.4, row['y_position']-0.4],
            fill='toself',
            fillcolor=country_colors[row['country_code']],
            line=dict(color=country_colors[row['country_code']], width=1),
            hovertemplate=f"<b>{row['country_code']}</b><br>" +
                         f"Month: {row['month_date'].strftime('%Y-%m')}<br>" +
                         f"Publications: {row['publication_count']}<br>" +
                         f"Unique Countries: {row['unique_countries']}<extra></extra>",
            showlegend=False,
            name=row['country_code']
        ))
    
    # Add manual legend
    legend_traces = []
    for country in unique_countries:
        legend_traces.append(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=country_colors[country]),
                name=country,
                showlegend=True
            )
        )
    
    for trace in legend_traces:
        fig.add_trace(trace)
    
    # Update layout
    fig.update_layout(
        title=dict(
            # text="Timeline of Top Publishing Countries by Month",
            text="",
            x=0.5,
            font=dict(size=18, family="Arial, sans-serif"),
        ),
        xaxis=dict(
            title="Time Period",
            showgrid=False,
            zeroline=False,
            tickformat='%Y-%m'
        ),
        yaxis=dict(
            title="Country",
            tickmode='array',
            tickvals=list(range(len(unique_countries))),
            ticktext=unique_countries,
            showgrid=False,
            zeroline=False
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=max(400, len(unique_countries) * 50),
        width=1000,
        margin=dict(l=100, r=50, t=80, b=60),
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="black",
            borderwidth=1
        )
    )
    
    return fig

def create_yoy_growth_chart(df):
    """Create YoY growth rate line chart"""
    if df.empty or 'year' not in df.columns:
        st.error("Invalid data for YoY growth chart")
        return None
    
    fig = go.Figure()
    
    # Single line trace
    df = df.sort_values('year')
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['yoy_growth'],
        mode='lines+markers',
        line=dict(width=3, color='#2E8B57'),
        marker=dict(size=8, color='#2E8B57'),
        hovertemplate="Year: %{x}<br>Growth Rate: %{y:.1f}%<extra></extra>",
        showlegend=False
    ))
    
    # Zero line reference
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        opacity=0.6
    )

    # Add CAGR line
    cagr = df["cagr"].max()
    fig.add_hline(
        y=cagr,
        line_dash="dot",
        line_color="red",
        annotation_text=f"CAGR: {cagr:.1f}%",
        annotation_position="top right"
    )
    
    # Layout
    fig.update_layout(
        # title="Year-over-Year Growth Rate of Patent Publications in English (2017-2024)",
        xaxis_title="Year",
        yaxis_title="YoY Growth Rate (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        width=800,
        height=500,
        margin=dict(l=50, r=50, t=60, b=50),
        xaxis=dict(
            showgrid=False,
            tickmode='linear',
            dtick=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5
        )
    )
    
    return fig

def create_citation_table(df):
    """Create styled citation analysis table"""
    if df.empty:
        return None
    
    cols_to_keep = ["country_code", "total_patents", "patent_share", "patents_with_citations", 
                    "citation_rate_pct", "avg_citations_per_patent", "highly_cited_patents"]
    
    # Filter columns that actually exist in the dataframe
    available_cols = [col for col in cols_to_keep if col in df.columns]
    
    if not available_cols:
        return df
    
    styled_df = df[available_cols].sort_values(
        available_cols[1] if len(available_cols) > 1 else available_cols[0], 
        ascending=False
    ).style.background_gradient(
        subset=available_cols[1:],
        cmap="RdYlGn"
    )
    
    return styled_df

def create_cpc_bar_chart(df, column_name="cpc_share", title_prefix="Top 5 CPCs"):
    """Create horizontal bar chart for CPC analysis"""
    if df.empty:
        st.error("No data available for CPC chart")
        return None
    
    # Determine the y-axis column
    y_col = "cpc_code" if "cpc_code" in df.columns else df.columns[1]
    x_col = column_name if column_name in df.columns else df.columns[1]

    if title_prefix == "Technology Convergence":
        custom_text = [f"{val/1000:.1f}K" for val in df[x_col]]
        hover_text = '<b>Avg# of Patents%=%{x}</b><br><b>Classification=%{y}<extra></extra>'
    else:
        custom_text = [f"{val:.2f}%" for val in df[x_col]]
        hover_text = '<b>Share%=%{x}</b><br><b>Classification=%{y}<extra></extra>'
    
    
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        text=custom_text,
        # title=f"{title_prefix} by Share",
        labels={y_col: "Classification", x_col: "Share %"} if title_prefix != "Technology Convergence" else {y_col: "Classification", x_col: "Avg # of Patents"},
        color_discrete_sequence=["#1f77b4"],
        orientation="h"
    )
    
    if title_prefix == "Technology Convergence":
        fig.update_traces(
            marker=dict(
                color='#1f77b4',
                opacity=0.8,
                line=dict(width=0)
            ),
            hovertemplate=hover_text
    )
    else:
        fig.update_traces(
            texttemplate='%{text}', 
            textposition='inside',
            marker=dict(
                color='#1f77b4',
                opacity=0.8,
                line=dict(width=0)
            ),
            hovertemplate=hover_text
        )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            tickangle=0,
            showgrid=False,
            zeroline=False,
            title="Classification",
            autorange='reversed'
        ),
        xaxis=dict(
            title="Percentage" if title_prefix != "Technology Convergence" else "Avg # of Patents",
            showgrid=False,
            zeroline=False
        ),
        showlegend=False,
        height=500,
        width=1100,
        title=dict(
            # text=f"{title_prefix} Percentage (2017 - 2025)",
            text="",
            x=0.5,
            font=dict(size=16)
        ),
        margin=dict(l=20, r=20, t=60, b=60)
    )
    
    return fig

def create_sankey_chart(df):
    """Create Sankey diagram for patent flow"""
    if df.empty:
        st.error("No data available for Sankey chart")
        return None
    
    try:
        years = sorted(df['year'].unique())
        countries = sorted(df['country_code'].unique())
        sections = sorted(df['section_description'].unique())
        
        # Create unique node labels
        year_nodes = [f"Year {year}" for year in years]
        country_nodes = [f"{country}" for country in countries]
        section_nodes = [f"{section}" for section in sections]
        
        # Combine all nodes
        all_nodes = year_nodes + country_nodes + section_nodes
        
        # Create node indices mapping
        node_dict = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Prepare data for flows
        source_indices = []
        target_indices = []
        values = []
        
        # Define color palettes for each level
        year_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        country_colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#17becf', '#bcbd22']
        section_colors = ['#c5b0d5', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#17becf', '#bcbd22', '#2ca02c', '#d62728', '#9467bd']
        
        # Assign colors to nodes
        node_colors = []
        for node in all_nodes:
            if node in year_nodes:
                idx = year_nodes.index(node)
                node_colors.append(year_colors[idx % len(year_colors)])
            elif node in country_nodes:
                idx = country_nodes.index(node)
                node_colors.append(country_colors[idx % len(country_colors)])
            else:  # section nodes
                idx = section_nodes.index(node)
                node_colors.append(section_colors[idx % len(section_colors)])
        
        # Create Year → Country flows
        year_country_flows = df.groupby(['year', 'country_code'])['publication_cpc_count'].sum().reset_index()
        for _, row in year_country_flows.iterrows():
            source_idx = node_dict[f"Year {row['year']}"]
            target_idx = node_dict[f"{row['country_code']}"]
            source_indices.append(source_idx)
            target_indices.append(target_idx)
            values.append(row['publication_cpc_count'])
        
        # Create Country → Section flows
        for _, row in df.iterrows():
            source_idx = node_dict[f"{row['country_code']}"]
            target_idx = node_dict[f"{row['section_description']}"]
            source_indices.append(source_idx)
            target_indices.append(target_idx)
            values.append(row['publication_cpc_count'])
        
        # Create the Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes,
                color=node_colors,
                x=[0.1] * len(year_nodes) + [0.5] * len(country_nodes) + [0.9] * len(section_nodes),
                y=None,
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color="rgba(128, 128, 128, 0.3)"
            ),
            arrangement='snap',
        )])
        
        # Update layout
        fig.update_layout(
            title=dict(
                # text="Patent Publications Flow: Year → Country → Technology Section",
                text="",
                x=0.5,
                font=dict(size=20, family="Arial, sans-serif")
            ),
            font=dict(size=12),
            height=800,
            width=1200,
            margin=dict(l=50, r=50, t=80, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Add annotations for the three levels
        fig.add_annotation(x=0.1, y=1.05, text="<b>Years</b>", showarrow=False, 
                          xref="paper", yref="paper", font=dict(size=14))
        fig.add_annotation(x=0.5, y=1.05, text="<b>Countries</b>", showarrow=False,
                          xref="paper", yref="paper", font=dict(size=14))
        fig.add_annotation(x=0.9, y=1.05, text="<b>Technology Sections</b>", showarrow=False,
                          xref="paper", yref="paper", font=dict(size=14))
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating Sankey chart: {str(e)}")
        return None

def main():
    st.title("⚗️ Patent Analytics Dashboard")
    st.markdown("### Comprehensive Patent Landscape Analysis (2017-2025)")
    
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        
        # Date range filter
        st.subheader("Date Range")
        MIN_DATE = date(2017, 1, 1)
        MAX_DATE = date(2024, 12, 31)

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
        
        # Country filter
        st.subheader("Countries")
        
        COUNTRY_LIST = ['US', 'KR', 'WO', 'EP', 'RU', 'CN', 'CA', 'JP', 'TW', 'AU']
        use_country_filter = st.selectbox("Country filtering:", ["All countries", "Select specific"])
        if use_country_filter == "Select specific":
            selected_countries = st.multiselect(
                "Select Countries", 
                COUNTRY_LIST, 
                # default=["All"]
            )
        else:
            selected_countries = None
        
        # # Technology filter
        # st.subheader("Technology Areas")
        # tech_areas = ['All', 'Human Necessities', 'Operations & Transport', 
        #               'Chemistry & Metallurgy', 'Textiles', 'Construction',
        #               'Mechanical Engineering', 'Physics', 'Electricity',
        #               'Emerging Technologies']
        # selected_tech = st.selectbox("Technology Focus", tech_areas)
        
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Summary", "🌍 Countries", "🔬 Technology (CPC)", 
        "📈 Trends"
    ])
    
    with tab1:
        st.header("Patent Portfolio Overview")
        
        # Load and display summary metrics
        summary_df = load_summary_data()
        
        if not summary_df.empty:
            # Update metrics based on your styled column names
            display_metrics_html_streamlit(summary_df)
            # metrics_config = {
            #     'total_patents': {'label': 'Total Patents', 'format': 'comma'},
            #     'unique_countries': {'label': 'Countries', 'format': 'comma'},
            #     'unique_families': {'label': 'Patent Families', 'format': 'comma'},
            #     'avg_title_length': {'label': 'Avg Title Length', 'format': 'decimal'}
            # }
            
            # create_metric_cards(summary_df, metrics_config)
            
            # # Quality metrics row
            # st.subheader("Data Quality Metrics")
            # quality_metrics = {
            #     'title_completeness_pct': {'label': 'Title Completeness', 'format': 'percentage'},
            #     'abstract_completeness_pct': {'label': 'Abstract Completeness', 'format': 'percentage'},
            #     'claims_completeness_pct': {'label': 'Claims Completeness', 'format': 'percentage'}
            # }
            
            # create_metric_cards(summary_df, quality_metrics)
            
            # # Additional metrics if available
            # if 'avg_abstract_length' in summary_df.columns:
            #     st.subheader("Content Analysis")
            #     content_metrics = {
            #         'avg_abstract_length': {'label': 'Avg Abstract Length', 'format': 'decimal'},
            #     }
            #     create_metric_cards(summary_df, content_metrics)
        else:
            st.error("Unable to load summary data")
    
    with tab2:
        st.header("Country Analysis")
        
        # Load country data
        country_df = load_country_data()
        
        if not country_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Country publications bar chart
                fig_countries = create_country_bar_chart(country_df)
                if fig_countries:
                    st.plotly_chart(fig_countries, width=True)
            
            with col2:
                # Top countries metrics
                st.subheader("Leading Countries")
                for i, row in country_df.head(5).iterrows():
                    with st.container():
                        st.markdown(f"**{row['country_code']}**")
                        if 'percentage' in country_df.columns:
                            st.markdown(f"Share: {row['percentage']:.1f}%")
                        if 'patent_count' in country_df.columns:
                            st.markdown(f"Patents: {row['patent_count']:,.0f}")
                        st.markdown("---")
        
        # Timeline of top countries by month
        st.subheader("Country Dominance Timeline")
        timeline_df = load_country_data()
        if not timeline_df.empty:
            try:
                timeline_data = top_country_each_month()
                fig_timeline = create_timeline_chart(timeline_data)
                if fig_timeline:
                    st.plotly_chart(fig_timeline, width=True)
            except:
                st.info("Timeline data not available - implement top_country_each_month() function")
        
        # Citation analysis if available
        citation_df = load_citation_data()
        if not citation_df.empty:
            st.subheader("Citation Analysis by Country")
            styled_table = create_citation_table(citation_df)
            if styled_table is not None:
                st.dataframe(styled_table, width='content')
            else:
                st.dataframe(citation_df.style.format({
                    col: '{:.1%}' if 'rate' in col.lower() or 'pct' in col.lower() else '{:,.0f}'
                    for col in citation_df.select_dtypes(include=['number']).columns
                }), width=True)
            st.write("While China leads in patent volume, US patents show higher citation rates (72% cited vs. 36% for Chinese patents), suggesting different innovation strategies. " \
            "Semantic search could help identify high-quality innovations across all countries regardless of citation patterns or publication language.")
    
    with tab3:
        st.header("Technology Classification (CPC) Analysis")
        
        cpc_df = load_cpc_data()
        tech_area_df = load_tech_area_data()
        tech_convergence_df = load_tech_convergence_data()
        
        col1, col2, col3 = st.columns(3)
        
        # Top CPCs analysis
        if not cpc_df.empty:
            with col1:
                st.subheader(f"Top {len(cpc_df)} CPC Classifications")
                fig_cpc = create_cpc_bar_chart(cpc_df, column_name="cpc_share", title_prefix="Top 5 CPCs")
                if fig_cpc:
                    st.plotly_chart(fig_cpc, width=True)
                else:
                    # Fallback chart if column names are different
                    if len(cpc_df.columns) >= 2:
                        fig_cpc_alt = px.bar(
                            cpc_df.head(5),
                            x=cpc_df.columns[1],
                            y=cpc_df.columns[0],
                            orientation='h',
                            title=f"Top {len(cpc_df)} CPC Classifications"
                        )
                        st.plotly_chart(fig_cpc_alt, width=True)
                
                st.markdown("""
                ### 📘 Legend

                    **Y — Emerging Technologies**
                    - Y02E60/10 — Energy storage using batteries

                    **A — Human Necessities**
                    - A61P35/00 — Antineoplastic agents  
                    - A61K45/06 — Mixtures of active ingredients without chemical characterisation (e.g. antiphlogistics and cardiaca)

                    **G — Physics**
                    - G06N3/08 — Learning methods (Neural Networks)  
                    - G06N20/00 — Machine learning
                """)

        
        # Technology areas analysis  
        if not tech_area_df.empty:
            with col2:
                st.subheader("Technology Areas")
                fig_tech = create_cpc_bar_chart(tech_area_df, column_name="section_patent_percentage", title_prefix="Technology Areas")
                if fig_tech:
                    st.plotly_chart(fig_tech, width=True)
                    st.markdown("""
                    ### 📋 Technical Notes

                    **CPC Section Definition**: The 'section' in a CPC code is the first letter. For example, in CPC code `G06N3/08`, the letter `G` represents the Physics section.

                    **Data Methodology**: Unique patent counts were aggregated by section to avoid double-counting patents with multiple CPC codes in the same section.

                    ### 📚 Official References

                    - **[USPTO CPC Scheme](https://www.uspto.gov/web/patents/classification/cpc/html/cpc.html)** - Complete classification hierarchy and definitions
                    - **[Cooperative Patent Classification](https://www.cooperativepatentclassification.org/)** - Joint USPTO-EPO classification system documentation
                    - **[CPC Browser](https://worldwide.espacenet.com/patent/cpc-browser?locale=en_EP)** - Interactive classification explorer

                    ---
                    *Analysis based on 49M+ patents from the Google Patents Public Dataset (2017-2025)*
                    """)
                else:
                    # Fallback chart
                    if len(tech_area_df.columns) >= 2:
                        fig_tech_alt = px.bar(
                            tech_area_df.head(8),
                            x=tech_area_df.columns[1],
                            y=tech_area_df.columns[0],
                            orientation='h',
                            title="Technology Areas Analysis"
                        )
                        st.plotly_chart(fig_tech_alt, width=True)
        
        # Technology convergence analysis
        if not tech_convergence_df.empty:
            with col3:
                st.subheader("Technology Convergence")
                fig_tech = create_cpc_bar_chart(tech_convergence_df, column_name="avg_recent_patents", title_prefix="Technology Convergence")
                if fig_tech:
                    st.plotly_chart(fig_tech, width=True)
                    st.markdown("""
                    ### 📋 Technical Notes

                    **CPC Section Definition**: The 'section' in a CPC code is the first letter. For example, in CPC code `G06N3/08`, the letter `G` represents the Physics section.
                    
                    **Convergence Analysis**: "Physics + Electronics" represents patents that span both domains, capturing technologies like:
                    - Computing/AI hardware and semiconductors
                    - Smart sensors and measurement electronics  
                    - Optical-electronic systems
                    - Digital signal processing devices
                    """)

        
        # Detailed tables
        tab3_col1, tab3_col2 = st.columns(2)
        
        with tab3_col1:
            if not cpc_df.empty:
                st.subheader("CPC Details")
                st.dataframe(cpc_df.head(10), width="content")
        
        with tab3_col2:
            if not tech_area_df.empty:
                st.subheader("Technology Area Details")
                st.dataframe(tech_area_df.head(10), width="content")
    
    with tab4:
        st.header("Patent Publication Trends & Flow")
        
        # Load time series data
        trends_df = load_time_series_data()
        if selected_countries:
            filter_clause = " AND a.country_code IN (" + ", ".join([f"'{c}'" for c in selected_countries]) + ")"
            country_trends_df = load_country_trends_data(filter_clause=filter_clause)
        else:
           filter_clause = " AND a.country_code IN (" + ", ".join([f"'{c}'" for c in COUNTRY_LIST]) + ")"
           country_trends_df = load_country_trends_data(filter_clause=filter_clause) 
        
        if not trends_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Overall YoY growth
                fig_growth = create_yoy_growth_chart(trends_df)
                if fig_growth:
                    st.plotly_chart(fig_growth, width=True)
            
            with col2:
                # Key trend metrics
                if 'yoy_growth' in trends_df.columns:
                    latest_growth = trends_df['yoy_growth'].iloc[-1]
                    cagr = trends_df['cagr'].max()
                    
                    st.subheader("Growth Metrics")
                    st.metric("Latest Growth Rate", f"{latest_growth:.1f}%")
                    st.metric("CAGR", f"{cagr:.1f}%")
        
        # Multi-country trends
        if not country_trends_df.empty:
            st.subheader("Country-wise Growth Trends")
            fig_multi_country = create_multi_country_yoy_chart(country_trends_df)
            if fig_multi_country:
                st.plotly_chart(fig_multi_country, width=True)
        
        # Patent Flow Sankey Chart
        try:
            flow_df = load_patent_flow_data()
            if not flow_df.empty:
                st.subheader("Patent Publication Flow")
                fig_sankey = create_sankey_chart(flow_df)
                if fig_sankey:
                    st.plotly_chart(fig_sankey, width=True)
                else:
                    st.info("Sankey chart data structure not compatible - showing data table")
                    st.dataframe(flow_df.head(20), width=True)
            else:
                st.info("No patent flow data available")
        except Exception as e:
            st.error(f"Error loading patent flow: {str(e)}")
            
def create_multi_country_yoy_chart(df):
    """Create multi-line YoY growth chart by countries"""
    if df.empty:
        st.error("No data available for multi-country YoY chart")
        return None
    
    fig = px.line(
        df.sort_values('year'),
        x="year",
        y="yoy_growth",
        color="country_code",
        markers=True,
        # title="Year-over-Year Growth Rate by Top 10 Countries",
        labels={
            "year": "Year",
            "yoy_growth": "YoY Growth Rate (%)",
            "country_code": "Country"
        },
        color_discrete_sequence=px.colors.qualitative.Dark2
    )
    
    # Update traces
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8, line=dict(width=1, color='white')),
        opacity=0.6,
        hovertemplate="<b>%{fullData.name}</b><br>" +
                     "Year: %{x}<br>" +
                     "Growth Rate: %{y:.2f}%<extra></extra>"
    )
    
    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        opacity=0.7,
        annotation_text="0% Growth",
        annotation_position="bottom right"
    )
    
    # Update layout
    fig.update_layout(
        plot_bgcolor='rgba(0 ,0, 0, 0)',
        paper_bgcolor='rgba(0 ,0, 0, 0)',
        xaxis=dict(
            title="Year",
            showgrid=False,
            zeroline=False,
            tickmode='linear',
            dtick=1
        ),
        yaxis=dict(
            title="YoY Growth Rate (%)",
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5,
            zeroline=False
        ),
        title=dict(
            # text="Year-over-Year Growth Rate by Top 10 Countries",
            text="",
            x=0.5,
            font=dict(size=18, family="Arial, sans-serif")
        ),
        legend=dict(
            title="Country",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(0,0,0,0.9)",
            bordercolor="lightgray",
            borderwidth=1
        ),
        height=600,
        width=1000,
        margin=dict(l=60, r=120, t=80, b=60),
        hovermode='x unified'
    )
    
    return fig
    
    # with tab5:
    #     st.header("Semantic Search Demo")
        
    #     st.markdown("""
    #     This section demonstrates the semantic patent search capabilities built using BigQuery ML 
    #     and vector embeddings.
    #     """)
        
    #     # Search interface
    #     col1, col2 = st.columns([3, 1])
        
    #     with col1:
    #         search_query = st.text_input(
    #             "Enter search query:",
    #             placeholder="e.g., artificial intelligence machine learning",
    #             key="semantic_search"
    #         )
        
    #     with col2:
    #         search_button = st.button("🔍 Search", type="primary")
        
    #     if search_query and search_button:
    #         with st.spinner("Searching patent database..."):
    #             # Simulate search results
    #             st.success(f"Found semantic matches for: '{search_query}'")
                
    #             # Mock search results
    #             results_data = {
    #                 'Patent ID': ['US2024123456A1', 'CN2024789012A', 'EP2024345678A1'],
    #                 'Title': [
    #                     'Neural Network Architecture for Autonomous Vehicles',
    #                     'Deep Learning System for Medical Diagnosis',
    #                     'Machine Learning Framework for Industrial Automation'
    #                 ],
    #                 'Country': ['US', 'CN', 'EP'],
    #                 'Similarity Score': [0.89, 0.82, 0.76],
    #                 'Publication Date': ['2024-03-15', '2024-02-28', '2024-01-10']
    #             }
                
    #             results_df = pd.DataFrame(results_data)
    #             st.dataframe(
    #                 results_df.style.format({'Similarity Score': '{:.2f}'}),
    #                 width=True
    #             )
        
    #     # Semantic search statistics
    #     st.subheader("Search Capabilities")
        
    #     search_stats_col1, search_stats_col2, search_stats_col3 = st.columns(3)
        
    #     with search_stats_col1:
    #         st.metric("Indexed Patents", "45,000", help="Patents with semantic embeddings")
    #     with search_stats_col2:
    #         st.metric("Vector Dimensions", "768", help="Embedding dimension size")
    #     with search_stats_col3:
    #         st.metric("Avg Search Time", "0.8s", help="Query response time")

if __name__ == "__main__":
    main()