"""
Patent Visualization Functions
Extracted chart functions and data loading functions for Kaggle notebook use
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from loguru import logger
from datetime import datetime, date

from generate_patent_analysis import (
    dataset_size_table, country_wise_breakdown, top_country_each_month,
    yoy_lang_growth_rate, yoy_country_growth_rate, citations_top_countries,
    top_cpc, tech_area_cpc, tech_convergence, patent_flow
)

# Data loading functions
def load_summary_data():
    """Load actual summary statistics from BigQuery"""
    try:
        return dataset_size_table()
    except Exception as e:
        print(f"Error loading summary data: {e}")
        return pd.DataFrame()

def load_country_data(top_n=10):
    """Load actual country-wise statistics"""
    try:
        return country_wise_breakdown(top_n)
    except Exception as e:
        print(f"Error loading country data: {e}")
        return pd.DataFrame()

def load_citation_data(top_n=10):
    """Load citation patterns for top countries"""
    try:
        return citations_top_countries(top_n)
    except Exception as e:
        print(f"Error loading citation data: {e}")
        return pd.DataFrame()

def load_cpc_data(top_n=10):
    """Load CPC classification data"""
    try:
        return top_cpc(top_n)
    except Exception as e:
        print(f"Error loading CPC data: {e}")
        return pd.DataFrame()

def load_tech_area_data(top_n=10):
    """Load technology area analysis data"""
    try:
        return tech_area_cpc(top_n)
    except Exception as e:
        print(f"Error loading tech area data: {e}")
        return pd.DataFrame()

def load_tech_convergence_data(top_n=5):
    """Load technology convergence analysis data"""
    try:
        return tech_convergence(top_n)
    except Exception as e:
        print(f"Error loading tech convergence data: {e}")
        return pd.DataFrame()

def load_time_series_data():
    """Load YoY trend data"""
    try:
        return yoy_lang_growth_rate()
    except Exception as e:
        print(f"Error loading time series data: {e}")
        return pd.DataFrame()

def load_country_trends_data(top_n=10):
    """Load country-wise YoY trends"""
    try:
        return yoy_country_growth_rate(top_n)
    except Exception as e:
        print(f"Error loading country trends: {e}")
        return pd.DataFrame()

def load_patent_flow_data():
    """Load patent flow data for Sankey"""
    try:
        return patent_flow()
    except Exception as e:
        print(f"Error loading patent flow data: {e}")
        return pd.DataFrame()

def load_timeline_data():
    """Load timeline data"""
    try:
        return top_country_each_month()
    except Exception as e:
        print(f"Error loading timeline data: {e}")
        return pd.DataFrame()

# Chart creation functions
def create_country_bar_chart(df):
    """Create country-wise distribution of publications"""
    if df.empty:
        logger.warning("No data available for country-wise bar chart")
        return None
    
    topn_countries_publication = df.sort_values('percentage', ascending=False)
    
    fig = px.bar(
        topn_countries_publication,
        x="percentage",
        y="country_code",
        text="percentage",
        labels={"country_code": "Country", "percentage": "Publication %"},
        color_discrete_sequence=["#1f77b4"],
        orientation="h"
    )

    fig.update_traces(
        texttemplate='%{text:.2f}%', 
        textposition='outside',
        marker=dict(
            color='#1f77b4',
            opacity=0.8,
            line=dict(width=0)
        )
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            tickangle=0,
            showgrid=False,
            zeroline=False,
            title="Country",
            autorange='reversed',
        ),
        xaxis=dict(
            title="Publication Percentage",
            showgrid=False,
            zeroline=False
        ),
        showlegend=False,
        height=500,
        width=1100,
        title=dict(
            text="Top 10 Countries by Publication Percentage (2017 - 2025)",
            x=0.5,
            font=dict(size=16)
        ),
        margin=dict(l=20, r=20, t=60, b=60)
    )

    return fig

def create_timeline_chart(df):
    """Create timeline chart for top countries by month"""
    if df.empty:
        logger.warning("No data available for timeline chart")
        return None
    
    df_timeline = df.sort_values('month_date').copy()
    df_timeline['start_date'] = df_timeline['month_date']
    df_timeline['end_date'] = df_timeline['month_date'] + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    
    unique_countries = df_timeline['country_code'].unique()
    country_positions = {country: i for i, country in enumerate(unique_countries)}
    df_timeline['y_position'] = df_timeline['country_code'].map(country_positions)
    
    fig = go.Figure()
    
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
    
    fig.update_layout(
        title=dict(
            text="Timeline of Top Publishing Countries by Month",
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
        logger.warning("Invalid data for YoY growth chart")
        return None
    
    fig = go.Figure()
    
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
    
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        opacity=0.6
    )

    cagr = df["cagr"].max()
    fig.add_hline(
        y=cagr,
        line_dash="dot",
        line_color="red",
        annotation_text=f"CAGR: {cagr:.1f}%",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="Year-over-Year Growth Rate of Patent Publications in English (2017-2024)",
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

def create_multi_country_yoy_chart(df):
    """Create multi-line YoY growth chart by countries"""
    if df.empty:
        print("No data available for multi-country YoY chart")
        return None
    
    fig = px.line(
        df.sort_values('year'),
        x="year",
        y="yoy_growth",
        color="country_code",
        markers=True,
        labels={
            "year": "Year",
            "yoy_growth": "YoY Growth Rate (%)",
            "country_code": "Country"
        },
        color_discrete_sequence=px.colors.qualitative.Dark2
    )
    
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8, line=dict(width=1, color='white')),
        opacity=0.6,
        hovertemplate="<b>%{fullData.name}</b><br>" +
                     "Year: %{x}<br>" +
                     "Growth Rate: %{y:.2f}%<extra></extra>"
    )
    
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        opacity=0.7,
        annotation_text="0% Growth",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
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
            text="Year-over-Year Growth Rate by Top 10 Countries",
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

def create_cpc_bar_chart(df, column_name="cpc_share", title_prefix="Top 5 CPCs"):
    """Create horizontal bar chart for CPC analysis"""
    if df.empty:
        logger.warning("No data available for CPC chart")
        return None
    
    y_col = "cpc_code" if "cpc_code" in df.columns else df.columns[0]
    x_col = column_name if column_name in df.columns else df.columns[1]
    
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        text=x_col,
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
            )
        )
    else:
        fig.update_traces(
            texttemplate='%{text:.2f}%', 
            textposition='outside',
            marker=dict(
                color='#1f77b4',
                opacity=0.8,
                line=dict(width=0)
            )
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
            text=f"{title_prefix} Analysis (2017 - 2025)",
            x=0.5,
            font=dict(size=16)
        ),
        margin=dict(l=20, r=20, t=60, b=60)
    )
    
    return fig

def create_sankey_chart(df):
    """Create Sankey diagram for patent flow"""
    if df.empty:
        logger.warning("No data available for Sankey chart")
        return None
    
    try:
        years = sorted(df['year'].unique())
        countries = sorted(df['country_code'].unique())
        sections = sorted(df['section_description'].unique())
        
        year_nodes = [f"Year {year}" for year in years]
        country_nodes = [f"{country}" for country in countries]
        section_nodes = [f"{section}" for section in sections]
        
        all_nodes = year_nodes + country_nodes + section_nodes
        node_dict = {node: idx for idx, node in enumerate(all_nodes)}
        
        source_indices = []
        target_indices = []
        values = []
        
        year_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        country_colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#17becf', '#bcbd22']
        section_colors = ['#c5b0d5', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#17becf', '#bcbd22', '#2ca02c', '#d62728', '#9467bd']
        
        node_colors = []
        for node in all_nodes:
            if node in year_nodes:
                idx = year_nodes.index(node)
                node_colors.append(year_colors[idx % len(year_colors)])
            elif node in country_nodes:
                idx = country_nodes.index(node)
                node_colors.append(country_colors[idx % len(country_colors)])
            else:
                idx = section_nodes.index(node)
                node_colors.append(section_colors[idx % len(section_colors)])
        
        year_country_flows = df.groupby(['year', 'country_code'])['publication_cpc_count'].sum().reset_index()
        for _, row in year_country_flows.iterrows():
            source_idx = node_dict[f"Year {row['year']}"]
            target_idx = node_dict[f"{row['country_code']}"]
            source_indices.append(source_idx)
            target_indices.append(target_idx)
            values.append(row['publication_cpc_count'])
        
        for _, row in df.iterrows():
            source_idx = node_dict[f"{row['country_code']}"]
            target_idx = node_dict[f"{row['section_description']}"]
            source_indices.append(source_idx)
            target_indices.append(target_idx)
            values.append(row['publication_cpc_count'])
        
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
        
        fig.update_layout(
            title=dict(
                text="Patent Publications Flow: Year → Country → Technology Section",
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
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating Sankey chart: {str(e)}")
        return None

def create_citation_table_styled(df):
    """Create styled citation analysis table"""
    if df.empty:
        return None
    
    cols_to_keep = ["country_code", "total_patents", "patent_share", "patents_with_citations", 
                    "citation_rate_pct", "avg_citations_per_patent", "highly_cited_patents"]
    
    available_cols = [col for col in cols_to_keep if col in df.columns]
    
    if not available_cols:
        return df
    
    return df[available_cols].sort_values(
        available_cols[1] if len(available_cols) > 1 else available_cols[0], 
        ascending=False
    )

# Demo function to display all charts
def create_patent_dashboard_demo():
    """Create a demonstration of all patent visualization charts"""
    
    logger.info("Loading Patent Analytics Dashboard Demo")
    logger.info("=====================================")
    
    # Load data
    logger.info("1. Loading Summary Data...")
    summary_df = load_summary_data()
    if not summary_df.empty:
        logger.info(f"   Loaded summary data with {len(summary_df)} records")
        logger.info(f"   Columns: {list(summary_df.columns)}")
    
    logger.info("2. Loading Country Data...")
    country_df = load_country_data()
    if not country_df.empty:
        logger.info(f"   Loaded country data with {len(country_df)} records")
        fig = create_country_bar_chart(country_df)
        if fig:
            fig.show()
    
    logger.info("3. Loading CPC Data...")
    cpc_df = load_cpc_data()
    if not cpc_df.empty:
        logger.info(f"   Loaded CPC data with {len(cpc_df)} records")
        fig = create_cpc_bar_chart(cpc_df)
        if fig:
            fig.show()
    
    logger.info("4. Loading Time Series Data...")
    trends_df = load_time_series_data()
    if not trends_df.empty:
        logger.info(f"   Loaded trends data with {len(trends_df)} records")
        fig = create_yoy_growth_chart(trends_df)
        if fig:
            fig.show()
    
    logger.info("5. Loading Country Trends Data...")
    country_trends_df = load_country_trends_data()
    if not country_trends_df.empty:
        logger.info(f"   Loaded country trends with {len(country_trends_df)} records")
        fig = create_multi_country_yoy_chart(country_trends_df)
        if fig:
            fig.show()
    
    logger.info("6. Loading Patent Flow Data...")
    flow_df = load_patent_flow_data()
    if not flow_df.empty:
        logger.info(f"   Loaded flow data with {len(flow_df)} records")
        fig = create_sankey_chart(flow_df)
        if fig:
            fig.show()
    
    logger.success("Dashboard demo complete!")

if __name__ == "__main__":
    # Run the demo
    create_patent_dashboard_demo()