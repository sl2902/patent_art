"""Display viusals tracking various metrics to analyse latency, partition pruning and semantic search discoverability"""
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Any, List, Dict, Optional, Tuple
from loguru import logger
from generate_patent_metrics_analysis import (
    latency_measurement,
    partition_pruning_efficiency_measurement,
    bytes_and_time_reduction_measurement,
    discovery_rate_measurement
)

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

try:
    from IPython.display import display, HTML
    HAS_KAGGLE = True
except ImportError:
    HAS_KAGGLE = False



def create_bigquery_partition_efficiency(efficiency_df: pd.DataFrame, reduction_df:pd.DataFrame):
    """
    Create a 2-panel dashboard showing BigQuery AI partition pruning efficiency
    
    Args:
        efficiency_df (pd.DataFrame): DataFrame with columns:
            - run_environment, test_type, avg_search_time_sec, avg_bytes_processed_gb, avg_results
            
        reduction_df (pd.DataFrame): DataFrame with columns:
            - run_environment, test_type, bytes_reduction_pct, time_reduction_pct
    """
    
    # Create 2-panel subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'BigQuery AI Partition Optimization: Cost vs Performance',
            'Efficiency Analysis: Time vs Cost Trade-offs'
        ),
        specs=[[{"secondary_y": True}, {"type": "scatter"}]],
        horizontal_spacing=0.15
    )
    
    # Use Kaggle environment data (production-like setup)
    kaggle_reduction = reduction_df[
        (reduction_df['run_environment'] == 'kaggle') & 
        (reduction_df['test_type'] != 'full_scan')
    ].copy()
    
    # Define partition types and labels
    partition_types = ['partition_1_month', 'partition_3_months', 'partition_6_months']
    partition_labels = ['1-Month', '3-Month', '6-Month']
    
    # Panel 1: Cost vs Performance Trade-off
    cost_reduction = kaggle_reduction['bytes_reduction_pct'].astype(float)
    time_improvement = kaggle_reduction['time_reduction_pct'].astype(float)
    
    # Cost reduction bars (primary y-axis)
    fig.add_trace(go.Bar(
        name='Cost Reduction',
        x=partition_labels,
        y=cost_reduction,
        marker_color='#2E86AB',
        opacity=0.8,
        text=[f'{val:.1f}%' for val in cost_reduction],
        textposition='auto',
        yaxis='y'
    ), row=1, col=1)
    
    # Performance improvement line
    fig.add_trace(go.Scatter(
        name='Performance Improvement',
        x=partition_labels,
        y=time_improvement,
        mode='markers+lines',
        marker=dict(size=12, color='#A23B72', symbol='diamond'),
        line=dict(width=4, color='#A23B72'),
        text=[f'{val:.1f}%' for val in time_improvement],
        textposition='top center',
        yaxis='y2'
    ), row=1, col=1, secondary_y=True)

    kaggle_efficiency = efficiency_df[efficiency_df['run_environment'] == 'kaggle']
    partition_efficiency = kaggle_efficiency[kaggle_efficiency['test_type'].isin(partition_types)]
    
    bytes_processed = partition_efficiency['avg_bytes_processed_gb'].astype(float).values

    min_size, max_size = 15, 25
    if len(bytes_processed) > 1:
        inverted_bytes = bytes_processed.max() - bytes_processed + bytes_processed.min()
        normalized_sizes = min_size + (inverted_bytes - inverted_bytes.min()) *  (max_size - min_size) / (inverted_bytes.max() - inverted_bytes.min())
    else:
        normalized_sizes = [20] # default for single point
    
    # Panel 2: Efficiency Scatter Plot
    fig.add_trace(go.Scatter(
        name='Partition Strategies',
        x=time_improvement,
        y=cost_reduction,
        mode='markers+text',
        marker=dict(
            size=normalized_sizes,  # Dynamic sizing based on bytes processed
            color=['#F18F01', '#C73E1D', '#592E83'],
            opacity=0.8,
            line=dict(width=2, color='white')
        ),
        text=['1-Month', '3-Month', '6-Month'],
        textposition='middle right',
        textfont=dict(size=12, color='white'),
        showlegend=False,
        customdata=bytes_processed,
        hovertemplate='<b>%{text}</b><br>' +
                      'Performance Improvement: %{x:.1f}%<br>' +
                      'Cost Reduction: %{y:.1f}%<br>' +
                      'Data Processed: %{customdata:.1f} GB<br>' +
                      '<extra></extra>'
    ), row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'BigQuery AI Vector Search: Partition Pruning Efficiency<br><sub>84% Cost Reduction with Performance Gains Through Intelligent Partitioning</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16},
            'pad': {'b': 30}  # Space below title/subtitle
        },
        height=650,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,  # Higher position to create space above subplots
            xanchor="center",
            x=0.5
        ),
        font=dict(size=12),
        margin=dict(t=180, b=50, l=60, r=60)
    )
    
    # Update Panel 1 axes
    fig.update_xaxes(title_text="Partition Strategy", row=1, col=1)
    fig.update_yaxes(
        title_text="Cost Reduction (%)", 
        range=[0, 100],
        row=1, col=1
    )
    
    # Secondary y-axis for Panel 1
    fig.update_yaxes(
        title_text="Performance Improvement (%)",
        range=[0, 15],
        row=1, col=1,
        secondary_y=True
    )
    
    # Update Panel 2 axes
    fig.update_xaxes(
        title_text="Performance Improvement (%)",
        range=[0, 15],
        row=1, col=2
    )
    fig.update_yaxes(
        title_text="Cost Reduction (%)",
        range=[0, 100],
        row=1, col=2
    )
    
    # Add key insight annotation (focused on the main finding)
    fig.add_annotation(
        text="<b>Key Insight:</b> 1-month partitioning achieves<br>84% cost optimization with 12% performance gain",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=11, color="darkblue"),
        bgcolor="rgba(173, 216, 230, 0.9)",
        bordercolor="darkblue",
        borderwidth=1
    )
    
    # Add grid for better readability
    # fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    # fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def create_bigquery_visualization() -> Optional[go.Figure]:
    """
    Convenience function to create BigQuery AI visualization from raw data
    """
    efficiency_df = partition_pruning_efficiency_measurement()
    reduction_df = bytes_and_time_reduction_measurement()
    
    fig = create_bigquery_partition_efficiency(efficiency_df, reduction_df)
    if HAS_KAGGLE:
        fig.show()
    else:
        return fig

def create_discovery_comparison_dashboard(df: pd.DataFrame):
    """
    Create a comprehensive dashboard comparing semantic vs keyword search performance
    
    Args:
        df (pd.DataFrame): DataFrame with columns:
            - run_environment: 'kaggle' or 'laptop'
            - test_type: 'vector_search' or 'keyword_search'
            - avg_time_ms: average search time in milliseconds
            - total_semantic_results, total_keyword_results: result counts
            - total_overlap: overlapping results between methods
            - total_semantic_unique, total_keyword_unique: unique results
            - overall_discovery_rate: uniqueness search percentage
    """
    
    # Create subplot structure: 2x2 grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Semantic Search Uniqueness: Non-Overlapping Results',
            'Search Performance by Environment', 
            'Total Results Comparison',
            'Discovery Overlap Analysis'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "pie"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Process data for visualizations
    semantic_data = df[df['test_type'] == 'vector_search'].iloc[0]  # Both environments have same results
    keyword_data = df[df['test_type'] == 'keyword_search'].iloc[0]
    
    # Panel 1: Semantic Search Uniqueness Percentage Comparison (Key Finding)
    discovery_methods = ['Semantic Search', 'Keyword Search']
    unique_counts = [int(semantic_data['total_semantic_unique']), int(keyword_data['total_keyword_unique'])]
    discovery_colors = ['#1f77b4', '#ff7f0e']
    
    fig.add_trace(go.Bar(
        x=discovery_methods,
        y=unique_counts,
        name='Unique Patents',
        marker_color=discovery_colors,
        text=[f'{count}<br>({count/300*100:.1f}%)' for count in unique_counts],
        textposition='auto',
        showlegend=False
    ), row=1, col=1)
    
    # Panel 2: Performance Comparison by Environment
    environments = ['Kaggle', 'Laptop']
    semantic_times = [float(df[(df['test_type'] == 'vector_search') & (df['run_environment'] == 'kaggle')]['avg_time_ms'].iloc[0])/1000,
                     float(df[(df['test_type'] == 'vector_search') & (df['run_environment'] == 'laptop')]['avg_time_ms'].iloc[0])/1000]
    keyword_times = [float(df[(df['test_type'] == 'keyword_search') & (df['run_environment'] == 'kaggle')]['avg_time_ms'].iloc[0])/1000,
                    float(df[(df['test_type'] == 'keyword_search') & (df['run_environment'] == 'laptop')]['avg_time_ms'].iloc[0])/1000]
    
    fig.add_trace(go.Bar(
        x=environments,
        y=semantic_times,
        name='Semantic Search',
        marker_color='#1f77b4',
        text=[f'{time:.1f}s' for time in semantic_times],
        textposition='auto'
    ), row=1, col=2)
    
    fig.add_trace(go.Bar(
        x=environments,
        y=keyword_times,
        name='Keyword Search',
        marker_color='#ff7f0e',
        text=[f'{time:.1f}s' for time in keyword_times],
        textposition='auto'
    ), row=1, col=2)
    
    # Panel 3: Total Results Comparison
    total_methods = ['Semantic Search', 'Keyword Search']
    total_results = [int(semantic_data['total_semantic_results']), int(keyword_data['total_keyword_results'])]
    
    fig.add_trace(go.Bar(
        x=total_methods,
        y=total_results,
        name='Total Results',
        marker_color=['#2ca02c', '#d62728'],
        text=[f'{result} patents' for result in total_results],
        textposition='auto',
        showlegend=False
    ), row=2, col=1)
    
    # Panel 4: Discovery Overlap Analysis (Pie Chart)
    overlap_labels = ['Semantic Only', 'Keyword Only', 'Shared Results']
    overlap_values = [
        int(semantic_data['total_semantic_unique']), 
        int(keyword_data['total_keyword_unique']), 
        int(semantic_data['total_overlap'])
    ]
    overlap_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig.add_trace(go.Pie(
        labels=overlap_labels,
        values=overlap_values,
        marker_colors=overlap_colors,
        textinfo='label+percent+value',
        textposition='auto',
        showlegend=False
    ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Patent Discovery: Semantic vs Keyword Search Analysis<br>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16},
            'pad': {'b': 30}
        },
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        ),
        font=dict(size=12),
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Search Method", row=1, col=1)
    fig.update_yaxes(title_text="Unique Patents Found", row=1, col=1)
    
    fig.update_xaxes(title_text="Environment", row=1, col=2)
    fig.update_yaxes(title_text="Average Search Time (seconds)", row=1, col=2)
    
    fig.update_xaxes(title_text="Search Method", row=2, col=1)
    fig.update_yaxes(title_text="Total Patents Found", row=2, col=1)
    
    # Add annotation for key insight
    fig.add_annotation(
        text="<b>Key Insight:</b> 98.3% semantic search uniqueness<br>shows complementary search approaches",
        xref="paper", yref="paper",
        x=0.21, y=0.98,
        showarrow=False,
        font=dict(size=11, color="darkgreen"),
        bgcolor="lightgreen",
        opacity=0.8,
        bordercolor="darkgreen",
        borderwidth=1
    )
    
    return fig

def create_discoverability_visualization() -> Optional[go.Figure]:
    """
    Convenience function to create BigQuery AI visualization from raw data
    """
    df = discovery_rate_measurement()

    fig = create_discovery_comparison_dashboard(df)
    if HAS_KAGGLE:
        fig.show()
    else:
        return fig


def render_latency_chart(df):
    # Ensure numeric
    df["mean_latency"] = df["mean_latency"].astype(float)
    df["median_latency"] = df["median_latency"].astype(float)
    
    # max_val = df[["mean_latency", "median_latency"]].max().max()
    # min_val = df[["mean_latency", "median_latency"]].min().min()
    mean_max = df['mean_latency'].max()
    mean_min = df['mean_latency'].min()
    median_max = df['median_latency'].max()
    median_min = df['median_latency'].min()
    
    html = """
    <style>
      body { font-family: Arial, sans-serif; background: #fafafa; }
      h2 { margin-bottom: 20px; }
      .chart { max-width: 900px; }
      .row { margin: 14px 0; }
      .label { font-weight: bold; margin-bottom: 6px; color: #333; }
      .bar-group { display: flex; align-items: center; margin: 4px 0; }
      .bar-container { flex: 1; background: #eee; border-radius: 6px; margin: 0 6px; height: 22px; position: relative; }
      .bar { height: 100%; border-radius: 6px; text-align: right; color: #fff; padding-right: 6px; font-size: 12px; line-height: 22px; box-sizing: border-box; }
      .bar.green { background: #4caf50; }
      .bar.red   { background: #e53935; }
      .bar.blue  { background: #2196f3; }
      .metric-label { width: 55px; font-size: 12px; text-align: right; color: #666; }
      .value { width: 100px; font-size: 12px; text-align: left; color: #444; }
    </style>
    <h2>Latency Comparison (ms) — Mean vs Median</h2>
    <div class="chart">
    """

    for _, row in df.iterrows():
        html += f"""
        <div class="row">
          <div class="label">{row['test_type']} ({row['run_environment']})</div>
          <div class="bar-group">
            <div class="metric-label">Mean:</div>
            <div class="bar-container">
              <div class="bar {'green' if row['mean_latency']==mean_min else 'red' if row['mean_latency']==mean_max else 'blue'}"
                   style="width:{(row['mean_latency']/mean_max)*100}%;">
                   {row['mean_latency']:.0f} ms
              </div>
            </div>
            <div class="value">{'Fastest' if row['mean_latency']==mean_min else f"{(row['mean_latency']/mean_min):.2f}× slower"}</div>
          </div>
          <div class="bar-group">
            <div class="metric-label">Median:</div>
            <div class="bar-container">
              <div class="bar {'green' if row['median_latency']==median_min else 'red' if row['median_latency']==median_max else 'blue'}"
                   style="width:{(row['median_latency']/median_max)*100}%;">
                   {row['median_latency']:.0f} ms
              </div>
            </div>
            <div class="value">{'Fastest' if row['median_latency']==median_min else f"{(row['median_latency']/median_max):.2f}× slower"}</div>
          </div>
        </div>
        """
        
    note_html = """
    <div style="font-size:14px; margin-top:12px; color:#555; max-width:800px;">
      <strong>Note:</strong> Bars are scaled relative to the <em> slowest performance </em> in each category (mean vs median).
      The fastest value is highlighted in <span style="color:green;font-weight:bold;">green</span> and labeled "Fastest".
      The slowest is highlighted in <span style="color:red;font-weight:bold;">red</span>.
      All other values show how many times slower they are compared to the fastest baseline.
    </div>
    """
    
    html += note_html + "</div>"
    display(HTML(html))

def create_latency_visualization():
    """
    Convenience function to create BigQuery AI visualization from raw data
    """
    df = latency_measurement()

    render_latency_chart(df)

    
def render_latency_chart_streamlit(df: pd.DataFrame):
    """Create a Plotly version that matches the Kaggle HTML chart design"""
    df = latency_measurement()
    
    # Ensure numeric
    df["mean_latency"] = df["mean_latency"].astype(float)
    df["median_latency"] = df["median_latency"].astype(float)
    
    # Get min/max for scaling
    mean_max = df['mean_latency'].max()
    mean_min = df['mean_latency'].min()
    median_max = df['median_latency'].max()
    median_min = df['median_latency'].min()
    
    # Create figure with subplots - one row per test configuration
    fig = go.Figure()
    
    # Prepare data - create labels and organize by test configuration
    df['config_label'] = df['test_type'] + ' (' + df['run_environment'] + ')'
    
    y_positions = []
    mean_widths = []
    median_widths = []
    colors_mean = []
    colors_median = []
    labels = []
    mean_texts = []
    median_texts = []
    
    for i, (_, row) in enumerate(df.iterrows()):
        base_y = len(df) - i  # Reverse order to match original
        
        # Mean bar
        y_positions.extend([base_y + 0.2, base_y - 0.2])
        mean_width = (row['mean_latency'] / mean_max) * 100
        median_width = (row['median_latency'] / median_max) * 100
        
        mean_widths.extend([mean_width, 0])  # 0 for median placeholder
        median_widths.extend([0, median_width])  # 0 for mean placeholder
        
        # Colors based on performance
        mean_color = '#4caf50' if row['mean_latency'] == mean_min else '#e53935' if row['mean_latency'] == mean_max else '#2196f3'
        median_color = '#4caf50' if row['median_latency'] == median_min else '#e53935' if row['median_latency'] == median_max else '#2196f3'
        
        colors_mean.extend([mean_color, 'rgba(0,0,0,0)'])
        colors_median.extend(['rgba(0,0,0,0)', median_color])
        
        # Labels and text
        labels.extend([f"{row['config_label']} - Mean", f"{row['config_label']} - Median"])
        # mean_text = f"{row['mean_latency']:.0f} ms ({'Fastest' if row['mean_latency'] == mean_min else str(round(row['mean_latency']/mean_min, 2)) + '× slower'})"
        if row["mean_latency"] == mean_min:
            status = "Fastest"
        else:
            status = f"{row['mean_latency'] / mean_min:.2f}× slower"

        mean_text = f"{row['mean_latency']:.0f} ms ({status})"

        # median_text = f"{row['median_latency']:.0f} ms ({'Fastest' if row['median_latency'] == median_min else str(round(row["median_latency"]/median_min, 2)) + '× slower'})"
        if row["median_latency"] == median_min:
            status = "Fastest"
        else:
            status = f"{row['median_latency'] / median_min:.2f}× slower"

        median_text = f"{row['median_latency']:.0f} ms ({status})"
        mean_texts.extend([mean_text, ''])
        median_texts.extend(['', median_text])
    
    # Add mean bars
    fig.add_trace(go.Bar(
        name='Mean Latency',
        y=labels,
        x=mean_widths,
        orientation='h',
        marker_color=colors_mean,
        text=mean_texts,
        textposition='auto',
        showlegend=True
    ))
    
    # Add median bars  
    fig.add_trace(go.Bar(
        name='Median Latency',
        y=labels,
        x=median_widths,
        orientation='h',
        marker_color=colors_median,
        text=median_texts,
        textposition='auto',
        showlegend=True
    ))
    
    # Update layout to match original design
    fig.update_layout(
        title='Latency Comparison - Mean vs Median<br><sub>Bars scaled relative to slowest performance in each category</sub>',
        xaxis_title='Relative Performance (%)',
        yaxis_title='Test Configuration',
        height=400 + len(df) * 50,  # Dynamic height based on data
        barmode='overlay',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update axes
    fig.update_xaxes(range=[0, 105])  # Slight padding beyond 100%
    fig.update_yaxes(categoryorder='array', categoryarray=labels)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add the explanatory note
    st.markdown("""
    **Note:** Bars are scaled relative to the slowest performance in each category (mean vs median). 
    The fastest value is highlighted in green and labeled "Fastest". 
    The slowest is highlighted in red. All other values show how many times slower they are compared to the fastest baseline.
    """)

def create_latency_visualization_streamlit():
    """Streamlit version of latency visualization"""
    df = latency_measurement()
    render_latency_chart_streamlit(df)
    


# # Example usage with your JSON as DataFrame
# import pandas as pd

# data = [
#   {"test_type":"complete_pipeline","run_environment":"kaggle","query_count":"10","mean_latency":"5163.83","median_latency":"5001.22"},
#   {"test_type":"complete_pipeline","run_environment":"laptop","query_count":"10","mean_latency":"12870.59","median_latency":"11629.97"},
#   {"test_type":"vector_search","run_environment":"kaggle","query_count":"10","mean_latency":"3000.99","median_latency":"2872.14"},
#   {"test_type":"vector_search","run_environment":"laptop","query_count":"10","mean_latency":"4519.68","median_latency":"4162.68"}
# ]
# df = pd.DataFrame(data)




if __name__ == "__main__":
    # logger.info("Patent Metrics analysis")
    # create_bigquery_visualization()
    create_discoverability_visualization()
    # # Sample data for testing
    # efficiency_sample = [
    #     {"run_environment": "kaggle", "test_type": "partition_1_month", "avg_search_time_sec": "46.42", "avg_bytes_processed_gb": "2.0", "avg_results": "20.0"},
    #     {"run_environment": "kaggle", "test_type": "partition_3_months", "avg_search_time_sec": "47.52", "avg_bytes_processed_gb": "5.0", "avg_results": "20.0"},
    #     {"run_environment": "kaggle", "test_type": "partition_6_months", "avg_search_time_sec": "49.94", "avg_bytes_processed_gb": "11.0", "avg_results": "20.0"}
    # ]
    
    # reduction_sample = [
    #     {"run_environment": "kaggle", "test_type": "partition_1_month", "bytes_reduction_pct": "84.1", "time_reduction_pct": "12.3"},
    #     {"run_environment": "kaggle", "test_type": "partition_3_months", "bytes_reduction_pct": "52.5", "time_reduction_pct": "10.2"},
    #     {"run_environment": "kaggle", "test_type": "partition_6_months", "bytes_reduction_pct": "0.0", "time_reduction_pct": "5.6"}
    # ]
    
    # fig = create_bigquery_visualization(efficiency_sample, reduction_sample)
    # fig.show()