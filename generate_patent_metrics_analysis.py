"""Compute various metrics to analyse latency, partition pruning and semantic search discoverability"""
import os
from typing import Any, List, Dict, Tuple
from loguru import logger
from src.sql_queries import bq_queries
from src.google import google_client
from dotenv import load_dotenv
load_dotenv()


try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Get configuration with fallbacks
if HAS_STREAMLIT and hasattr(st, 'secrets'):
    project_id = os.getenv("project_id") or st.secrets["google"]["project_id"]
    dataset_id = os.getenv("dataset_id") or st.secrets["google"]["dataset_id"]
    publication_table = os.getenv("publication_table") or st.secrets["google"]["publication_table"]
else:
    project_id = os.getenv("project_id")
    dataset_id = os.getenv("dataset_id") 
    publication_table = os.getenv("publication_table")

client = google_client.GoogleClient(
        project_id=project_id,
        credentials_path=os.getenv("service_account_path")
    )

def latency_measurement():
    """Measure latency of semantic search and explainability queries"""
    query = bq_queries.latency_measurement_query
    query = query.format(
        project_id=project_id,
        dataset_id=dataset_id,
        latency_table="latency_test_results"
    )

    # logger.info("Running query to measure latency metrics")
    return client.query_to_dataframe(query)

def partition_pruning_efficiency_measurement():
    """Measure efficiency of partition pruning"""
    query = bq_queries.efficiency_across_partition_query
    query = query.format(
        project_id=project_id,
        dataset_id=dataset_id,
        partition_pruned_table="partition_pruning_results"
    )

    # logger.info("Running query to measure partition pruning efficiency metrics")
    return client.query_to_dataframe(query)


def bytes_and_time_reduction_measurement():
    """Measure bytes and time reduction due to partition pruning"""
    query = bq_queries.calculate_bytes_and_time_reduction_query
    query = query.format(
        project_id=project_id,
        dataset_id=dataset_id,
        partition_pruned_table="partition_pruning_results"
    )

    # logger.info("Running query to measure bytes and time reduction metrics")
    return client.query_to_dataframe(query)

def discovery_rate_measurement():
    """Measure discovery rate due to semantic search"""
    query = bq_queries.discovery_rate_analysis_query
    query = query.format(
        project_id=project_id,
        dataset_id=dataset_id,
        search_comparison_table="search_comparison_test_results"
    )

    # logger.info("Running query to measure semantic search discoverability metrics")
    return client.query_to_dataframe(query)