""""Produce insights from the Patents analysis"""
import os
import pandas as pd
import pandas_gbq
from typing import Any, Dict, List, Optional
from loguru import logger
import streamlit as st
from src.sql_queries import bq_queries
from src.google import google_client
from dotenv import load_dotenv
load_dotenv()

project_id = os.getenv("project_id") or st.secrets["google"]["project_id"]
dataset_id = os.getenv("dataset_id") or st.secrets["google"]["dataset_id"]
publication_table = os.getenv("publication_table") or st.secrets["google"]["publication_table"]

client = google_client.GoogleClient(
        project_id=os.getenv("project_id"),
        credentials_path=os.getenv("service_account_path")
    )

def dataset_size_table():
    """Summary statistics of the dataset"""
    qry = bq_queries.dataset_size_qry
    qry = qry.format(
        project_id=project_id,
        dataset_id=dataset_id,
        publication_table=publication_table,
    )
    
    logger.info("Running query for summary statistics of the dataset")
    return client.query_to_dataframe(qry)

def country_wise_breakdown(top_n: int = 10):
    """Country wise patent publication - Bar chart"""
    qry = bq_queries.country_wise_publications
    qry = qry.format(
        project_id=project_id,
        dataset_id=dataset_id,
        publication_table=publication_table,
        top_n=top_n,
    )
    
    logger.info("Running query for country-wise publications")
    return client.query_to_dataframe(qry)

def top_country_each_month():
    """Top country each month - Timeline chart"""
    qry = bq_queries.top_country_each_month
    qry = qry.format(
        project_id=project_id,
        dataset_id=dataset_id,
        publication_table=publication_table,
    )
    
    logger.info("Running query for top country each month")
    return client.query_to_dataframe(qry)

def yoy_lang_growth_rate():
    """YoY english language publication growth rate - Line chart"""
    qry = bq_queries.yoy_eng_lang_publications_gr
    qry = qry.format(
        project_id=project_id,
        dataset_id=dataset_id,
        publication_table=publication_table,
    )
    
    logger.info("Running query for YoY English language publication growth rate")
    return client.query_to_dataframe(qry)

def yoy_country_growth_rate(top_n: int = 10):
    """YoY top n country publication growth rate - MultiLine chart"""
    qry = bq_queries.yoy_top_n_countries
    qry = qry.format(
        project_id=project_id,
        dataset_id=dataset_id,
        publication_table=publication_table,
        top_n=top_n,
    )
    
    logger.info("Running query for YoY English language publication growth rate")
    return client.query_to_dataframe(qry)

def citations_top_countries(top_n: int = 10):
    """Citation patterns by top countries - Table chart"""
    qry = bq_queries.citation_top_n_countries
    qry = qry.format(
        project_id=project_id,
        dataset_id=dataset_id,
        publication_table=publication_table,
        top_n=top_n,
    )
    
    logger.info("Running query for citation patterns for top countries")
    return client.query_to_dataframe(qry)

def top_cpc(top_n: int = 10):
    """Top n CPC - Bar chart"""
    qry = bq_queries.top_n_cpc
    qry = qry.format(
        project_id=project_id,
        dataset_id=dataset_id,
        publication_table=publication_table,
        top_n=top_n,
    )
    
    logger.info("Running query for top n CPCs")
    return client.query_to_dataframe(qry)

def tech_area_cpc(top_n: int = 10):
    """Technology area analysis by CPC main class - Bar chart"""
    qry = bq_queries.tech_area_cpc_class
    qry = qry.format(
        project_id=project_id,
        dataset_id=dataset_id,
        publication_table=publication_table,
        top_n=top_n,
    )
    
    logger.info("Running query for Technology area analyis by CPC main class")
    return client.query_to_dataframe(qry)

def patent_flow():
    """Patent publication flow - Sankey chart"""
    qry = bq_queries.patent_flow
    qry = qry.format(
        project_id=project_id,
        dataset_id=dataset_id,
        publication_table=publication_table,
    )
    
    logger.info("Running query for Patent publication flow")
    return client.query_to_dataframe(qry)

# if __name__ == "__main__":
#     logger.info(dataset_size_table())