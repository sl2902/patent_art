"""Run the Patent Semantic Search pipeline for user queries or a list of patents"""
import os
import sys
import pandas as pd
import pandas_gbq
from typing import Any, Dict, List, Optional
from loguru import logger
from src.sql_queries import bq_queries
from src.patent_search.semantic_search import PatentSemanticSearch
import torch
from sentence_transformers import SentenceTransformer
import gc
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

project_id = os.getenv("project_id")
dataset_id = os.getenv("dataset_id")
credentials_path = os.getenv("service_account_path")


pss_client = PatentSemanticSearch(
    project_id=project_id,
    dataset_id=dataset_id,
    credentials_path=credentials_path
)

def run_user_patents_query(table_name: str, patent_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """Run the query for a list of user provided patents"""
    qry = bq_queries.test_patents_query
    qry = qry.format(
        project_id=project_id,
        dataset_id=dataset_id,
        table_name=table_name
    )
    try:
        results = pss_client.client.query_to_dataframe(qry)
    except Exception as err:
        logger.error(f"Failed to run user patents query - {err}")
        raise
    return results

def concat_patent_query_responses(df: pd.DataFrame) -> str:
    """Concatenate the combined_text field for multi patent search queries"""
    return " ".join([row.combined_text for row in df.itertuples(index=False)])

def run_explainability():
    pass

def run_semantic_search_pipeline(
        start_date: str, 
        end_date: str, 
        query_text: str = None,
        patent_ids: List[str] = None,
        top_k: int = 1
    ):

    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    embedding_table_name = 'patent_embeddings_local'


    logger.info("Running Patent semantic search in BigQuery")
    candidate_df = pss_client.semantic_search_with_bq(
        model,
        embedding_table_name,
        query_text=query_text,
        query_patent_numbers=patent_ids,
        date_start=start_date,
        date_end=end_date,
        top_k=top_k
    )

    # this query is to answer the "why" the candidate_df results match with the
    # user's query. In short - explainability
    if patent_ids:
        logger.info("Running explainability for user patents")
        user_patents = run_user_patents_query(embedding_table_name, patent_ids=patent_ids)
        query_text = concat_patent_query_responses(user_patents)
    else:
        logger.info("Running explainability for user text query")

    logger.info("Running explainability steps")
    semantic_matches = pss_client.semantic_search_with_explanability(
        model, 
        candidate_df,
        query_text=query_text,
        query_patents=patent_ids
    )
    # pd.set_option("display.max_colwidth", None)
    return semantic_matches

if __name__ == "__main__":
    # test patents
    patent_ids = ['TW-M650298-U', 'CN-117475991-A', 'CN-113053411-B']
    start_date = "2024-01-01"
    end_date = "2024-02-01"
    run_semantic_search_pipeline(start_date, end_date, patent_ids=patent_ids)

