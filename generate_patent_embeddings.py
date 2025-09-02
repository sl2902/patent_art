"""Generate Embeddings for the patents dataset"""
import os
import sys
import pandas as pd
import pandas_gbq
from typing import Any, Dict, List, Optional
from loguru import logger
from src.sql_queries import bq_queries
# from src.google import google_client
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


# client = google_client.GoogleClient(
#         project_id=os.getenv("project_id"),
#         credentials_path=os.getenv("service_account_path")
#     )

pss_client = PatentSemanticSearch(
    project_id=project_id,
    dataset_id=dataset_id,
    credentials_path=credentials_path
)


def extract_patents(source_table: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch patents for a given date range"""
    qry = bq_queries.extract_qry
    qry = qry.format(
        project_id=project_id,
        dataset_id=dataset_id,
        table_name=source_table,
        start_date=start_date,
        end_date=end_date)
    
    return pss_client.client.query_to_dataframe(qry)


def get_processed_count(table_name: str, start_date: str, end_date: str) -> int:
    """Keeps track of how many batches were processed"""
    get_processed_qry = bq_queries.track_embedding_count.format(
        project_id=project_id,
        dataset_id=dataset_id,
        table_name=table_name,
        start_date=start_date,
        end_date=end_date
    )
    result = pss_client.client.query_to_dataframe(get_processed_qry)
    return result["processed_count"].iloc[0]

def run_embedding_pipeline(
        start_date: str, 
        end_date: str, 
        batch_size: int, 
        model_id: str, 
        device: str
    ):

    source_table = 'patents_2017_2025_en'
    embedding_table = 'patents_embedding_local'
    vector_index = 'patent_semantic_index'
    embedding_col_name = 'text_embedding'

    logger.info(f"Create embedding table {embedding_table}")
    pss_client.create_embeddings_table(embedding_table)

    logger.info(f"Create BigQuery vector index {vector_index} on field {embedding_col_name}")
    pss_client.create_vector_index(vector_index, embedding_table, embedding_col_name)

    logger.info("Fetching processed count from embedding table")
    track_processing_count_table = 'patent_embeddings_local'

    num_processed = get_processed_count(track_processing_count_table, start_date, end_date)
    logger.info(f"Number of rows processed {num_processed}")
    start_index = num_processed
    batch_size = batch_size

    logger.info(f"Fetch patents from {start_date} to {end_date}")
    df = extract_patents(source_table, start_date, end_date)
    logger.info(f"Number of patents fetched {df.shape[0]}")
    num_batches = len(df) // batch_size

    logger.info(f"Number of batches to process {num_batches}")

    logger.info("Starting embedding generation using sentence transformer")
    for i in range(start_index, num_batches, batch_size):
        batch = df[i: i + batch_size]
        pss_client.generate_local_batch_embeddings(
            model_id, 
            embedding_table,
            batch, 
            num_batches, 
            device=device, 
            batch_num = i // batch_size
        )

if __name__ == "__main__":

    # could be passed as arguments from the cli
    batch_size = 1024
    start_date, end_date = "2024-01-01", "2024-06-30"

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        device = 'cpu'
        logger.info("Using CPU")
    
    model_id = 'all-MiniLM-L6-v2' # 384 dims  'all-mpnet-base-v2' # 768 dims
    model = SentenceTransformer(model_id, device=device)

    run_embedding_pipeline(start_date, end_date, batch_size, model_id, device)
        