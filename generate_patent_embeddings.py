"""Generate Embeddings for the patents dataset"""
import os
import sys
import pandas as pd
import pandas_gbq
from typing import Any, Dict, List, Optional
from loguru import logger
from src.sql_queries import bq_queries
from src.google import google_client
import torch
from sentence_transformers import SentenceTransformer
import gc
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

project_id = os.getenv("project_id")
dataset_id = os.getenv("dataset_id")

client = google_client.GoogleClient(
        project_id=os.getenv("project_id"),
        credentials_path=os.getenv("service_account_path")
    )


def extract_patents(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch patents for a given date range"""
    qry = bq_queries.extract_qry
    qry = qry.format(
        project_id=project_id,
        dataset_id=dataset_id,
        start_date=start_date,
        end_date=end_date)
    
    return client.query_to_dataframe(qry)


def create_embedding_table(table_name: str = "patent_embeddings_local"):
    """Create embedding table for patents"""
    qry = bq_queries.create_embedding_ddl
    qry = qry.format(
        project_id=project_id,
        dataset_id=dataset_id,
        table_name=table_name
    )

    try:
        client._client.query(qry)
    except Exception as err:
        logger.error(f"Create table {table_name} failed {err}")
        raise

    print(f"Created table: {table_name}")

def upload_to_bq(
        df: pd.DataFrame, 
        table_name: str, 
        chunk_size: int, 
        if_exists: str="replace"
    ) -> None:
    """Upload the patent embeddings DataFrame to BigQuery"""
    # upload_df = df[['publication_number', 'country_code', 'pub_date', 
    #                    'title_en', 'abstract_en', 'combined_text', 'text_embedding']]
    
    try:
        # df.to_gbq(
        # table_name,
        # if_exists=if_exists,
        # chunksize=chunk_size
        # )
        pandas_gbq.to_gbq(
            df,
            table_name,
            project_id=project_id,
            # chunksize=chunk_size,
            if_exists=if_exists
        )
    except Exception as err:
        logger.error(f"Upload to BigQuery for {table_name} failed {err}")
        raise

def generate_local_embeddings_in_batches(
    model:  SentenceTransformer,
    df_patents: pd.DataFrame,
    num_batches: int,
    device: str = "cpu",
    embedding_batch_size: int = 1024,
    batch_num: int = 0,
    chunk_size: int = 1024,
):
    """Generate embeddings in batches to avoid out of memory errors"""
    embeddings = model.encode(
            df_patents["combined_text"].tolist(),
            batch_size=embedding_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    df_batch_with_embeddings = df_patents.copy()
    df_batch_with_embeddings["text_embedding"] = embeddings.tolist()

    table_name = f'{project_id}.{dataset_id}.patent_embeddings_local'
    client.upload_dataframe(df_batch_with_embeddings, table_name, chunk_size)

    del embeddings, df_batch_with_embeddings
    torch.cuda.empty_cache()
    _ = gc.collect()
    logger.info(f"Completed batch {batch_num + 1} of {num_batches}")

def get_processed_count(start_date: str, end_date: str) -> int:
    """Keeps track of how many batches were processed"""
    qry = f"""
        SELECT
            COUNT(*) as processed_count
        FROM `{project_id}.{dataset_id}.patent_embeddings_local`
        WHERE pub_date BETWEEN '{start_date}' AND '{end_date}'
    """
    result = client.query_to_dataframe(qry)
    return result["processed_count"].iloc[0]

def run_embedding_pipeline(
        start_date: str, 
        end_date: str, 
        batch_size: int, 
        model_id: str, 
        device: str
    ):

    logger.info("Fetching processed count from embedding table")
    num_processed = get_processed_count(start_date, end_date)
    logger.info(f"Number of rows processed {num_processed}")
    start_index = num_processed
    batch_size = batch_size

    logger.info(f"Fetch patents from {start_date} to {end_date}")
    df = extract_patents(start_date, end_date)
    logger.info(f"Number of patents fetched {df.shape[0]}")
    num_batches = len(df) // batch_size

    logger.info(f"Number of batches to process {num_batches}")

    logger.info("Starting embedding generation using sentence transformer")
    for i in range(start_index, num_batches, batch_size):
        batch = df[i: i + batch_size]
        generate_local_embeddings_in_batches(model_id, batch, num_batches, device=device, batch_num = i // batch_size)

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
        