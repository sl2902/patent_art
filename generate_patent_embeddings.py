"""Pipeline for eenerating batch Embeddings for the patents dataset"""
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
import math
from datetime import date, datetime, timedelta
from argparse import ArgumentParser
from dotenv import load_dotenv
load_dotenv()

project_id = os.getenv("project_id")
dataset_id = os.getenv("dataset_id")
credentials_path = os.getenv("service_account_path")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        embedding_table: str,
        batch_size: int, 
        model_id: str, 
        device: str,
        is_delete: bool
    ):

    source_table = os.getenv("publication_table")
    # embedding_table = 'patents_embedding_local'
    vector_index = 'patent_semantic_index'
    embedding_col_name = 'text_embedding'

    logger.info(f"Create embedding table {embedding_table}")
    pss_client.create_embeddings_table(embedding_table)

    logger.info(f"Create BigQuery vector index {vector_index} on field {embedding_col_name}")
    pss_client.create_vector_index(vector_index, embedding_table, embedding_col_name)

    if is_delete:
        logger.info("Delete rows from embeding table")
        job = pss_client.delete_embeddings_table(embedding_table, start_date, end_date)
        logger.info(f"Number of rows deleted between {start_date} and {end_date} is {job.num_dml_affected_rows}")


    logger.info("Fetching processed count from embedding table")
    track_processing_count_table = embedding_table

    num_processed = get_processed_count(track_processing_count_table, start_date, end_date)
    logger.info(f"Number of rows processed {num_processed}")
    start_index = num_processed
    batch_size = batch_size

    logger.info(f"Fetch patents from {start_date} to {end_date}")
    df = extract_patents(source_table, start_date, end_date)
    if df.empty or len(df) == 0:
        logger.warning(f"No patents to process for date range between  {start_date} and {end_date}")
        return None
    total_records = len(df)
    logger.info(f"Number of patents fetched {df.shape[0]}")

    if num_processed >= total_records:
        logger.warning(f"The start_index {num_processed} appears to be greater than or equal to the number of patents {total_records} to process")
        logger.warning(f"Make sure this is a new batch. Otherwise, delete the batch for the given date ranges and proceed.")
        return None
    
    num_batches = math.ceil(total_records / batch_size)

    logger.info(f"Number of batches to process {num_batches}")

    logger.info("Starting embedding generation using sentence transformer")
    for b_num, idx in enumerate(range(start_index, total_records, batch_size), start=1):
        batch = df[idx: idx + batch_size]
        pss_client.generate_local_batch_embeddings(
            model_id, 
            embedding_table,
            batch, 
            num_batches, 
            device=device, 
            batch_num= b_num,
        )

if __name__ == "__main__":

    parser = ArgumentParser(description="Generate Patent embeddings using SentenceTransformer")
    parser.add_argument(
        '--date-start',
        required=True,
        type=str,
        help="Enter start date in format yyyy-mm-dd. Start date cannot be prior to year 2017 or greater than 2025"
    )
    parser.add_argument(
        '--date-end',
        required=True,
        type=str,
        help="Enter end date, inclusive, in format yyyy-mm-dd. Start date cannot be prior to year 2017 or greater than 2025. Should be after date-start"
    )
    parser.add_argument(
        '--embedding-table',
        type=str,
        help="Enter name of the embedding table to store the embeddings. Check schema in file src/sql_queries/bq_queries.py - create_embedding_ddl",
        default=os.getenv("embedding_table")
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help="Enter embedding batch size. A valid integer. Eg., 1000, 2000. Maximum value is 5000",
        default=os.getenv("embedding_batch_size"),
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help="Whether you want to delete the existing embedding table for the specified date range",
    )
    args = parser.parse_args()

    try:
        start_date = datetime.strptime(args.date_start, "%Y-%m-%d")
    except ValueError as err:
        logger.error(f"Invalid date_start parameter. Format should be yyyy-mm-dd")
        raise

    try:
        end_date = datetime.strptime(args.date_end, "%Y-%m-%d")
    except ValueError as err:
        logger.error(f"Invalid date_end parameter. Format should be yyyy-mm-dd")
        raise

    if start_date.date() < date(2017, 1, 1) or start_date > datetime.now():
        logger.warning(f"date_start value outside valid range. Valid range is 2017-01 to 2025-02")
        raise

    if end_date.date() < date(2017, 1, 1) or end_date > datetime.now():
        logger.warning(f"date_end value outside valid range. Valid range is 2017-01 to 2025-02")
        raise

    if start_date > end_date:
        logger.warning(f"date_start value greater than date_end.")
        raise

    

    if args.batch_size < 1000 or args.batch_size > 5000:
        logger.warning(f"Batch size falls outside valid range [1000-5000]. Using default")
        batch_size = os.getenv("embedding_batch_size")
    else:
        batch_size = args.batch_size
    
    if not args.embedding_table:
        embedding_table = os.getenv("embedding_table")
    else:
        embedding_table = args.embedding_table
    
    is_delete = False
    if args.delete:
        is_delete = args.delete
    

    start_date, end_date = str(start_date.date()), str(end_date.date())

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        device = 'cpu'
        logger.info("Using CPU")
    
    model_id = os.getenv("small_model_id")
    model = SentenceTransformer(model_id, device=device)

    run_embedding_pipeline(start_date, end_date, embedding_table, batch_size, model, device, is_delete)
        