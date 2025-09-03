"""Run the Patent Semantic Search pipeline for user queries or a list of patents"""
import os
import sys
import re
import string
import pandas as pd
import pandas_gbq
from typing import Any, Dict, List, Tuple, Optional, Union
from loguru import logger
from src.sql_queries import bq_queries
from src.patent_search.semantic_search import PatentSemanticSearch
import torch
from sentence_transformers import SentenceTransformer
import gc
from tqdm import tqdm
from collections import Counter
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

def sanitize_input_query(query_text: str) -> Tuple[str, bool, str]:
    """Sanitize user input for patent search"""
    if not query_text:
        return "", False, "Query cannot be empty"
    
    query_text = query_text.strip()
    words = query_text.split()

    if len(query_text) < 2:
        return "", False, "Query too short (minimum 3 characters)"
    
    if len(query_text) > 500:
        return "", False, "Query too long (maximum 500 characters)"
    
    if len(words) < 2:
        return "", False, "Query has 1 word"
    
    for word in words:
        if len(word) > 4:
            char_counts = Counter(word)
            # If any character appears more than 60% of the time in a word
            max_char_ratio = max(char_counts.values()) / len(word)
            if max_char_ratio > 0.6:
                return "", False, "Please enter meaningful words (avoid excessive repetition)"
    
    keyboard_patterns = ['qwerty', 'asdf', 'zxcv', 'qazwsx', 'plmokn', 'abcd', '123']
    query_lower = query_text.lower().replace(' ', '')
    if any(pattern in query_lower for pattern in keyboard_patterns):
        return "", False, "Please enter a meaningful technology description"

    special_char_ratio = sum(1 for c in query_text if c in string.punctuation) / len(query_text)
    if special_char_ratio > 0.1:
        return "", False, "Too many special characters"
    
    if len(words) > 2 and (len(set(words)) / len(words)) < 0.3:
        logger.info(set(words) / len(words))
        return "", False, "Too much repetition in query"
    
    # gibberish
    if re.search(r'([a-z])\1{4,}', query_text.lower()):
        return "", False, "Please enter meaningful words"
    
    # meaningful content; this may or may not click
    if not re.search(r'[a-zA-Z]{2,}', query_text):
        return "", False, "Query must contain meaningful words"
    
    query_text = re.sub(r'\s+', ' ', query_text)
    # keep relevant characters for patent searches
    query_text = re.sub(r'[^\w\s\-\(\)\/\&\.\!\?\+\%\°\:\;]', '', query_text)
    
    return query_text.strip(), True, ""

def technology_selection(key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    A list of technologies present in 
    the dataset based on frequency of publication count. The
    purpose is provide users with examples to test the patent
    search feature on Streamlit
    """
    tech_categories = {
            "Computing & Software": {
                "description": "Data processing, software systems, computing methods",
                "cpc_codes": ["G06F", "G06T", "G06Q"],
                "sample_queries": [
                    "data management system",
                    "image processing algorithm", 
                    "business process automation"
                ]
            },
            "Medical & Healthcare": {
                "description": "Medical devices, pharmaceuticals, treatments",
                "cpc_codes": ["A61K", "A61B", "A61P", "A61M"],
                "sample_queries": [
                    "medical diagnostic device",
                    "surgical instrument design",
                    "pharmaceutical composition"
                ]
            },
            "Electronics & Semiconductors": {
                "description": "Displays, chips, electronic devices",
                "cpc_codes": ["H01L", "H10D"],
                "sample_queries": [
                    "semiconductor device structure",
                    "display panel technology",
                    "transistor design"
                ]
            },
            "Telecommunications": {
                "description": "Networks, wireless communication, data transmission",
                "cpc_codes": ["H04L", "H04W", "H04N"],
                "sample_queries": [
                    "wireless communication protocol",
                    "network security method",
                    "video transmission system"
                ]
            },
            "Energy Storage & Batteries": {
                "description": "Battery technology, energy systems",
                "cpc_codes": ["H01M"],
                "sample_queries": [
                    "lithium battery design",
                    "energy storage system",
                    "battery management circuit"
                ]
            },
            "Chemistry & Materials": {
                "description": "Chemical compounds, materials science",
                "cpc_codes": ["C07K", "C12N", "C07D", "B32B"],
                "sample_queries": [
                    "polymer material composition",
                    "chemical synthesis method",
                    "composite material structure"
                ]
            },
            "Testing & Measurement": {
                "description": "Sensors, measurement devices, testing equipment", 
                "cpc_codes": ["G01N", "G02B"],
                "sample_queries": [
                    "sensor measurement system",
                    "optical detection method",
                    "testing apparatus design"
                ]
            },
            "Separation & Filtering": {
                "description": "Filtration, purification, separation processes",
                "cpc_codes": ["B01D"],
                "sample_queries": [
                    "filtration system design",
                    "purification process method",
                    "separation technology"
                ]
            }
        }
    if key in tech_categories:
        return tech_categories.get(key)
    return tech_categories


def run_semantic_search_pipeline(
        start_date: str, 
        end_date: str, 
        query_text: str = None,
        patent_ids: List[str] = None,
        countries: List[str] = None,
        top_k: int = 1
    ) -> Optional[pd.DataFrame]:

    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    embedding_table_name = 'patent_embeddings_local'


    logger.info("Running Patent semantic search in BigQuery")
    candidate_df = pss_client.semantic_search_with_bq(
        model,
        embedding_table_name,
        query_text=query_text,
        query_patent_numbers=patent_ids,
        countries=countries,
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
    if not candidate_df.empty:
        semantic_matches = pss_client.semantic_search_with_explanability(
            model, 
            candidate_df,
            query_text=query_text,
            query_patents=patent_ids
        )
        return semantic_matches
    else:
        if patent_ids:
            logger.warning(f"Semantic search did not return any candidate results for query patents - {patent_ids}")
        else:
            logger.warning(f"Semantic search did not return any candidate results for query text - {query_text}")
    
    return pd.DataFrame()
                         
    # pd.set_option("display.max_colwidth", None)

if __name__ == "__main__":
    # test patents
    patent_ids = ['TW-M650298-U', 'CN-117475991-A', 'CN-113053411-B']
    start_date = "2024-01-01"
    end_date = "2024-02-01"
    run_semantic_search_pipeline(start_date, end_date, patent_ids=patent_ids)

    query_text = "ssssss ekwpkwp"
    print(sanitize_input_query(query_text))

