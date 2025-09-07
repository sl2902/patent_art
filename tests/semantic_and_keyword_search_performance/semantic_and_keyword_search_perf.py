"""Comparison of Patents semantic search and keyword search performance"""
import sys
import os
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from src.sql_queries import bq_queries
from src.patent_search.semantic_search import(
    PatentSemanticSearch, 
    generate_local_embeddings_for_query
)
from google.cloud import bigquery
from loguru import logger
from typing import Any, List, Dict, Optional, Tuple
import json
from dotenv import load_dotenv
load_dotenv()


from run_patent_search_pipeline import (
    sanitize_input_query,
    run_semantic_search_pipeline,
    technology_selection,
    generate_random_patents
)

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Get configuration with fallbacks
if HAS_STREAMLIT and hasattr(st, 'secrets'):
    project_id = os.getenv("project_id") or st.secrets["google"]["project_id"]
    dataset_id = os.getenv("dataset_id") or st.secrets["google"]["dataset_id"]
    hf_token = os.getenv('hf_token') or st.secrets["hf"]["hf_token"]
    credentials_path = os.getenv("service_account_path")
else:
    project_id = os.getenv("project_id")
    dataset_id = os.getenv("dataset_id") 
    credentials_path = os.getenv("service_account_path")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


pss_client = PatentSemanticSearch(
    project_id=project_id,
    dataset_id=dataset_id,
    credentials_path=credentials_path
)

model_name =  "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name, token=hf_token)

def create_search_comparison_table():
    """Create search comparison results table in BigQuery"""
    query = bq_queries.create_search_comparison_ddl
    query = query.format(
        project_id=project_id,
        dataset_id=dataset_id,
        search_comparison_table="search_comparison_test_results"
    )
    
    try:
        job = pss_client.client.execute_sql_query(query)
        job.result()  # Wait for completion
        logger.info("Search comparison results table created successfully")
    except Exception as e:
        logger.error(f"Failed to create search comparison table: {e}")

def run_sanitization_step(query_text: str) -> Tuple[str, int]:
    """Sanitize the input query"""
    start_time = time.time()
    query_text, is_valid, error_msg = sanitize_input_query(query_text)
    if not is_valid:
        logger.warning(f"{error_msg}")
        logger.warning(f"Latency test failed. Change query terms and retry")
        return
    sanitization_time = time.time() - start_time

    return query_text, sanitization_time

def generate_embeddings(query_text: str) -> Tuple[List[float], int]:
    """Generate query embeddings"""
    start_time = time.time()
    embeddings = generate_local_embeddings_for_query(model, query_text)
    embedding_time = time.time() - start_time

    return embeddings, embedding_time

def execute_vector_search(
        query_embeddings: List[float],
        top_k: int = 30,
        test_label: str = 'semantic_search'
        ) -> Tuple[pd.DataFrame, float, bigquery.job.QueryJob]:
    """Execute BigQuery AI Vector Search"""
    query = bq_queries.vector_search_query
    table_name = "patent_embeddings_local"

    filters = ["pub_date >= @date_start AND pub_date < @date_end"]
    filter_clause = " AND ".join(filters)
    if filter_clause:
        filter_clause = "AND " + filter_clause
    else:
        filter_clause = ""
    
    query = query.format(
            project_id=project_id,
            dataset_id=dataset_id,
            table_name=table_name,
            top_k=top_k,
            filter_clause=filter_clause,
            options=json.dumps({"use_brute_force": True})
        )
    unique_query = f"{query} -- {test_label}_{int(time.time())}"


    params = [bigquery.ArrayQueryParameter("query_embeddings", "FLOAT64", query_embeddings)]
    params.append(bigquery.ScalarQueryParameter("date_start", "DATE", '2024-01-01'))
    params.append(bigquery.ScalarQueryParameter("date_end", "DATE", '2024-07-01'))

    start_time = time.time()
    try:
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        job = pss_client.client.execute_sql_query(unique_query,  job_config=job_config)
    except Exception as err:
        logger.error(f"Vector search query {unique_query} execution failed {err}")
        raise

    results = job.to_dataframe()
    search_time = time.time() - start_time
    
    return results, search_time, job

def execute_keyword_search(
        keyword: str,
        top_k: int = 30,
        test_label: str = 'keyword_search'
) -> Tuple[pd.DataFrame, float, bigquery.job.QueryJob]:
    """Execute keyword search query"""

    words_list =  [word for word in keyword.strip().split()]
    title_clause = " AND ". join([f"CONTAINS_SUBSTR(title_en, '{word}')" for word in words_list])
    abstract_clause = " AND ". join([f"CONTAINS_SUBSTR(abstract_en, '{word}')" for word in words_list])


    keyword_query = f"""
    SELECT 
      publication_number,
      title_en
    FROM `{project_id}.{dataset_id}.patent_embeddings_local`
    WHERE (
      ({title_clause})
        OR
     ({abstract_clause})
     )
    AND pub_date >= '2024-01-01' AND pub_date < '2024-07-01'
    LIMIT 30
    """

    unique_query = f"{keyword_query} -- {test_label}_{int(time.time())}"

    start_time = time.time()
    try:
        job = pss_client.client.execute_sql_query(unique_query)
    except Exception as err:
        logger.error(f"Keyword search query {unique_query} execution failed {err}")
        raise

    results = job.to_dataframe()
    # logger.info(results)
    search_time = time.time() - start_time

    return results, search_time, job

def compare_semantic_vs_keyword_search(
        semantic_searches: pd.DataFrame,
        keyword_searches: pd.DataFrame,
) -> Dict[str, Any]:
    """Compare the results from th two search scenarios"""
    if not semantic_searches.empty:
        semantic_patents = set(semantic_searches['publication_number'])
    else:
        logger.warning(f"No results from semantic search")
        semantic_patents = set()
    
    if not keyword_searches.empty:
        keyword_patents = set(keyword_searches['publication_number'])
    else:
        logger.warning(f"No results from keyword search")
        keyword_patents = set()
    
    overlap = len(semantic_patents.intersection(keyword_patents))
    semantic_unique = len(semantic_patents - keyword_patents)
    keyword_unique = len(keyword_patents - semantic_patents)

    return {
        'semantic_results_count': len(semantic_searches) or 0,
        'keyword_results_count': len(keyword_searches) or 0,
        'overlap_count': overlap or 0,
        'semantic_unique_count': abs(semantic_unique) or 0,
        'keyword_unique_count': abs(keyword_unique) or 0,
        'semantic_discovery_rate': abs(semantic_unique) / len(semantic_patents) * 100 if len(semantic_patents) > 0 else 0
    }
    


def test_single_query(query_text: str, top_k: int = 30, run_environment: str = "laptop"):
    """Test end-to-end latency for a single query with explainability"""
    
    # Step 1: Run validation on input query
    logger.info(f"\nTesting query: '{query_text}'")
    query_text, sanitization_time = run_sanitization_step(query_text)
    logger.info(f"  Sanitization of query: {sanitization_time*1000:.1f}ms")
    
    # Step 2: Generate query embedding (CPU)
    query_embeddings, embedding_time = generate_embeddings(query_text)
    logger.info(f"  Embedding generation: {embedding_time*1000:.1f}ms")
    
    # Step 3: Execute vector search (BigQuery AI)
    try:
        semantic_results, search_time, job = execute_vector_search(query_embeddings, top_k)
        logger.info(f"  Vector search: {search_time*1000:.1f}ms")
        
        
        # Calculate total times
        vector_search_total = sanitization_time + embedding_time + search_time
        
        logger.info(f"  Vector search pipeline: {vector_search_total*1000:.1f}ms")

        
        base_semantic_metrics = {
            'query': query_text,
            'run_environment': run_environment,
            'embedding_time_ms': embedding_time * 1000,
            'search_time_ms': search_time * 1000,
            'total_bytes_processed': job.total_bytes_processed,
            }
      

        
        # Vector search only metrics
        vector_search_metrics = {
            **base_semantic_metrics,
            'test_type': 'vector_search',
            'total_time_ms': vector_search_total * 1000
        }
        
        
    except Exception as e:
        logger.error(f"  Error: {e}")
        return None
    
    # Step 4: Execute keyword search (BigQuery)
    try:
        keywod_results, search_time, job = execute_keyword_search(query_text, top_k)
        logger.info(f"  Keyword search: {search_time*1000:.1f}ms")
        
        
        # Calculate total times
        keyword_search_total = sanitization_time  + search_time
        
        logger.info(f"  Keyword search pipeline: {keyword_search_total*1000:.1f}ms")

        
        base_keyword_metrics = {
            'query': query_text,
            'run_environment': run_environment,
            'embedding_time_ms': None,
            'search_time_ms': search_time * 1000,
            'total_bytes_processed': job.total_bytes_processed,
            }
      

        
        # Keyword search only metrics
        keyword_search_metrics = {
            **base_keyword_metrics,
            'test_type': 'keyword_search',
            'total_time_ms': keyword_search_total * 1000
        }
        
        
    except Exception as e:
        logger.error(f"  Error: {e}")
        return None
    
    # Step 5 Compare keyword search with vector search
    comparison_results = compare_semantic_vs_keyword_search(semantic_results, keywod_results)
    
    vector_search_metrics["semantic_results_count"] = comparison_results.get("semantic_results_count")
    vector_search_metrics["overlap_count"] = comparison_results.get("overlap_count")
    vector_search_metrics["semantic_unique_count"] = comparison_results.get("semantic_unique_count")
    vector_search_metrics["semantic_discovery_rate"] = comparison_results.get("semantic_discovery_rate")

    keyword_search_metrics["keyword_results_count"] = comparison_results.get("keyword_results_count")
    keyword_search_metrics["overlap_count"] = comparison_results.get("overlap_count")
    keyword_search_metrics["keyword_unique_count"] = comparison_results.get("keyword_unique_count")
    keyword_search_metrics["semantic_discovery_rate"] = None

    return [vector_search_metrics, keyword_search_metrics]


def save_results_to_bigquery(results_list: list):
    """Save latency test results to BigQuery"""
    if not results_list:
        logger.warning("No results to save")
        return
    
    # Flatten the list (since each test returns 2 records)
    flattened_results = []
    for result in results_list:
        if result:
            if isinstance(result, list):
                flattened_results.extend(result)
            else:
                flattened_results.append(result)
    
    if not flattened_results:
        logger.warning("No valid results to save")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(flattened_results)

    
    try:
        # Upload to BigQuery
        table_id = f"{project_id}.{dataset_id}.search_comparison_test_results"
        
        # job_config = bigquery.LoadJobConfig(
        #     write_disposition="WRITE_APPEND",
        #     create_disposition="CREATE_IF_NEEDED"
        # )
        
        job = pss_client.client.upload_dataframe(
            df, table_id, if_exists="append"
        )
        # job.result()  # Wait for completion
        
        logger.info(f"Successfully saved {len(df)} test results to BigQuery table: {table_id}")
        
    except Exception as e:
        logger.error(f"Failed to save results to BigQuery: {e}")
        
        # Fallback: save to JSON
        timestamp = int(time.time())
        backup_file = f"search_comparison_test_backup_{timestamp}.json"
        with open(backup_file, 'w') as f:
            json.dump(flattened_results, f, indent=2, default=str)
        logger.info(f"Results saved to backup file: {backup_file}")

def run_comprehensive_latency_test(run_environment: str = "laptop"):
    """Run comprehensive search comparison testing across multiple queries"""
    logger.info("\n" + "="*60)
    logger.info("COMPREHENSIVE SEARCH COMPARISON TEST")
    logger.info(f"Environment: {run_environment}")
    logger.info("="*60)
    
    # Create table if it doesn't exist
    create_search_comparison_table()
    
    test_queries = [
            "quantum computing algorithms",
            "machine learning neural networks", 
            "renewable energy solar panels",
            "medical diagnostic imaging",
            "autonomous vehicle navigation",
            "blockchain cryptocurrency technology",
            "artificial intelligence natural language processing",
            "biotechnology genetic engineering",
            "quantum error correction topological qubits fault tolerance",
            "microfluidic devices lab on chip diagnostic biosensors"
    ]
    
    all_results = []
    
    for query in test_queries:
        result = test_single_query(query, run_environment=run_environment)
        if result:
            all_results.append(result)
        time.sleep(1)  # Brief pause between queries
    
    if all_results:
        # Save to BigQuery
        save_results_to_bigquery(all_results)
        
        # Still run analysis for immediate feedback
        analyze_results(all_results)
        return all_results
    else:
        logger.info("No successful test results to analyze")
        return []

def analyze_results(results_list):
    """Analyze search comparison results"""
    # Flatten results
    flattened_results = []
    for result in results_list:
        if result and isinstance(result, list):
            flattened_results.extend(result)
        elif result:
            flattened_results.append(result)
    
    df = pd.DataFrame(flattened_results)
    
    # Separate by search type
    semantic_df = df[df['test_type'] == 'vector_search']
    keyword_df = df[df['test_type'] == 'keyword_search']
    
    logger.info("\n" + "="*60)
    logger.info("SEARCH COMPARISON ANALYSIS")
    logger.info("="*60)
    
    if not semantic_df.empty:
        logger.info(f"\nSEMANTIC SEARCH PERFORMANCE:")
        logger.info(f"  Average query time: {semantic_df['total_time_ms'].mean():.0f}ms")
        logger.info(f"  Average results per query: {semantic_df['semantic_results_count'].mean():.1f}")
        logger.info(f"  Average unique discoveries: {semantic_df['semantic_unique_count'].mean():.1f}")
        logger.info(f"  Average discovery rate: {semantic_df['semantic_discovery_rate'].mean():.1f}%")
    
    if not keyword_df.empty:
        logger.info(f"\nKEYWORD SEARCH PERFORMANCE:")
        logger.info(f"  Average query time: {keyword_df['total_time_ms'].mean():.0f}ms")
        logger.info(f"  Average results per query: {keyword_df['keyword_results_count'].mean():.1f}")
    
    # Overall comparison
    if not semantic_df.empty and not keyword_df.empty:
        total_semantic_unique = semantic_df['semantic_unique_count'].sum()
        total_semantic_results = semantic_df['semantic_results_count'].sum()
        overall_discovery_rate = (total_semantic_unique / total_semantic_results * 100) if total_semantic_results > 0 else 0
        
        logger.info(f"\nOVERALL COMPARISON:")
        logger.info(f"  Total semantic results: {total_semantic_results}")
        logger.info(f"  Total keyword results: {keyword_df['keyword_results_count'].sum()}")
        logger.info(f"  Total overlap: {semantic_df['overlap_count'].sum()}")
        logger.info(f"  Total unique to semantic: {total_semantic_unique}")
        logger.info(f"  Overall discovery rate: {overall_discovery_rate:.1f}%")


def main():
    """Main function to run search comparison tests"""
    logger.info("Patent Semantic Search Vs Keyword Search Comaprison - Testing")
    logger.info("="*50)
    
    # Determine run environment (you can modify this)
    run_environment = "laptop"  # Change to "kaggle" when running on Kaggle

    # query_text = "Machine learning"
    # test_single_query(query_text)
    
    # Run comprehensive test
    results = run_comprehensive_latency_test(run_environment=run_environment)
    
    # Test concurrent performance (optional)
    # test_concurrent_performance()
    
    logger.info("Search comparison testing complete!")
    logger.info("Results saved to BigQuery for visualization")

if __name__ == "__main__":
    main()