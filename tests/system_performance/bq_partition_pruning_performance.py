"""Measure performance of semantic search using BQ partition pruning"""
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

def create_partition_test_table():
    """Create table for partition pruning test results"""
    query = bq_queries.create_paritioned_pruning_table
    query = query.format(
        project_id=project_id,
        dataset_id=dataset_id,
        partition_pruned_table="partition_pruning_results"
    )
    
    try:
        job = pss_client.client.execute_sql_query(query)
        if job:
            job.result()
        logger.info("Partition pruning test table created successfully")
    except Exception as e:
        logger.error(f"Failed to create partition pruning table: {e}")

def execute_vector_search_with_filter(
        query_embeddings: List[float],
        date_start: str = None,
        date_end: str = None,
        top_k: int = 20,
        test_label: str = "",
        ) -> Dict[str, Any]:
    """Execute BigQuery AI Vector Search"""
    query = bq_queries.vector_search_query
    table_name = "patent_embeddings_local"

    
    filters = []
    params = [
            bigquery.ArrayQueryParameter("query_embeddings", "FLOAT64", query_embeddings)
    ]

    if date_start:
        filters = ["pub_date >= @date_start AND pub_date < @date_end"]
        params.append(bigquery.ScalarQueryParameter("date_start", "DATE", date_start))
        params.append(bigquery.ScalarQueryParameter("date_end", "DATE", date_end))
    
    

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
    # Add unique identifier to avoid cache hits between tests
    unique_query = f"{query} -- {test_label}_{int(time.time())}"


    start_time = time.time()
    try:
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        job = pss_client.client.execute_sql_query(unique_query, job_config=job_config)
        
        if job:
            job.result()  # Wait for completion
        
        results = job.to_dataframe() if job else pd.DataFrame()
        search_time = time.time() - start_time
        
        return {
            'results': results,
            'search_time': search_time,
            'job': job,
            'query': unique_query
        }
        
    except Exception as err:
        logger.error(f"Vector search with filter failed: {err}")
        raise

def test_partition_pruning_single_query(
    query_text: str, 
    run_environment: str = "laptop"
) -> List[Dict]:
    """Test partition pruning efficiency for a single query"""
    
    logger.info(f"\nTesting partition pruning for: '{query_text}'")
    
    # Generate embedding
    embedding_start = time.time()
    query_embeddings = generate_local_embeddings_for_query(model, query_text)
    embedding_time = time.time() - embedding_start
    
    test_results = []
    
    # Test scenarios with different date ranges
    test_scenarios = [
        {
            'label': 'full_scan',
            'date_start': None,
            'date_end': None,
            'date_range_months': 0,
            'description': 'Full table scan (all 2.9M patents)'
        },
        {
            'label': 'partition_6_months', 
            'date_start': '2024-01-01',
            'date_end': '2024-07-01',
            'date_range_months': 6,
            'description': '6-month window (2024 Jan-Jun)'
        },
        {
            'label': 'partition_3_months',
            'date_start': '2024-01-01',
            'date_end': '2024-04-01', 
            'date_range_months': 3,
            'description': '3-month window (2024 Jan-Mar)'
        },
        {
            'label': 'partition_1_month',
            'date_start': '2024-01-01',
            'date_end': '2024-02-01',
            'date_range_months': 1, 
            'description': '1-month window (2024 Jan only)'
        }
    ]
    
    for scenario in test_scenarios:
        logger.info(f"  Testing: {scenario['description']}")
        
        try:
            # Execute search with specific filter
            search_result = execute_vector_search_with_filter(
                query_embeddings,
                date_start=scenario['date_start'],
                date_end=scenario['date_end'],
                top_k=20,
                test_label=scenario['label']
            )
            
            results_df = search_result['results']
            job = search_result['job']
            
            # Calculate metrics
            total_time = embedding_time + search_result['search_time']
            
            # Safe metric extraction
            bytes_processed = getattr(job, 'total_bytes_processed', 0) or 0
            slot_millis = getattr(job, 'slot_millis', 0) or 0
            cache_hit = getattr(job, 'cache_hit', False) or False
            
            # Similarity metrics
            if not results_df.empty and 'cosine_score' in results_df.columns:
                min_similarity = results_df['cosine_score'].min()
                avg_similarity = results_df['cosine_score'].mean()
                max_similarity = results_df['cosine_score'].max()
            else:
                min_similarity = avg_similarity = max_similarity = 0.0
            
            # Log results
            logger.info(f"    Search time: {search_result['search_time']*1000:.0f}ms")
            logger.info(f"    Bytes processed: {bytes_processed:,}")
            logger.info(f"    Results found: {len(results_df)}")
            logger.info(f"    Cache hit: {cache_hit}")
            
            # Store result
            test_result = {
                'test_query': query_text,
                'test_type': scenario['label'],
                'run_environment': run_environment,
                'filter_clause': scenario['label'],
                'date_range_months': scenario['date_range_months'],
                'embedding_time_ms': embedding_time * 1000,
                'search_time_ms': search_result['search_time'] * 1000,
                'total_time_ms': total_time * 1000,
                'results_count': len(results_df),
                'bytes_processed': bytes_processed,
                'slot_millis': slot_millis,
                'cache_hit': cache_hit,
                'min_similarity': min_similarity,
                'avg_similarity': avg_similarity,
                'max_similarity': max_similarity,
            }
            
            test_results.append(test_result)
            
        except Exception as e:
            logger.error(f"    Failed: {e}")
            continue
        
        # Small delay between tests to avoid rate limiting
        time.sleep(2)
    
    return test_results

def run_comprehensive_partition_test(run_environment: str = "laptop"):
    """Run partition pruning tests across multiple queries"""
    
    logger.info("\n" + "="*60)
    logger.info("PARTITION PRUNING EFFICIENCY TEST")
    logger.info("="*60)
    
    # Create results table
    create_partition_test_table()
    
    # Test queries designed to hit different partitions
    partition_test_queries = [
        "machine learning neural network algorithms",
        "solar panel photovoltaic energy conversion", 
        "quantum computing error correction protocols",
        "battery management system electric vehicles",
        "medical diagnostic imaging systems"
    ]

    # query = "machine learning neural network algorithms"
    # test_partition_pruning_single_query(query, run_environment)
    all_results = []
    
    for query in partition_test_queries:
        query_results = test_partition_pruning_single_query(query, run_environment)
        if query_results:
            all_results.extend(query_results)
        
        # Pause between queries
        time.sleep(3)
    
    # Save results to BigQuery
    if all_results:
        save_partition_results_to_bigquery(all_results)
        analyze_partition_results(all_results)
    
    return all_results

def save_partition_results_to_bigquery(results_list: List[Dict]):
    """Save partition test results to BigQuery"""
    try:
        df = pd.DataFrame(results_list)
        table_id = f"{project_id}.{dataset_id}.partition_pruning_results"
        
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            create_disposition="CREATE_IF_NEEDED"
        )
        
        # Use your existing BigQuery client method
        job = pss_client.client._client.load_table_from_dataframe(
            df, table_id, job_config=job_config
        )
        
        logger.info(f"Successfully saved {len(df)} partition test results to BigQuery")
        
    except Exception as e:
        logger.error(f"Failed to save partition results: {e}")
        
        # Fallback to JSON
        timestamp = int(time.time())
        backup_file = f"partition_test_backup_{timestamp}.json"
        with open(backup_file, 'w') as f:
            json.dump(results_list, f, indent=2, default=str)
        logger.info(f"Results saved to backup file: {backup_file}")

def analyze_partition_results(results_list: List[Dict]):
    """Analyze partition pruning efficiency"""
    df = pd.DataFrame(results_list)
    
    logger.info("\n" + "="*60)
    logger.info("PARTITION PRUNING ANALYSIS")
    logger.info("="*60)
    
    # Group by test type
    summary = df.groupby('test_type').agg({
        'search_time_ms': ['mean', 'std'],
        'bytes_processed': ['mean', 'std'], 
        'results_count': 'mean',
        'cache_hit': 'sum'
    }).round(2)
    
    logger.info("\nPerformance by test type:")
    logger.info(summary)
    
    # Calculate efficiency gains
    full_scan = df[df['test_type'] == 'full_scan']
    pruned_6m = df[df['test_type'] == 'partition_6_months']
    
    if not full_scan.empty and not pruned_6m.empty:
        bytes_reduction = (1 - pruned_6m['bytes_processed'].mean() / full_scan['bytes_processed'].mean()) * 100
        time_reduction = (1 - pruned_6m['search_time_ms'].mean() / full_scan['search_time_ms'].mean()) * 100
        
        logger.info(f"\nPARTITION PRUNING EFFICIENCY (6-month vs full scan):")
        logger.info(f"  Bytes processed reduction: {bytes_reduction:.1f}%")
        logger.info(f"  Query time reduction: {time_reduction:.1f}%")
        logger.info(f"  Cost optimization achieved!")

# Run the test
if __name__ == "__main__":
    results = run_comprehensive_partition_test(run_environment="laptop")