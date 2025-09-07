"""Measure performance of Patent Semantic Search with and without(brute force) BiqQuery Vector Index"""
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
from datetime import datetime
import pyarrow as pa
from dotenv import load_dotenv
load_dotenv()

from run_patent_search_pipeline import sanitize_input_query

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Configuration
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

model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name, token=hf_token)

def create_vector_index_test_table():
    """Create table for vector index performance results"""
    query = bq_queries.create_vector_index_table
    query = query.format(
        project_id=project_id,
        dataset_id=dataset_id,
        vector_index_table="vector_index_performance_results"
    )
    
    try:
        job = pss_client.client.execute_sql_query(query)
        if job:
            job.result()
        logger.info("Vector index performance test table created successfully")
    except Exception as e:
        logger.error(f"Failed to create vector index test table: {e}")

def check_vector_index_exists(table_name: str = "patent_embeddings_local") -> bool:
    """Check if vector index exists on the embeddings table"""
    check_query = f"""
    SELECT COUNT(*) as index_count
    FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.VECTOR_INDEXES`
    WHERE table_name = '{table_name}'
    """
    
    try:
        job = pss_client.client.execute_sql_query(check_query)
        if job:
            job.result()
            results = job.to_dataframe()
            index_exists = results['index_count'].iloc[0] > 0
            logger.info(f"Vector index exists: {index_exists}")
            return index_exists
    except Exception as e:
        logger.error(f"Failed to check vector index: {e}")
        return False

def create_vector_index(
    table_name: str = "patent_embeddings_local",
    index_name: str = "patent_semantic_index",
    num_lists: int = 1000
) -> float:
    """Create vector index on embeddings table"""
    
    create_index_query = bq_queries.create_vector_index
    create_index_query = create_index_query.format(
        project_id=project_id,
        dataset_id=dataset_id,
        table_name=table_name,
        vector_index=index_name,
        embedding_column="text_embedding",
        num_lists=num_lists
    )
    
    logger.info(f"Creating vector index '{index_name}' with {num_lists} lists...")
    
    start_time = time.time()
    try:
        job = pss_client.client.execute_sql_query(create_index_query)
        if job:
            job.result()  # Wait for index creation to complete
        
        creation_time = time.time() - start_time
        logger.info(f"Vector index created successfully in {creation_time:.1f} seconds")
        return creation_time
        
    except Exception as e:
        logger.error(f"Failed to create vector index: {e}")
        raise

def extract_vector_search_statistics(job):
    """Extract vector search statistics from BigQuery job"""
    content = None
    if hasattr(job, '_properties'):
        props = job._properties
        if isinstance(props, dict):
            for key, value in props.items():
                if key == 'statistics' and isinstance(value, dict):
                    content = value
                    # logger.info(f"Statistics content: {json.dumps(value, indent=2)}")
    try:
        # Access the job statistics
        if 'query' in content and 'vectorSearchStatistics' in content.get('query'):
            vector_stats = content.get('query').get('vectorSearchStatistics')
            
            return {
                'index_usage_mode': getattr(vector_stats, 'indexUsageMode', None),
                'index_unused_reasons': [
                    {
                        'code': reason.get('code'),
                        'message': reason.get('message'),
                        'base_table': {
                            'project_id': reason.get('baseTable').get('projectId'),
                            'dataset_id': reason.get('baseTable').get('datasetId'), 
                            'table_id': reason.get('baseTable').get('tableId')
                        } if 'baseTable' in reason else None
                    }
                    for reason in vector_stats.get('indexUnusedReasons', [])
                ]
            }
        else:
            return None
            
    except Exception as e:
        logger.warning(f"Could not extract vector search statistics: {e}")
        return None

def complete_job_inspection(job):
    """Comprehensive inspection of job object"""
    logger.info("=== COMPLETE JOB INSPECTION ===")
    
    # Basic job info
    logger.info(f"Job ID: {job.job_id}")
    logger.info(f"Job type: {type(job)}")
    logger.info(f"Job state: {job.state}")
    
    # All attributes
    all_attrs = [attr for attr in dir(job) if not attr.startswith('_')]
    logger.info(f"All job attributes: {all_attrs}")
    
    # Check specific attributes
    for attr in ['statistics', 'query', '_properties']:
        if hasattr(job, attr):
            obj = getattr(job, attr)
            logger.info(f"{attr} type: {type(obj)}")
            if hasattr(obj, '__dict__'):
                logger.info(f"{attr} attributes: {list(obj.__dict__.keys())}")
    
    # Raw properties deep dive
    if hasattr(job, '_properties'):
        props = job._properties
        if isinstance(props, dict):
            for key, value in props.items():
                if key == 'statistics' and isinstance(value, dict):
                    logger.info(f"Statistics content: {json.dumps(value, indent=2)}")
    
    logger.info("=== END JOB INSPECTION ===")

def convert_to_bigquery_struct(index_unused_reasons_json_str):
    """Convert JSON string to BigQuery STRUCT format"""
    if not index_unused_reasons_json_str:
        return None
    
    try:
        # Parse the JSON string
        reasons_list = json.loads(index_unused_reasons_json_str)
        
        # Convert to BigQuery STRUCT format
        bigquery_structs = []
        for reason in reasons_list:
            struct_data = {
                'code': reason.get('code'),
                'message': reason.get('message'),
                'base_table': {
                    'project_id': reason.get('base_table', {}).get('project_id'),
                    'dataset_id': reason.get('base_table', {}).get('dataset_id'),
                    'table_id': reason.get('base_table', {}).get('table_id')
                }
            }
            bigquery_structs.append(struct_data)
        
        return bigquery_structs
        
    except json.JSONDecodeError:
        return None


def execute_brute_force_search(
        query_embeddings: List[float],
        date_start: str = None,
        date_end: str = None,
        top_k: int = 20,
        test_label: str = "",
) -> Dict[str, Any]:
    """Execute brute force vector search (exact results)"""
    
    query = bq_queries.vector_search_performance_query
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
            # complete_job_inspection(job) # Used to check whether vector stats keys are available or not 
        
        results = job.to_dataframe() if job else pd.DataFrame()
        search_time = time.time() - start_time

        vector_stats = extract_vector_search_statistics(job)
        
        if vector_stats:
            logger.info(f"    Index usage mode: {vector_stats['index_usage_mode']}")
            if vector_stats['index_unused_reasons']:
                for reason in vector_stats['index_unused_reasons']:
                    logger.info(f"    Index unused reason: {reason['code']} - {reason['message']}")
        
        return {
            'results': results,
            'query': unique_query,
            'search_time': search_time,
            'job': job,
            'index_usage_mode': vector_stats['index_usage_mode'] if vector_stats else None,
            'index_unused_reasons': convert_to_bigquery_struct(json.dumps(vector_stats['index_unused_reasons']))
        }
        
    except Exception as err:
        logger.error(f"Vector search with filter failed: {err}")
        raise

def execute_indexed_search(
    query_embeddings: List[float],
        date_start: str = None,
        date_end: str = None,
        top_k: int = 20,
        fraction_lists_searched: float = 0.01,
        test_label: str = "",
) -> Dict[str, Any]:
    """Execute vector search using index (approximate results)"""
    
    # Indexed search query
    query = bq_queries.vector_search_performance_query
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
        options=json.dumps({"fraction_lists_to_search": fraction_lists_searched})
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
        
        vector_stats = extract_vector_search_statistics(job)
        
        if vector_stats:
            logger.info(f"    Index usage mode: {vector_stats['index_usage_mode']}")
            if vector_stats['index_unused_reasons']:
                for reason in vector_stats['index_unused_reasons']:
                    logger.info(f"    Index unused reason: {reason['code']} - {reason['message']}")
        
        return {
            'results': results,
            'search_time': search_time,
            'job': job,
            'query': unique_query,
            'index_usage_mode': vector_stats['index_usage_mode'] if vector_stats else None,
            'index_unused_reasons': convert_to_bigquery_struct(json.dumps(vector_stats['index_unused_reasons']))
        }
        
    except Exception as err:
        logger.error(f"Vector search with filter failed: {err}")
        raise

def calculate_recall(brute_force_results: pd.DataFrame, indexed_results: pd.DataFrame) -> float:
    """Calculate recall of indexed search vs brute force ground truth"""
    if brute_force_results.empty or indexed_results.empty:
        return 0.0
    
    # Get patent numbers from both result sets
    brute_force_patents = set(brute_force_results['publication_number'].tolist())
    indexed_patents = set(indexed_results['publication_number'].tolist())
    
    # Calculate recall: |intersection| / |brute_force_results|
    intersection = brute_force_patents.intersection(indexed_patents)
    recall = len(intersection) / len(brute_force_patents) if len(brute_force_patents) > 0 else 0.0
    
    return recall

def test_vector_index_single_query(
    query_text: str,
    run_environment: str = "laptop"
) -> List[Dict]:
    """Test vector index performance for a single query"""
    
    logger.info(f"\nTesting vector index performance for: '{query_text}'")
    
    # Generate embedding
    embedding_start = time.time()
    query_embeddings = generate_local_embeddings_for_query(model, query_text)
    embedding_time = time.time() - embedding_start
    
    test_results = []
    
    # Test 1: Brute force search (ground truth)
    logger.info("  Running brute force search...")
    test_scenarios = [
        {
            'label': 'full_scan',
            'date_start': None,
            'date_end': None,
            'date_range_months': 0,
            'description': 'Full table scan (all 2.2M patents)'
        },
        # {
        #     'label': 'partition_6_months', 
        #     'date_start': '2024-01-01',
        #     'date_end': '2024-07-01',
        #     'date_range_months': 6,
        #     'description': '6-month window (2024 Jan-Jun)'
        # },
        # {
        #     'label': 'partition_3_months',
        #     'date_start': '2024-01-01',
        #     'date_end': '2024-04-01', 
        #     'date_range_months': 3,
        #     'description': '3-month window (2024 Jan-Mar)'
        # },
        # {
        #     'label': 'partition_1_month',
        #     'date_start': '2024-01-01',
        #     'date_end': '2024-02-01',
        #     'date_range_months': 1, 
        #     'description': '1-month window (2024 Jan only)'
        # }
    ]
    
    for scenario in test_scenarios:
        logger.info(f"  Testing: {scenario['description']}")
        
        try:
            brute_force_result = execute_brute_force_search(
                query_embeddings, 
                date_start=scenario["date_start"],
                date_end=scenario["date_end"],
                top_k=20, 
                test_label="brute_force"
            )
            
            brute_force_df = brute_force_result['results']
            brute_force_job = brute_force_result['job']
            brute_force_index_usage_mode = brute_force_result['index_usage_mode']
            brute_force_index_unused_reasons = brute_force_result['index_unused_reasons']

            
            # Extract metrics
            bf_bytes_processed = getattr(brute_force_job, 'total_bytes_processed', 0) or 0
            bf_slot_millis = getattr(brute_force_job, 'slot_millis', 0) or 0
            bf_cache_hit = getattr(brute_force_job, 'cache_hit', False) or False
            
            # Similarity metrics
            if not brute_force_df.empty and 'cosine_score' in brute_force_df.columns:
                bf_min_sim = brute_force_df['cosine_score'].min()
                bf_avg_sim = brute_force_df['cosine_score'].mean()
                bf_max_sim = brute_force_df['cosine_score'].max()
            else:
                bf_min_sim = bf_avg_sim = bf_max_sim = 0.0
            
            logger.info(f"    Brute force time: {brute_force_result['search_time']*1000:.0f}ms")
            logger.info(f"    Bytes processed: {bf_bytes_processed:,}")
            logger.info(f"    Results found: {len(brute_force_df)}")
            
            # Store brute force result
            brute_force_metrics = {
                'test_query': query_text,
                'search_method': 'brute_force',
                'run_environment': run_environment,
                'index_name': None,
                'index_usage_mode': brute_force_index_usage_mode,
                'index_unused_reasons': brute_force_index_unused_reasons,
                'ivf_num_lists': None,
                'fraction_lists_searched': None,
                'embedding_time_ms': embedding_time * 1000,
                'search_time_ms': brute_force_result['search_time'] * 1000,
                'index_creation_time_ms': None,
                'total_time_ms': (embedding_time + brute_force_result['search_time']) * 1000,
                'results_count': len(brute_force_df),
                'bytes_processed': bf_bytes_processed,
                'slot_millis': bf_slot_millis,
                'cache_hit': bf_cache_hit,
                'recall_vs_brute_force': 1.0,  # Brute force is 100% recall by definition
                'min_similarity': bf_min_sim,
                'avg_similarity': bf_avg_sim,
                'max_similarity': bf_max_sim
            }
            
            test_results.append(brute_force_metrics)
            
        except Exception as e:
            logger.error(f"    Brute force search failed: {e}")
            return test_results
    
    # Test 2: Check if index exists, create if needed
    index_exists = check_vector_index_exists()
    index_creation_time = 0.0
    
    if not index_exists:
        logger.info("  Creating vector index...")
        try:
            index_creation_time = create_vector_index(
                table_name="patent_embeddings_local",
                index_name="patent_semantic_index",
                num_lists=1000
            )
            
            # Store index creation metrics
            index_creation_metrics = {
                'test_query': query_text,
                'search_method': 'index_creation',
                'run_environment': run_environment,
                'index_name': 'patent_semantic_index',
                'index_usage_mode': None,
                'index_unused_reasons': None,
                'ivf_num_lists': 1000,
                'fraction_lists_searched': None,
                'embedding_time_ms': None,
                'search_time_ms': None,
                'index_creation_time_ms': index_creation_time * 1000,
                'total_time_ms': index_creation_time * 1000,
                'results_count': None,
                'bytes_processed': None,
                'slot_millis': None,
                'cache_hit': None,
                'recall_vs_brute_force': None,
                'min_similarity': None,
                'avg_similarity': None,
                'max_similarity': None
            }
            
            test_results.append(index_creation_metrics)
            
        except Exception as e:
            logger.error(f"    Failed to create index: {e}")
            return test_results
    
    # Test 3: Indexed search with different fraction_lists_to_search values
    fraction_values = [0.01, 0.05, 0.1]  # Test different speed/accuracy tradeoffs

    for scenario in test_scenarios:
        logger.info(f"  Testing: {scenario['description']}")
    
        for fraction in fraction_values:
            logger.info(f"  Running indexed search (fraction={fraction})...")
            try:
                indexed_result = execute_indexed_search(
                    query_embeddings,
                    date_start=scenario["date_start"],
                    date_end=scenario["date_end"],
                    top_k=20,
                    fraction_lists_searched=fraction,
                    test_label=f"indexed_{fraction}"
                )
                
                indexed_df = indexed_result['results']
                indexed_job = indexed_result['job']
                indexed_usage_mode = indexed_result['index_usage_mode']
                indexed_unused_reasons = indexed_result['index_unused_reasons']
                
                # Calculate recall vs brute force
                recall = calculate_recall(brute_force_df, indexed_df)
                
                # Extract metrics
                idx_bytes_processed = getattr(indexed_job, 'total_bytes_processed', 0) or 0
                idx_slot_millis = getattr(indexed_job, 'slot_millis', 0) or 0
                idx_cache_hit = getattr(indexed_job, 'cache_hit', False) or False
                
                # Similarity metrics
                if not indexed_df.empty and 'cosine_score' in indexed_df.columns:
                    idx_min_sim = indexed_df['cosine_score'].min()
                    idx_avg_sim = indexed_df['cosine_score'].mean()
                    idx_max_sim = indexed_df['cosine_score'].max()
                else:
                    idx_min_sim = idx_avg_sim = idx_max_sim = 0.0
                
                logger.info(f"    Indexed search time: {indexed_result['search_time']*1000:.0f}ms")
                logger.info(f"    Recall vs brute force: {recall:.3f}")
                logger.info(f"    Results found: {len(indexed_df)}")
                
                # Store indexed result
                indexed_metrics = {
                    'test_query': query_text,
                    'search_method': 'vector_index',
                    'run_environment': run_environment,
                    'index_name': 'patent_semantic_index',
                    'index_usage_mode': indexed_usage_mode,
                    'index_unused_reasons': indexed_unused_reasons,
                    'ivf_num_lists': 1000,
                    'fraction_lists_searched': fraction,
                    'embedding_time_ms': embedding_time * 1000,
                    'search_time_ms': indexed_result['search_time'] * 1000,
                    'index_creation_time_ms': index_creation_time * 1000 if index_creation_time > 0 else None,
                    'total_time_ms': (embedding_time + indexed_result['search_time']) * 1000,
                    'results_count': len(indexed_df),
                    'bytes_processed': idx_bytes_processed,
                    'slot_millis': idx_slot_millis,
                    'cache_hit': idx_cache_hit,
                    'recall_vs_brute_force': recall,
                    'min_similarity': idx_min_sim,
                    'avg_similarity': idx_avg_sim,
                    'max_similarity': idx_max_sim
                }
                
                test_results.append(indexed_metrics)
                
            except Exception as e:
                logger.error(f"    Indexed search failed (fraction={fraction}): {e}")
                continue
            
            # Small delay between tests
            time.sleep(2)
    
    return test_results

def run_comprehensive_vector_index_test(run_environment: str = "laptop"):
    """Run vector index tests across multiple queries"""
    
    logger.info("\n" + "="*60)
    logger.info("VECTOR INDEX PERFORMANCE TEST")
    logger.info("="*60)
    
    # Create results table
    create_vector_index_test_table()
    
    # Test queries
    index_test_queries = [
        "machine learning neural network algorithms",
        "quantum computing error correction", 
        "renewable energy solar cells",
        "medical diagnostic imaging",
        "autonomous vehicle navigation"
    ]
    # query = "medical diagnostic imaging"
    # results = test_vector_index_single_query(query)
    all_results = []
    
    for query in index_test_queries:
        query_results = test_vector_index_single_query(query, run_environment)
        if query_results:
            all_results.extend(query_results)
        
        # Pause between queries
        time.sleep(5)
    
    # Save results to BigQuery
    if all_results:
        save_vector_index_results_to_bigquery(all_results)
        analyze_vector_index_results(all_results)
    
    return all_results

def save_vector_index_results_to_bigquery(results_list: List[Dict]):
    """Save vector index test results to BigQuery"""
    try:
        df = pd.DataFrame(results_list)
        # df["index_unused_reasons"] = df["index_unused_reasons"].apply(json.dumps)
        # py_df = pa.Table.from_pandas(df)

        table_id = f"{project_id}.{dataset_id}.vector_index_performance_results"
        
        # job_config = bigquery.LoadJobConfig(
        #     write_disposition="WRITE_APPEND",
        #     create_disposition="CREATE_IF_NEEDED"
        # )
        
        # job = pss_client.client._client.load_table_from_dataframe(
        #     df, table_id, job_config=job_config
        # )
        job = pss_client.client.upload_dataframe(
            df, table_id, if_exists="append"
        )
        
        logger.info(f"Successfully saved {len(df)} vector index test results to BigQuery")
        
    except Exception as e:
        logger.error(f"Failed to save vector index results: {e}")
        
        # Fallback to JSON
        timestamp = int(time.time())
        backup_file = f"vector_index_test_backup_{timestamp}.json"
        with open(backup_file, 'w') as f:
            json.dump(results_list, f, indent=2, default=str)
        logger.info(f"Results saved to backup file: {backup_file}")

def analyze_vector_index_results(results_list: List[Dict]):
    """Analyze vector index performance"""
    df = pd.DataFrame(results_list)
    
    logger.info("\n" + "="*60)
    logger.info("VECTOR INDEX PERFORMANCE ANALYSIS")
    logger.info("="*60)
    
    # Separate by search method
    brute_force_df = df[df['search_method'] == 'brute_force']
    indexed_df = df[df['search_method'] == 'vector_index']
    creation_df = df[df['search_method'] == 'index_creation']
    
    if not brute_force_df.empty and not indexed_df.empty:
        # Performance comparison
        avg_bf_time = brute_force_df['search_time_ms'].mean()
        avg_idx_time = indexed_df['search_time_ms'].mean()
        avg_recall = indexed_df['recall_vs_brute_force'].mean()
        
        speedup = avg_bf_time / avg_idx_time
        
        logger.info(f"\nPERFORMANCE COMPARISON:")
        logger.info(f"  Brute force avg time: {avg_bf_time:.0f}ms")
        logger.info(f"  Vector index avg time: {avg_idx_time:.0f}ms")
        logger.info(f"  Speedup: {speedup:.1f}x faster")
        logger.info(f"  Average recall: {avg_recall:.3f} ({avg_recall*100:.1f}%)")
        
        # Accuracy by fraction_lists_searched
        logger.info(f"\nACCURACY BY SEARCH FRACTION:")
        for fraction in indexed_df['fraction_lists_searched'].unique():
            if pd.notna(fraction):
                subset = indexed_df[indexed_df['fraction_lists_searched'] == fraction]
                avg_recall_fraction = subset['recall_vs_brute_force'].mean()
                avg_time_fraction = subset['search_time_ms'].mean()
                logger.info(f"  Fraction {fraction}: {avg_recall_fraction:.3f} recall, {avg_time_fraction:.0f}ms avg")
    
    if not creation_df.empty:
        avg_creation_time = creation_df['index_creation_time_ms'].mean()
        logger.info(f"\nINDEX CREATION:")
        logger.info(f"  Average creation time: {avg_creation_time/1000:.1f} seconds")

# Run the test
if __name__ == "__main__":
    results = run_comprehensive_vector_index_test(run_environment="laptop")