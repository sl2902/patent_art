"""This module is used to test the latency of the patent semantic search pipeline"""
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

def create_latency_table():
    """Create latency results table in BigQuery"""
    query = bq_queries.create_latency_ddl
    query = query.format(
        project_id=project_id,
        dataset_id=dataset_id,
        latency_table="latency_test_results"
    )
    
    try:
        job = pss_client.client.execute_sql_query(query)
        job.result()  # Wait for completion
        logger.info("Latency results table created successfully")
    except Exception as e:
        logger.error(f"Failed to create latency table: {e}")

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
        top_k: int = 20
        ) -> Tuple[pd.DataFrame, int, bigquery.job.QueryJob]:
    """Execute BigQuery AI Vector Search"""
    query = bq_queries.vector_search_query
    table_name = "patent_embeddings_local"
    filter_clause = ""
    query = query.format(
            project_id=project_id,
            dataset_id=dataset_id,
            table_name=table_name,
            top_k=top_k,
            filter_clause=filter_clause,
            options=json.dumps({"use_brute_force": True})
        ) 

    start_time = time.time()
    try:
        params = [bigquery.ArrayQueryParameter("query_embeddings", "FLOAT64", query_embeddings)]
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        job = pss_client.client.execute_sql_query(query,  job_config=job_config)
    except Exception as err:
        logger.error(f"Vector search query {query} execution failed {err}")
        raise

    results = job.to_dataframe()
    # logger.info(results)
    search_time = time.time() - start_time
    
    return results, search_time, job

def execute_semantic_search_explainability(
        candidate_df: pd.DataFrame,
        query_text: str,
        query_patents: List[str] = None
):
    """Execute Sentence Similarity Explainability"""
    start_time = time.time()
    semantic_matches = pss_client.semantic_search_with_explanability(
            model, 
            candidate_df,
            query_text=query_text,
            query_patents=query_patents
        )
    explain_time = time.time() - start_time

    return semantic_matches, explain_time

def test_single_query(query_text: str, top_k: int = 20, run_environment: str = "laptop"):
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
        results, search_time, job = execute_vector_search(query_embeddings, top_k)
        logger.info(f"  Vector search: {search_time*1000:.1f}ms")
        
        # Step 4: Execute semantic search explainability (CPU)
        explainability_results, explainability_time = execute_semantic_search_explainability(
            results, query_text
        )
        logger.info(f"  Explainability analysis: {explainability_time*1000:.1f}ms")
        
        # Calculate total times
        vector_search_total = sanitization_time + embedding_time + search_time
        complete_pipeline_total = vector_search_total + explainability_time
        
        logger.info(f"  Vector search pipeline: {vector_search_total*1000:.1f}ms")
        logger.info(f"  Complete pipeline total: {complete_pipeline_total*1000:.1f}ms")
        logger.info(f"  Results found: {len(results)}")
        logger.info(f"  Explainability results: {len(explainability_results) if explainability_results is not None else 0}")
        
        # BigQuery job metrics
        if not job.cache_hit:
            logger.info(f"  Bytes processed: {job.total_bytes_processed:,}")
            logger.info(f"  Slot milliseconds: {job.slot_millis:,}")
            base_metrics = {
                'query': query_text,
                'run_environment': run_environment,
                'cache_hit': job.cache_hit,
                'embedding_time_ms': embedding_time * 1000,
                'search_time_ms': search_time * 1000,
                'results_count': len(results),
                'bytes_processed': job.total_bytes_processed,
                'slot_millis': job.slot_millis,
                'min_similarity': results['cosine_score'].min() if len(results) > 0 else 0,
                'avg_similarity': results['cosine_score'].mean() if len(results) > 0 else 0,
                'max_similarity': results['cosine_score'].max() if len(results) > 0 else 0
            }
        else:
            logger.info(f"  Bytes processed: 0")
            logger.info(f"  Slot milliseconds: 0")
            base_metrics = {
                'query': query_text,
                'run_environment': run_environment,
                'cache_hit': job.cache_hit,
                'embedding_time_ms': embedding_time * 1000,
                'search_time_ms': search_time * 1000,
                'results_count': len(results),
                'bytes_processed': 0,
                'slot_millis': 0,
                'min_similarity': results['cosine_score'].min() if len(results) > 0 else 0,
                'avg_similarity': results['cosine_score'].mean() if len(results) > 0 else 0,
                'max_similarity': results['cosine_score'].max() if len(results) > 0 else 0
            }
        
        # Prepare results for both test types

        
        # Vector search only metrics
        vector_search_metrics = {
            **base_metrics,
            'test_type': 'vector_search',
            'explainability_time_ms': 0.0,  # No explainability in this test
            'total_time_ms': vector_search_total * 1000
        }
        
        # Complete pipeline metrics
        complete_pipeline_metrics = {
            **base_metrics,
            'test_type': 'complete_pipeline',
            'explainability_time_ms': explainability_time * 1000,
            'total_time_ms': complete_pipeline_total * 1000
        }
        
        return [vector_search_metrics, complete_pipeline_metrics]
        
    except Exception as e:
        logger.error(f"  Error: {e}")
        return None

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
    
    # # Add timestamp
    # df['run_date'] = datetime.now()
    
    try:
        # Upload to BigQuery
        table_id = f"{project_id}.{dataset_id}.latency_test_results"
        
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
        backup_file = f"latency_test_backup_{timestamp}.json"
        with open(backup_file, 'w') as f:
            json.dump(flattened_results, f, indent=2, default=str)
        logger.info(f"Results saved to backup file: {backup_file}")

def run_comprehensive_latency_test(run_environment: str = "laptop"):
    """Run comprehensive latency testing across multiple queries"""
    logger.info("\n" + "="*60)
    logger.info("COMPREHENSIVE LATENCY TEST")
    logger.info(f"Environment: {run_environment}")
    logger.info("="*60)
    
    # Create table if it doesn't exist
    create_latency_table()
    
    # since these queries were tested before adding explainability, google returned cached results
    # test_queries = [
    #     "artificial intelligence machine learning neural networks",
    #     "renewable energy solar photovoltaic efficiency optimization",
    #     "autonomous vehicles self-driving navigation lidar sensors",
    #     "quantum computing quantum bits entanglement algorithms",
    #     "medical devices diagnostic imaging signal processing",
    #     "blockchain cryptocurrency distributed ledger consensus",
    #     "biotechnology genetic engineering CRISPR gene editing",
    #     "robotics automation control systems manufacturing",
    #     "cybersecurity encryption authentication network security",
    #     "nanotechnology materials science molecular engineering"
    # ]
    test_queries = [
             "optical fiber communication networks wavelength division multiplexing",
            "battery thermal management systems electric vehicle cooling",
            "CRISPR gene editing therapeutic applications protein engineering",
            "carbon nanotube composite materials strength enhancement",
            "millimeter wave antenna design 5G wireless communication",
            "photovoltaic cell efficiency perovskite semiconductor materials",
            "robotic surgical instruments haptic feedback control systems",
            "blockchain consensus mechanisms proof of stake validation",
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
    """Analyze and summarize test results"""
    # Flatten results for analysis
    flattened_results = []
    for result in results_list:
        if result:
            if isinstance(result, list):
                flattened_results.extend(result)
            else:
                flattened_results.append(result)
    
    df = pd.DataFrame(flattened_results)
    
    # Separate analysis by test type
    vector_search_df = df[df['test_type'] == 'vector_search']
    complete_pipeline_df = df[df['test_type'] == 'complete_pipeline']
    
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE ANALYSIS SUMMARY")
    logger.info("="*60)
    
    if not vector_search_df.empty:
        logger.info(f"\nVECTOR SEARCH ONLY PERFORMANCE:")
        logger.info(f"  Average total latency: {vector_search_df['total_time_ms'].mean():.0f}ms")
        logger.info(f"  95th percentile latency: {vector_search_df['total_time_ms'].quantile(0.95):.0f}ms")
        logger.info(f"  Average embedding time: {vector_search_df['embedding_time_ms'].mean():.0f}ms")
        logger.info(f"  Average search time: {vector_search_df['search_time_ms'].mean():.0f}ms")
    
    if not complete_pipeline_df.empty:
        logger.info(f"\nCOMPLETE PIPELINE PERFORMANCE:")
        logger.info(f"  Average total latency: {complete_pipeline_df['total_time_ms'].mean():.0f}ms")
        logger.info(f"  95th percentile latency: {complete_pipeline_df['total_time_ms'].quantile(0.95):.0f}ms")
        logger.info(f"  Average explainability time: {complete_pipeline_df['explainability_time_ms'].mean():.0f}ms")
        logger.info(f"  Explainability overhead: {(complete_pipeline_df['explainability_time_ms'].mean() / complete_pipeline_df['total_time_ms'].mean() * 100):.1f}%")
    
    if not df.empty:
        logger.info(f"\nRESOURCE USAGE:")
        logger.info(f"  Average bytes processed: {df['bytes_processed'].mean():,.0f}")
        logger.info(f"  Average slot milliseconds: {df['slot_millis'].mean():,.0f}")
        logger.info(f"  Average similarity score: {df['avg_similarity'].mean():.3f}")
    
    return df

def main():
    """Main function to run latency tests"""
    logger.info("Patent Semantic Search - Latency Testing")
    logger.info("="*50)
    
    # Determine run environment (you can modify this)
    run_environment = "laptop"  # Change to "kaggle" when running on Kaggle

    # query_text = "Machine learning"
    # test_single_query(query_text)
    
    # Run comprehensive test
    results = run_comprehensive_latency_test(run_environment=run_environment)
    
    # Test concurrent performance (optional)
    # test_concurrent_performance()
    
    logger.info("\nLatency testing complete!")
    logger.info("Results saved to BigQuery for visualization")

if __name__ == "__main__":
    main()