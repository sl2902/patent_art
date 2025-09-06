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


pss_client = PatentSemanticSearch(
    project_id=project_id,
    dataset_id=dataset_id,
    credentials_path=credentials_path
)

model_name =  "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name, token=hf_token)


def run_sanitization_step(query_text: str) -> Tuple[str, float]:
    """Sanitize the input query"""
    start_time = time.time()
    query_text, is_valid, error_msg = sanitize_input_query(query_text)
    if not is_valid:
        logger.warning(f"{error_msg}")
        logger.warning(f"Latency test failed. Change query terms and retry")
        return
    sanitization_time = time.time() - start_time

    return query_text, sanitization_time

def generate_embeddings(query_text: str) -> Tuple[List[float], float]:
    """Generate query embeddings"""
    start_time = time.time()
    embeddings = generate_local_embeddings_for_query(model, query_text)
    embedding_time = time.time() - start_time

    return embeddings, embedding_time

def execute_vector_search(
        query_embeddings: List[float],
        top_k: int = 20
        ) -> Tuple[pd.DataFrame, float, bigquery.job.query.QueryJob]:
    """Execute BigQuery AI Vector Search"""
    query = bq_queries.vector_search_query
    table_name = "patent_embeddings_local"
    filter_clause = ""
    query = query.format(
            project_id=project_id,
            dataset_id=dataset_id,
            table_name=table_name,
            top_k=top_k,
            filter_clause=filter_clause
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
    logger.info(f"Cache hit: {job.cache_hit}")
    
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

def test_single_query(query_text: str, top_k: int = 20):
    """Test end-to-end latency for a single query"""

    # Step 1: Run validation on input query
    logger.info(f"\nTesting query: '{query_text}'")
    query_text, sanitization_time = run_sanitization_step(query_text)
    logger.info(f"  Sanitization of query: {sanitization_time*1000:.1f}ms")

        
    # Step 2: Generate query embedding (CPU)
    query_embeddings, embedding_time = generate_embeddings(query_text)
    logger.info(f"  Embedding generation: {embedding_time*1000:.1f}ms")
    
    # Step 3: Execute vector search (BigQuery AI)
    try:
        results, search_time, job_id = execute_vector_search(query_embeddings)
        logger.info(f"  Vector search: {search_time*1000:.1f}ms")
        
        # Step 3: Calculate total time
        total_time = sanitization_time + embedding_time + search_time
        logger.info(f"  Total latency: {total_time*1000:.1f}ms")
        logger.info(f"  Results found: {len(results)}")
        
        # BigQuery job metrics
        logger.info(f"DEBUG - job.total_bytes_processed value: {job_id.cache_hit}")
        if not job_id.cache_hit:
            logger.info(f"  Bytes processed: {job_id.total_bytes_processed:,}")
            logger.info(f"  Slot milliseconds: {job_id.slot_millis:,}")
        
            return {
                'query': query_text,
                'embedding_time_ms': embedding_time * 1000,
                'search_time_ms': search_time * 1000,
                'total_time_ms': total_time * 1000,
                'results_count': len(results),
                'bytes_processed': job_id.total_bytes_processed,
                'slot_millis': job_id.slot_millis,
                'avg_similarity': results['cosine_score'].mean() if len(results) > 0 else 0,
                'max_similarity': results['cosine_score'].max() if len(results) > 0 else 0
            }
        else:
            return {
                'query': query_text,
                'embedding_time_ms': embedding_time * 1000,
                'search_time_ms': search_time * 1000,
                'total_time_ms': total_time * 1000,
                'results_count': len(results),
                'bytes_processed': 0,
                'slot_millis': 0,
                'avg_similarity': results['cosine_score'].mean() if len(results) > 0 else 0,
                'max_similarity': results['cosine_score'].max() if len(results) > 0 else 0
            }

        
    except Exception as e:
        logger.error(f"  Error: {e}")
        return None
    
    # Step 4: Execute semantic search explainability (CPU)
    # try:
    #     results, explain_time = execute_semantic_search_explainability(results, query_text)
    
    # except Exception as e:
    #     logger.error(f"  Error: {e}")
    #     return None


    
    
def run_comprehensive_latency_test():
    """Run comprehensive latency testing across multiple queries"""
    logger.info("\n" + "="*60)
    logger.info("COMPREHENSIVE LATENCY TEST")
    logger.info("="*60)
    
    test_queries = [
        "artificial intelligence machine learning neural networks",
        "renewable energy solar photovoltaic efficiency optimization",
        "autonomous vehicles self-driving navigation lidar sensors",
        "quantum computing quantum bits entanglement algorithms",
        "medical devices diagnostic imaging signal processing",
        "blockchain cryptocurrency distributed ledger consensus",
        "biotechnology genetic engineering CRISPR gene editing",
        "robotics automation control systems manufacturing",
        "cybersecurity encryption authentication network security",
        "nanotechnology materials science molecular engineering"
    ]
    
    results = []
    
    for query in test_queries:
        result = test_single_query(query)
        if result:
            results.append(result)
        time.sleep(1)  # Brief pause between queries
    
    if results:
        analyze_results(results)
        return results
    else:
        logger.info("No successful test results to analyze")
        return []

def analyze_results(results):
    """Analyze and summarize test results"""
    df = pd.DataFrame(results)
    
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE ANALYSIS SUMMARY")
    logger.info("="*60)
    
    logger.info(f"\nLATENCY METRICS:")
    logger.info(f"  Average total latency: {df['total_time_ms'].mean():.0f}ms")
    logger.info(f"  Median total latency: {df['total_time_ms'].median():.0f}ms")
    logger.info(f"  95th percentile latency: {df['total_time_ms'].quantile(0.95):.0f}ms")
    logger.info(f"  Min latency: {df['total_time_ms'].min():.0f}ms")
    logger.info(f"  Max latency: {df['total_time_ms'].max():.0f}ms")
    
    logger.info(f"\nCOMPONENT BREAKDOWN:")
    logger.info(f"  Average embedding time: {df['embedding_time_ms'].mean():.0f}ms")
    logger.info(f"  Average search time: {df['search_time_ms'].mean():.0f}ms")
    logger.info(f"  Embedding % of total: {(df['embedding_time_ms'].mean() / df['total_time_ms'].mean() * 100):.1f}%")
    logger.info(f"  Search % of total: {(df['search_time_ms'].mean() / df['total_time_ms'].mean() * 100):.1f}%")
    
    logger.info(f"\nRESULT QUALITY:")
    logger.info(f"  Average results per query: {df['results_count'].mean():.1f}")
    logger.info(f"  Average similarity score: {df['avg_similarity'].mean():.3f}")
    logger.info(f"  Max similarity score: {df['max_similarity'].mean():.3f}")
    
    logger.info(f"\nRESOURCE USAGE:")
    logger.info(f"  Average bytes processed: {df['bytes_processed'].mean():,.0f}")
    logger.info(f"  Average slot milliseconds: {df['slot_millis'].mean():,.0f}")
    
    # Performance benchmarks
    fast_queries = len(df[df['total_time_ms'] < 500])
    medium_queries = len(df[(df['total_time_ms'] >= 500) & (df['total_time_ms'] < 1000)])
    slow_queries = len(df[df['total_time_ms'] >= 1000])
    
    logger.info(f"\nPERFORMANCE DISTRIBUTION:")
    logger.info(f"  Sub-500ms queries: {fast_queries}/{len(df)} ({fast_queries/len(df)*100:.0f}%)")
    logger.info(f"  500ms-1s queries: {medium_queries}/{len(df)} ({medium_queries/len(df)*100:.0f}%)")
    logger.info(f"  >1s queries: {slow_queries}/{len(df)} ({slow_queries/len(df)*100:.0f}%)")
    
    return df

def test_concurrent_performance(num_concurrent=3):
    """Test concurrent query performance (simplified simulation)"""
    logger.info(f"\nTesting concurrent performance with {num_concurrent} queries...")
    
    concurrent_queries = [
        "artificial intelligence deep learning",
        "renewable energy systems",
        "quantum computing applications"
    ][:num_concurrent]
    
    start_time = time.time()
    results = []
    
    for query in concurrent_queries:
        result = test_single_query(query)
        if result:
            results.append(result)
    
    total_concurrent_time = time.time() - start_time
    
    logger.info(f"\nCONCURRENT PERFORMANCE:")
    logger.info(f"  Total time for {num_concurrent} queries: {total_concurrent_time*1000:.0f}ms")
    logger.info(f"  Average time per query: {total_concurrent_time/num_concurrent*1000:.0f}ms")
    
    return results

def main():
    """Main function to run latency tests"""
    logger.info("Patent Semantic Search - Latency Testing")
    logger.info("="*50)
    
    query_text = "Machine learning optimization"
    test_single_query(query_text)
    # Run comprehensive test
    # results = run_comprehensive_latency_test()
    
    # # Test concurrent performance
    # test_concurrent_performance()
    
    # # Save results
    # if results:
    #     timestamp = int(time.time())
    #     results_file = f"latency_test_results_{timestamp}.json"
        
    #     with open(results_file, 'w') as f:
    #         json.dump(results, f, indent=2)
        
    #     logger.info(f"\nResults saved to: {results_file}")
    
    # logger.info("\nLatency testing complete!")

if __name__ == "__main__":
    main()