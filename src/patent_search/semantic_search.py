""""Define a class and various functions that perform Semantic Search on Patents"""
import os
import re
from google.cloud import bigquery
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from src.google import google_client
from src.sql_queries import bq_queries
from loguru import logger
import torch
import gc
from dotenv import load_dotenv
load_dotenv()

class PatentSemanticSearch:
    """
    Advanced Patent Semantic Search System using BigQuery AI Features
    """

    def __init__(
            self, 
            project_id: str, 
            dataset_id: str, 
            credentials_path: Optional[str] = None
        ):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.credentials_path = credentials_path
        self.client = google_client.GoogleClient(
            project_id,
            credentials_path
        )
    
    def create_embeddings_table(self, table_name: str):

        qry = bq_queries.create_embedding_ddl.format(
            project_id=self.project_id,
            dataset_id=self.dataset_id,
            table_name=table_name
        )
        try:
            self.client.execute_query(qry)
        except Exception as err:
            logger.error(f"Failed to create embedding table {table_name} - {err}")
            raise
    
    def create_vector_index(
            self, 
            vector_index: str, 
            embedding_table_name: str, 
            embedding_col: str = "text_embedding"
        ):

        qry = bq_queries.create_vector_index(
            project_id=self.project_id,
            dataset_id=self.dataset_id,
            vector_index=vector_index,
            embedding_table_name=embedding_table_name,
            embedding_col=embedding_col,
        )
        try:
            self.client.execute_query(qry)
        except Exception as err:
            logger.error(f"Failed to create vector index {vector_index} on table {embedding_table_name} - {err}")
    
    def generate_local_batch_embeddings(
            self, 
            model: SentenceTransformer,
            table_name: str,
            df_patents: pd.DataFrame,
            num_batches: int,
            embedding_batch_size: int = 1024,
            batch_num: int = 0,
            chunk_size: int = 1024,
    ):
        """"Generate batch embedding using Sentence Transformer and upload to BQ"""
        embeddings = model.encode(
            df_patents["combined_text"].tolist(),
            batch_size=embedding_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        df_batch_with_embeddings = df_patents.copy()
        df_batch_with_embeddings["text_embedding"] = embeddings.tolist()

        # table_name = f'{self.project_id}.{self.dataset_id}.{table_name}'
        self.client.upload_dataframe(
            df_batch_with_embeddings,
            table_name,
            if_exists="append",
            chunk_size=chunk_size,
        )

        # Free up Kaggle GPU memory
        del embeddings, df_batch_with_embeddings
        torch.cuda.empty_cache()
        _ = gc.collect()

        logger.info(f"Completed batch {batch_num} of {num_batches}")
    
    def get_processed_count(self, table_name: str, start_date: str, end_date: str) -> int:
        """Keeps track of how many batches of embeddings were processed"""
        qry = bq_queries.track_embedding_count(
                 project_id=self.project_id,
                 dataset_id=self.dataset_id,
                 table_name=table_name,
                 start_date=start_date,
                 end_date=end_date,
        )

        try:
            result = self.client.query_to_dataframe(qry)
        except Exception as err:
            logger.error(f"Failed to execute query to track embedding loading progress - {err}")
        
        return result["processed_count"].iloc[0]
    
    def semantic_search_with_bq(
            self,
            model: SentenceTransformer,
            table_name: str,
            query_text: str = None,
            query_patent_numbers: list = None,
            countries: list = None,
            date_start: str = None,
            date_end: str = None,
            top_k: int = 5,
        ) -> pd.DataFrame:
        """Perform Semantic Search using BigQuery's AI features"""
        if query_text:
            query_embedding = generate_local_embeddings_for_query(model, query_text)
        elif query_patent_numbers:
            avg_query = bq_queries.query_patents_embedding_avgs
            avg_query = avg_query.format(
                project_id=self.project_id,
                dataset_id=self.dataset_id,
                table_name=table_name
            )
        else:
            raise ValueError("Enter value for either a query_text` field or `query_patents_numbers` field")

        if query_patent_numbers:
            result = self.client.query_to_dataframe(avg_query,
                                job_config=bigquery.QueryJobConfig(
                                    query_parameters=[
                                        bigquery.ArrayQueryParameter("patent_numbers", "STRING", query_patent_numbers)
                                    ]
                                ))
            query_embedding = result['avg_embedding'][0].tolist()
            logger.info(f"Computed the average embedding vectors for list of patents")

        logger.info(f"Embedding dimension vector length - {len(query_embedding)}")

        filters = []
        params = [bigquery.ArrayQueryParameter("query_embeddings", "FLOAT64", query_embedding)]

        if date_start:
            filters.append("pub_date >= @date_start")
            params.append(bigquery.ScalarQueryParameter("date_start", "STRING", date_start))
        if date_end:
            filters.append("pub_date <= @date_end")
            params.append(bigquery.ScalarQueryParameter("date_end", "STRING", date_end))
        if countries:
            filters.append("country_code IN UNNEST(@countries)")
            params.append(bigquery.ArrayQueryParameter("countries", "STRING", countries))
        if query_patent_numbers:
            filters.append("publication_number NOT IN UNNEST(@query_patent_numbers)")
            params.append(bigquery.ScalarQueryParameter("publication_number", "STRING", query_patent_numbers))

        filter_clause = " AND ".join(filters)
        if filter_clause:
            filter_clause = "AND " + filter_clause
        else:
            filter_clause = ""

        
        query = bq_queries.vector_search_query
        query = query.format(
            project_id=self.project_id,
            dataset_id=self.dataset_id,
            table_name=table_name,
            top_k=top_k,
            filter_clause=filter_clause
        )

        params.append(bigquery.ScalarQueryParameter("top_k", "INT64", top_k))
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        logger.info("Computing cosine similarity with BigQuery vector search")
        return self.client.query_to_dataframe(query, job_config=job_config)
    
    def explain_via_sentence_similarity(
        self,
        model: SentenceTransformer, 
        query_text: str, 
        candidate_title: str, 
        candidate_abstract: str,
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Find which sentences in the candidate are most similar to the query
        """
        full_text = f"{candidate_title}. {candidate_abstract}"
        sentences = re.split(r'[.!?]+', full_text)
        sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 10]

        if not sentences:
            return [{"sentence": "No meaningful sentences found", "similarity": 0.0}]
        
        query_embedding = model.encode([query_text], show_progress_bar=False)
        sentence_embeddings = model.encode(sentences, show_progress_bar=False)

        similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]

        top_indices = np.argsort(similarities)[::-1]

        explanations = []
        for idx in top_indices[:3]:
            if similarities[idx] > threshold:
                explanations.append({
                    'sentence': sentences[idx],
                    'similarity': similarities[idx]
                })

        if not explanations and len(similarities) > 0:
            best_idx = np.argmax(similarities)
            explanations.append({
                    'sentence': sentences[best_idx],
                    'similarity': similarities[best_idx]
                })
        return explanations
    
    def generate_explanations_for_candidates(
            self,
            model: SentenceTransformer, 
            query_text: str, 
            candidate_df: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Generate semantic explanations for search results
        """
        explanations = []

        for row in candidate_df.itertuples(index=False):
            sentence_explanations = self.explain_via_sentence_similarity(
                model,
                query_text,
                row.title_en,
                row.abstract_en,
                threshold=0.3,
            )
            explanations.append(sentence_explanations)
        
        candidate_df = candidate_df.copy()
        candidate_df['explanation'] = explanations
        return candidate_df
    
    def semantic_search_with_explanability(
            self,
            model: SentenceTransformer, 
            candidate_df: pd.DataFrame, 
            query_text: str = None, 
            query_patents: List[str]= None
        ) -> pd.DataFrame:
        """Generate explanations for candidate results using user query or list of patents"""

        if query_text:
            results_with_explanations = self.generate_explanations_for_candidates(model, query_text, candidate_df)
        elif query_patents:
            results_with_explanations = self.generate_explanations_for_candidates(model, query_patents, candidate_df)
        return results_with_explanations

            



def generate_local_embeddings_for_query(
        model: SentenceTransformer,
        query_text: str,
    ) -> List[float]:
        """Generate embeddings for input query"""
        embeddings = model.encode(
                query_text,
                # batch_size=embedding_batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        return embeddings.tolist()

def extract_top_terms(corpus: List[str], top_n: int = 10, stop_words: str = "english"):
    """
    Returns top_n terms per document using TF-IDF weights.
    corpus: list of strings (doc0= query, doc1..docN = candidates)
    returns: list of lists (top terms per document)

    Initial attempt at explainability
    """
    vect = TfidfVectorizer(ngram_range=(1, 2), stop_words=stop_words, max_features=10000)
    X = vect.fit_transform(corpus)
    feature_names = np.array(vect.get_feature_names_out())
    top_terms = []
    for i in range(X.shape[0]):
        row = X[i].toarray().ravel()
        top_idx = np.argsort(row)[-top_n:][::-1]
        terms = feature_names[top_idx][row[top_idx] > 0].tolist()
        top_terms.append(terms)
    return top_terms

def explain_results(query_text: str, candidate_df: pd.DataFrame, n_terms: int = 10):
    """Build corpus: query first, then each candidate's combined text
    Initial attemp at explainability
    """
    docs = [query_text] + \
    [
        (row.title_en or "") + " " + (row.abstract_en or "")
        for row in candidate_df.itertuples(index=False)
    ]
    top_terms = extract_top_terms(docs, top_n=n_terms)
    query_terms = set(top_terms[0])
    explanations = []
    for idx, row in enumerate(candidate_df.itertuples(index=False), start=1):
        cand_terms = set(top_terms[idx])
        overlap = list(cand_terms & query_terms)
        if overlap:
            rationale = f"Shared terms: {', '.join(overlap[:5])}"
        else:
            rationale = "No top-term overlap found — similarity driven by latent semantics."
        explanations.append({
            "publication_number": row.publication_number,
            "score": row.similarity_score,
            "top_terms": top_terms[idx],
            "overlap": overlap,
            "rationale": rationale,
            "title": row.title_en,
            "abstract": row.abstract_en
        })
    return explanations