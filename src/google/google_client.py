""""Google Client Class"""
import os
from typing import Any, Dict, List, Optional, Sequence
import google
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import pandas_gbq
import streamlit as st
from loguru import logger
from dotenv import load_dotenv
load_dotenv()


class GoogleClient:
    def __init__(self, project_id: str, credentials_path: Optional[str] = None):
        self.project_id = project_id
        self.credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self._client = self._initialize_client()
    
    def _initialize_client(self) -> bigquery.Client:
        """Initialize BigQuery client with credentials"""
        if self.credentials_path and os.path.exists(self.credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=['https://www.googleapis.com/auth/bigquery',
                       'https://www.googleapis.com/auth/cloud-platform']
        )

        else:
            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"],
                scopes=["https://www.googleapis.com/auth/bigquery",
                "https://www.googleapis.com/auth/cloud-platform"]
        )
        
        # used if env variable is set
        return bigquery.Client(project=self.project_id, credentials=credentials)
    
    def execute_query(self, query: str) -> None:
        """Execute DDL queries"""
        try:
            self._client.query(query)
        except Exception as err:
            logger.error(f"BigQuery DDL {query} execution failed {err}")
            raise
    
    def execute_sql_query(self, query: str, job_config: Optional[Sequence[str]] = None) -> bigquery.job.query.QueryJob:
        """Execute BigQuery SQL queries with optional job configuration. Used in Latency Testing"""
        try:
            job = self._client.query(query, job_config=job_config)
        except Exception as err:
            logger.error(f"BigQuery statement {query} execution failed {err}")
            raise
        return job
    
    def query_to_dataframe(self, query: str, job_config: Optional[Sequence[str]] = None, **kwargs) -> pd.DataFrame:
        """Execute BigQuery query and return Pandas DataFrame"""
        try:
            df = self._client.query(query, job_config=job_config).to_dataframe(**kwargs)
        except google.api_core.exceptions.BadRequest as err:
            error_message = str(err)
            if 'query_patent_numbers' in error_message:
                return pd.DataFrame()
            else:
                raise
        except Exception as err:
            logger.error(f"BigQuery {query} execution failed {err}")
            raise

        return df
    
    def upload_dataframe(
            self, 
            df: pd.DataFrame, 
            table_name: str,
            if_exists: str = "replace",
            chunk_size: int = 2000,
        ) -> None:
        """Upload DataFrame to BigQuery table"""

        # df.to_gbq(
        #     table_name,
        #     project_id=self.project_id,
        #     chunksize=chunk_size,
        #     if_exists=if_exists,
        # )
        try:
            pandas_gbq.to_gbq(
                df,
                table_name,
                project_id=self.project_id,
                # chunksize=chunk_size,
                if_exists=if_exists
            )
        except Exception as err:
            logger.error(f"Upload to BigQuery table {table_name} failed {err}")
            raise
        logger.info(f"Uploaded {len(df)} records to {table_name}")
    
    def create_table_from_query(self, query: str, table_name: str) -> None:
        """Create table from query"""
        create_query = f"CREATE TABLE IF NOT EXISTS `{table_name}` AS {query}"
        try:
            job = self.client.query(create_query)
        except Exception as err:
            logger.error(f"Create table {table_name} failed {err}")
            raise
        print(f"Created table: {table_name}")
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        try:
            self.query.get_table(table_name)
        except:
            return False
        return True
    
    def get_table_info(self, table_name: str) -> Dict:
        """Get basic table information"""
        if not self.table_exists(table_name):
            return {"exists": False}
        
        query = f"SELECT COUNT(*) as row_count FROM `{table_name}`"
        result = self.query_to_dataframe(query)

        return {
            "exists": True,
            "row_count": result["row_count"].iloc[0]
        }

if __name__ == "__main__":
    # Initialize client
    gc_client = GoogleClient(
        project_id=os.getenv("project_id"),
        credentials_path=os.getenv("service_account_path")
    )
    
    # Test connection
    test_df = gc_client.query_to_dataframe("SELECT 1 as test")
    print("Connection successful:", test_df['test'].iloc[0])