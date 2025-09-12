"""Contains various BigQuery sql queries used for Data analysis, patents embedding, indexing, querying and search"""
import os
import pandas as pd


# Exploratory analysis queries

# Dataset size
dataset_size_qry = """
WITH base AS (
        SELECT 
        COUNT(*) as total_patents,
        MIN(pub_date) as earliest_date,
        MAX(pub_date) as latest_date,
        COUNT(DISTINCT country_code) as unique_countries,
        COUNT(DISTINCT family_id) as unique_families,
        -- Text quality metrics
        AVG(LENGTH(title_en)) as avg_title_length,
        AVG(LENGTH(abstract_en)) as avg_abstract_length,
        -- Data completeness
        ROUND(COUNT(title_en) / COUNT(*) * 100, 1) as title_completeness_pct,
        ROUND(COUNT(abstract_en) / COUNT(*) * 100, 1) as abstract_completeness_pct,
        ROUND(COUNT(claims_en) / COUNT(*) * 100, 1) as claims_completeness_pct
        FROM `{project_id}.{dataset_id}.{publication_table}`
), cpc AS (
        -- CPC percentage coverage
        SELECT 
        COUNT(CASE WHEN ARRAY_LENGTH(cpc_codes) > 0 THEN 1 END) as patents_with_codes,
        ROUND(
            COUNT(CASE WHEN ARRAY_LENGTH(cpc_codes) > 0 THEN 1 END) / COUNT(*) * 100, 
            1
        ) as coverage_pct,
        ROUND(AVG(ARRAY_LENGTH(cpc_codes)), 1) as avg_codes_per_patent
        FROM `{project_id}.{dataset_id}.{publication_table}`
    )
        SELECT
            total_patents,
            unique_countries,
            unique_families,
            avg_title_length,
            avg_abstract_length,
            title_completeness_pct,
            abstract_completeness_pct,
            claims_completeness_pct,
            patents_with_codes,
            coverage_pct,
            avg_codes_per_patent
        FROM base, cpc
"""

# Country-wise breakdown of patent publications
country_wise_publications = """SELECT 
        country_code,
        COUNT(*) as total_publications,
        ROUND(COUNT(*) / (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{publication_table}`) * 100, 2)  as percentage,
        MIN(pub_date) as earliest_date,
        MAX(pub_date) as latest_date,
        COUNT(DISTINCT country_code) as unique_countries,
        COUNT(DISTINCT family_id) as unique_families,
        -- Text quality metrics
        AVG(LENGTH(title_en)) as avg_title_length,
        AVG(LENGTH(abstract_en)) as avg_abstract_length,
        ROUND(AVG(ARRAY_LENGTH(cited_patents)), 1) as unique_patents,
        -- Data completeness
        ROUND(COUNT(title_en) / COUNT(*) * 100, 1) as title_completeness_pct,
        ROUND(COUNT(abstract_en) / COUNT(*) * 100, 1) as abstract_completeness_pct,
        ROUND(COUNT(claims_en) / COUNT(*) * 100, 1) as claims_completeness_pct
        FROM `{project_id}.{dataset_id}.{publication_table}`
        GROUP BY country_code
        ORDER BY total_publications DESC
        LIMIT {top_n}
"""

# Top country each month
top_country_each_month = """
        WITH year_month AS (
        SELECT
            country_code,
            EXTRACT(YEAR FROM pub_date) as year,
            EXTRACT(MONTH FROM pub_date) as month,
            DATE(EXTRACT(YEAR FROM pub_date), EXTRACT(MONTH FROM pub_date), 1) as month_date
        --   ROW_NUMBER() OVER(PARTITION BY EXTRACT(YEAR FROM pub_date), EXTRACT(MONTH FROM pub_date)  ORDER BY COUNT(*) DESC) as rn
        FROM `{project_id}.{dataset_id}.{publication_table}`
        ), year_wise_count AS (
        SELECT
            year,
            month,
            month_date,
            COUNT(*) as publication_count,
            COUNT(DISTINCT country_code) as unique_countries
        FROM year_month
        GROUP BY
            year,
            month,
            month_date
        ), rank_top_countries_by_month AS (
        SELECT
        country_code,
        year,
        month,
        month_date,
        COUNT(*) as publication_count_by_country,
        ROW_NUMBER() OVER(PARTITION BY month_date  ORDER BY COUNT(*) DESC) as rn
        FROM year_month
        GROUP BY
        country_code,
        year,
        month,
        month_date
        )
        SELECT
        country_code,
        a.year,
        a.month,
        a.month_date,
        unique_countries,
        publication_count_by_country,
        publication_count,
        SAFE_DIVIDE(publication_count_by_country, publication_count) as top_country_share
        FROM year_wise_count a JOIN rank_top_countries_by_month b ON a.month_date = b.month_date
        WHERE rn = 1
        ORDER BY 
        a.month_date
"""

# Year on Year English-language publications growth rate
yoy_eng_lang_publications_gr = """
 WITH yoy AS (
        SELECT
            EXTRACT(year FROM pub_date) as year,
            COUNT(*) pub_count
        FROM `{project_id}.{dataset_id}.{publication_table}`
        WHERE EXTRACT(year FROM pub_date) < EXTRACT(YEAR FROM CURRENT_TIMESTAMP())
        GROUP BY year
        ), compute_cagr AS (
        SELECT
            POW(
                SAFE_DIVIDE(MAX_BY(pub_count, year), MIN_BY(pub_count, year)),
                1.0 / (MAX(year) - MIN(year))
        ) - 1 AS cagr
        FROM yoy
        ) 
        SELECT
        year,
        pub_count,
        ROUND( 100 * (pub_count / lag(pub_count) over(order by year) - 1), 2) as yoy_growth,
        ROUND(100 * cagr, 2) as cagr
        FROM yoy, compute_cagr
        ORDER BY year


"""

# Year on Year growth rate of top 10 countries
yoy_top_n_countries = """
        WITH top_10_countries AS (
        SELECT
            country_code
        FROM `{project_id}.{dataset_id}.{publication_table}`
        GROUP BY
            country_code
        ORDER BY
            COUNT(*) DESC
        LIMIT {top_n}
        ), yearly_stats AS (
        SELECT
            EXTRACT(YEAR FROM pub_date) as year,
            a.country_code,
            COUNT(*) as publication_count
        FROM `{project_id}.{dataset_id}.{publication_table}` a join top_10_countries b on a.country_code = b.country_code
        WHERE 1=1
        {filter_clause}
        GROUP BY
            year,
            a.country_code
        )
        SELECT
        year,
        country_code,
        ROUND(
            (publication_count 
            / LAG(publication_count) OVER (PARTITION BY country_code ORDER BY year) - 1) * 100, 
            1) as yoy_growth
        FROM yearly_stats
        ORDER BY
        country_code,
        year
"""

# Citation pattern by top countries
citation_top_n_countries = """
        SELECT
        country_code,
        COUNT(*) as total_patents,
        ROUND((COUNT(*) / (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{publication_table}`)) * 100, 1) as patent_share,
        COUNT(CASE WHEN ARRAY_LENGTH(cited_patents) > 0 THEN 1 END) AS patents_with_citations,
        ROUND(
            COUNT(CASE WHEN ARRAY_LENGTH(cited_patents) > 0 THEN 1 END) / COUNT(*) * 100,
            1
        ) as citation_rate_pct,
        ROUND(AVG(ARRAY_LENGTH(cited_patents)), 1) as avg_citations_per_patent,
        MAX(ARRAY_LENGTH(cited_patents)) as max_citations,
        -- Highly cited patents (10+ citations)
        COUNT(CASE WHEN ARRAY_LENGTH(cited_patents) >= 10 THEN 1 END) as highly_cited_patents
        FROM `{project_id}.{dataset_id}.{publication_table}`
        GROUP BY country_code
        ORDER BY total_patents DESC
        LIMIT {top_n}
"""

# Top n CPCs (Corporate Patent Code)
top_n_cpc = """
    WITH cpc_unnested AS (
        SELECT 
            cpc_code,
            COUNT(*) as cpc_count
        FROM `{project_id}.{dataset_id}.{publication_table}`,
        UNNEST(cpc_codes) as cpc_code
        WHERE cpc_code IS NOT NULL
        GROUP BY cpc_code
    )
    SELECT 
    cpc_code,
    cpc_count,
    100 * cpc_count / (SELECT SUM(cpc_count) FROM cpc_unnested) as cpc_share,
    -- Get first 4 characters for main classification
    SUBSTR(cpc_code, 1, 5) as main_class
    FROM cpc_unnested
    ORDER BY cpc_share DESC
    LIMIT {top_n}
"""

# Technology area analysis by main CPC classes
tech_area_cpc_class = """
        WITH cpc_main_classes AS (
        SELECT 
            SUBSTR(cpc_code, 1, 1) as main_section,
            SUBSTR(cpc_code, 1, 4) as main_class,
            cpc_code,
            COUNT(*) as patent_count
        FROM `{project_id}.{dataset_id}.{publication_table}`,
        UNNEST(cpc_codes) as cpc_code
        WHERE cpc_code IS NOT NULL
        GROUP BY main_section, main_class, cpc_code
        )
        SELECT 
        main_section,
        -- Manual mapping of main CPC sections
        CASE main_section
            WHEN 'A' THEN 'Human Necessities'
            WHEN 'B' THEN 'Operations & Transport'
            WHEN 'C' THEN 'Chemistry & Metallurgy'
            WHEN 'D' THEN 'Textiles'
            WHEN 'E' THEN 'Construction'
            WHEN 'F' THEN 'Mechanical Engineering'
            WHEN 'G' THEN 'Physics'
            WHEN 'H' THEN 'Electricity'
            WHEN 'Y' THEN 'Emerging Technologies'
            ELSE 'Other'
        END as section_description,
        COUNT(DISTINCT main_class) as unique_classes,
        SUM(patent_count) as total_patents,
        ROUND(SUM(patent_count) / (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{publication_table}`) * 100, 1) as percentage
        FROM cpc_main_classes
        GROUP BY main_section, section_description
        ORDER BY total_patents DESC;
"""

# Patent publication flow
patent_flow = """
        WITH year_section AS (
        SELECT
        EXTRACT(YEAR FROM pub_date) as year,
        CASE 
        WHEN country_code IN ('US', 'KR', 'WO', 'EP', 'RU', 'CN', 'CA', 'JP', 'TW', 'AU') THEN country_code ELSE "Other" END as country_code,
        SUBSTR(cpc_codes, 1, 1) as main_section,
        COUNT(*) as publication_cpc_count
        FROM `{project_id}.{dataset_id}.{publication_table}`,
        UNNEST(cpc_codes) as cpc_codes
        GROUP BY year, country_code, main_section
        -- Filter noise
        HAVING publication_cpc_count >= 100000
        )
        SELECT
        year,
        country_code,
        main_section,
        CASE main_section
            WHEN 'A' THEN 'Human Necessities'
            WHEN 'B' THEN 'Operations & Transport'
            WHEN 'C' THEN 'Chemistry & Metallurgy'
            WHEN 'D' THEN 'Textiles'
            WHEN 'E' THEN 'Construction'
            WHEN 'F' THEN 'Mechanical Engineering'
            WHEN 'G' THEN 'Physics'
            WHEN 'H' THEN 'Electricity'
            WHEN 'Y' THEN 'Emerging Technologies'
            ELSE 'Other'
        END AS section_description,
        publication_cpc_count
        FROM year_section
        ORDER BY year, publication_cpc_count DESC
"""

# Technology convergence
tech_convergence = """
            -- Fetch the first letter to keep it simpler
            WITH cpc_combinations AS (
                SELECT
                    EXTRACT(YEAR FROM pub_date) as year,
                    ARRAY_TO_STRING(ARRAY(
                        SELECT DISTINCT SUBSTR(cpc, 1, 1) as cpc_class
                        FROM UNNEST(cpc_codes) as cpc
                        ORDER BY cpc_class
                    ), ',') as cpc_combo,
                    COUNT(*) patent_count
                FROM `{project_id}.{dataset_id}.{publication_table}`
                WHERE ARRAY_LENGTH(cpc_codes) >= 2
                GROUP BY year, cpc_combo
                HAVING patent_count >= 100 -- High threshold
            )
            SELECT 
            cpc_combo,
            CASE cpc_combo
                WHEN 'G,H' THEN 'Physics + Electricity'
                WHEN 'G,Y' THEN 'Physics + Emerging Tech'
                WHEN 'H,Y' THEN 'Electricity + Emerging Tech'
                WHEN 'A,C' THEN 'Human Needs + Chemistry'
                WHEN 'B,G' THEN 'Transport + Physics'
                ELSE 'Other' END cpc_combo_label,
            ROUND(AVG(patent_count)) as avg_recent_patents,
            MAX(patent_count) as peak_patents,
            COUNT(DISTINCT year) as years_with_data
            FROM cpc_combinations  
            -- WHERE year >= 2020
            WHERE ARRAY_LENGTH(SPLIT(cpc_combo, ',')) >= 2  -- Filter for actual combinations
            GROUP BY cpc_combo
            ORDER BY avg_recent_patents DESC
            LIMIT {top_n}
"""

# Fetch patents for a given date range"""
extract_qry = """
    SELECT
        publication_number,
        country_code,
        pub_date,
        title_en,
        abstract_en,
        CONCAT(COALESCE(title_en, ''), ' ', COALESCE(abstract_en, '')) as combined_text
    FROM `{project_id}.{dataset_id}.{table_name}`
    WHERE pub_date BETWEEN '{start_date}' AND '{end_date}'
        AND LENGTH(title_en) >= 30
        AND LENGTH(abstract_en) >= 100
    ORDER BY pub_date DESC
"""

create_embedding_ddl = """
    CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.{embeddings_table}`
    (
        publication_number STRING,
        country_code STRING,
        pub_date DATE,
        title_en STRING,
        abstract_en STRING,
        combined_text STRING,
        text_embedding ARRAY<FLOAT64>
    )
    PARTITION BY DATE_TRUNC(pub_date, MONTH)
    CLUSTER BY publication_number, country_code
    OPTIONS(
        description = "Core patent embeddings without CPC codes - optimized for semantic search performance"
    )
"""

delete_embedding_table = """
    DELETE FROM `{project_id}.{dataset_id}.{embeddings_table}`
    WHERE pub_date BETWEEN '{start_date}' and '{end_date}'
"""

create_vector_index = """
    CREATE VECTOR INDEX IF NOT EXISTS `{project_id}.{dataset_id}.{vector_index}`
    ON `{project_id}.{dataset_id}.{table_name}`({embedding_column})
    STORING (publication_number, title_en, abstract_en, pub_date, country_code)
    OPTIONS (
        index_type = 'IVF',
        distance_type = 'COSINE',
        ivf_options = '{{"num_lists": {num_lists}}}'
    )
"""

# Tracks count of embeding batches loaded so far
track_embedding_count = """
    SELECT
        COUNT(*) as processed_count
    FROM `{project_id}.{dataset_id}.{table_name}`
    WHERE pub_date BETWEEN '{start_date}' AND '{end_date}'
"""

# Store procedure to loading embeddings into BQ using ML.GenerateEmbeddings()

creat_embedding_model_bq = """
        CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.{embedding_model}`
        REMOTE WITH CONNECTION `{project_id}.{region}.{connection_id}`
        OPTIONS (
        ENDPOINT = 'text-embedding-005'
        )
"""

create_logs_ddl = """
    CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.{table_name}`
    (
        log_timestamp TIMESTAMP,
        batch_number INT64,
        batch_start INT64,
        batch_end INT64,
        records_processed INT64,
        processing_duration_seconds INT64,
        status STRING
    )
    PARTITION BY DATE(log_timestamp)
    CLUSTER BY batch_number;
"""

# Store procedure using BigQyery's ML.GENERATE_EMBEDDING()
proc_load_embeddings_bq = """
    CREATE OR REPLACE PROCEDURE `{project_id}.{dataset_id}.{embedding_proc}`
    (
        source_table STRING,
        start_date STRING,
        end_date STRING,
        batch_size INT64
    )
    BEGIN
        DECLARE total_rows INT64;
        DECLARE num_batches INT64;
        DECLARE batch_start INT64 DEFAULT 1;
        DECLARE batch_end INT64;
        DECLARE current_batch INT64 DEFAULT 1;
        DECLARE batch_start_time TIMESTAMP;
        DECLARE batch_end_time TIMESTAMP;
        DECLARE proc_start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP();
    
    INSERT INTO `{project_id}.{dataset_id}.{logs_table}`
    VALUES (CURRENT_TIMESTAMP(), 0, 0, 0, 0, 0, 'PROCEDURE_STARTED');

    EXECUTE IMMEDIATE FORMAT('''
        CREATE OR REPLACE TEMP TABLE filtered_patents AS
        SELECT
            publication_number,
            country_code,
            SAFE.PARSE_DATE('%%Y%%m%%d', CAST(publication_date AS STRING)) AS pub_date,
            title_en,
            abstract_en,
            CONCAT(COALESCE(title_en, ''), ' ', COALESCE(abstract_en, '')) AS combined_text
        FROM `%s.%s.%s`
        WHERE pub_date BETWEEN '%s' AND '%s'
            AND title_en IS NOT NULL
            AND abstract_en IS NOT NULL
            AND LENGTH(title_en) >= 30
            AND LENGTH(abstract_en) >= 100
        ''', '{project_id}', '{dataset_id}', source_table, start_date, end_date
        );
        
    CREATE OR REPLACE TEMP TABLE numbered_patents AS
    SELECT
        *,
        ROW_NUMBER() OVER(ORDER BY publication_number) AS rn
    FROM filtered_patents;

    SET total_rows = (SELECT COUNT(*) FROM numbered_patents);
    SET num_batches = CAST(CEIL(total_rows / batch_size) AS INT64);

    INSERT INTO `{project_id}.{dataset_id}.{logs_table}`
    VALUES (CURRENT_TIMESTAMP(), -1, 0, 0, total_rows, 0, 
        FORMAT('FOUND_%d_RECORDS_%d_BATCHES', total_rows, num_batches));
    
    WHILE current_batch <= num_batches DO
        SET batch_start_time = CURRENT_TIMESTAMP();
        SET batch_start = (current_batch - 1) * batch_size + 1;
        SET batch_end = LEAST(current_batch * batch_size, total_rows);

        INSERT INTO `{project_id}.{dataset_id}.{embedding_table_name}`
        (publication_number, country_code, pub_date, title_en, abstract_en, combined_text, text_embedding)

        WITH batch_data AS (
            SELECT
                publication_number,
                country_code,
                pub_date,
                title_en,
                abstract_en,
                combined_text as content
            FROM numbered_patents
            WHERE rn BETWEEN batch_start AND batch_end
        ),
        batch_embeddings AS (
            SELECT
                t.publication_number,
                t.country_code,
                t.pub_date,
                t.title_en,
                t.abstract_en,
                t.content as combined_text,
                t.ml_generate_embedding_result as text_embedding
            FROM ML.GENERATE_EMBEDDING(
                MODEL `{project_id}.{dataset_id}.text_embedding_model`,
                TABLE batch_data,
                STRUCT(TRUE AS flatten_json_output)
            ) AS t
        )
        SELECT * FROM batch_embeddings;

        SET batch_end_time = CURRENT_TIMESTAMP();
        
        INSERT INTO `{project_id}.{dataset_id}.{logs_table}`
        VALUES (
            CURRENT_TIMESTAMP(), 
            current_batch, 
            batch_start, 
            batch_end,
            batch_end - batch_start + 1,
            TIMESTAMP_DIFF(batch_end_time, batch_start_time, SECOND),
            FORMAT('BATCH_%d_OF_%d_COMPLETED', current_batch, num_batches)
        );

        SET current_batch = current_batch + 1;
    END WHILE;
    
    INSERT INTO `{project_id}.{dataset_id}.{logs_table}`
    VALUES (
        CURRENT_TIMESTAMP(), 
        999999,
        0, 
        0,
        total_rows,
        TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), proc_start_time, SECOND),
        'PROCEDURE_COMPLETED'
    );
END;    
"""

# Compute the average embeddings for a list of patents
query_patents_embedding_avgs = """
            WITH unnested AS (
                SELECT
                    pos,
                    val
                FROM `{project_id}.{dataset_id}.{table_name}`,
                UNNEST(text_embedding) AS val WITH OFFSET pos
                WHERE publication_number IN UNNEST(@patent_numbers)
                ),
                averaged AS (
                SELECT
                    pos,
                    AVG(val) AS avg_val
                FROM unnested
                GROUP BY pos
                )
                SELECT
                ARRAY_AGG(avg_val ORDER BY pos) AS avg_embedding
                FROM averaged;
            """

# Perform vector search for given query 
vector_search_query = """
            WITH query_embedding AS (
                SELECT @query_embeddings AS embedding
            )
            SELECT 
            base.publication_number,
            base.country_code,
            base.title_en,
            base.abstract_en,
            base.pub_date,
            distance,
            ROUND((1 - distance), 4) as cosine_score
            
            FROM VECTOR_SEARCH(
                    (
                    SELECT
                        publication_number,
                        country_code,
                        title_en,
                        abstract_en,
                        pub_date,
                        text_embedding
                    FROM `{project_id}.{dataset_id}.{table_name}`
                    WHERE 1=1
                    {filter_clause}
                    ),
                'text_embedding',
                TABLE query_embedding,
                'embedding',
                distance_type => 'COSINE',
                top_k => {top_k},
                options => '{options}'
            )
            ORDER BY distance
        """

# Create table to store flattened CPC codes which will be used in Streamlit to generate
# random patents for the demo
create_cpc_ddl = """
    CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{patents_cpc_table}` AS
    SELECT 
    a.publication_number,
    a.title_en,
    cpc_code
    FROM `{project_id}.{dataset_id}.{patents_source_table}` a,
    UNNEST(cpc_codes) AS cpc_code JOIN `{project_id}.{dataset_id}.{patents_embedding_table}` b
    ON a.publication_number = b.publication_number
"""

# Fetch random sample of patents using cpc_code subclass
patent_random_sample = """
            SELECT
            publication_number,
            title_en
            FROM `{project_id}.{dataset_id}.{table_name}` TABLESAMPLE SYSTEM (.05 PERCENT)
            WHERE cpc_code LIKE '{cpc_code}%'
            GROUP BY publication_number, title_en
            LIMIT {top_k}
"""

# Return a list of patents
patents_query = """
    SELECT publication_number, title_en, abstract_en, CONCAT(title_en, " ", abstract_en) AS combined_text
    FROM `{project_id}.{dataset_id}.{table_name}` 
    WHERE  1=1
    {filter_clause};
"""

# Latency measurement queries
create_latency_ddl = """
    CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.{latency_table}`
    (
        query STRING,
        test_type STRING,  -- 'vector_search' or 'explainability' or 'complete_pipeline'
        run_environment STRING,  -- 'laptop' or 'kaggle' or 'cloud',
        cache_hit BOOL,
        embedding_time_ms FLOAT64,
        search_time_ms FLOAT64,
        explainability_time_ms FLOAT64,  -- NEW: time for explainability step
        total_time_ms FLOAT64,
        results_count INT64,
        bytes_processed INT64,
        slot_millis INT64,
        min_similarity FLOAT64,
        avg_similarity FLOAT64,
        max_similarity FLOAT64,
        run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    )
    OPTIONS(
        description = "Latency test measurements for patent semantic search pipeline"
    )
"""

latency_analysis = """
    SELECT 
        test_type,
        run_environment,
        COUNT(*) as query_count,
        MIN(total_time_ms) as min_latency,
        APPROX_QUANTILES(total_time_ms, 100)[OFFSET(50)] as median_latency,
        AVG(total_time_ms) as mean_latency,
        MAX(total_time_ms) as max_latency,
        STDDEV(total_time_ms) as std_dev_latency,
        APPROX_QUANTILES(total_time_ms, 100)[OFFSET(90)] as p90_latency,
        APPROX_QUANTILES(total_time_ms, 100)[OFFSET(95)] as p95_latency,
        APPROX_QUANTILES(total_time_ms, 100)[OFFSET(99)] as p99_latency
    FROM `{project_id}.{dataset_id}.{latency_table}`
    GROUP BY test_type, run_environment
    ORDER BY test_type, run_environment

"""

### BQ partition pruning queries
create_paritioned_pruning_table =  """
    CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.{partition_pruned_table}`
    (
        test_query STRING,
        test_type STRING,  -- 'full_scan' or 'partition_pruned'
        run_environment STRING,
        filter_clause STRING,
        date_range_months INT64,  -- Number of months in date filter (0 for full scan)
        embedding_time_ms FLOAT64,
        search_time_ms FLOAT64,
        total_time_ms FLOAT64,
        results_count INT64,
        bytes_processed INT64,
        slot_millis INT64,
        cache_hit BOOLEAN,
        min_similarity FLOAT64,
        avg_similarity FLOAT64,
        max_similarity FLOAT64,
        run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    )
    OPTIONS(
        description = "Partition pruning efficiency test results"
    )
"""

# using table reference instead of select subquery; this query is 
# used in the vector index performance testing
vector_search_performance_query = """
            WITH query_embedding AS (
                SELECT @query_embeddings AS embedding
            )
            SELECT 
            base.publication_number,
            base.country_code,
            base.title_en,
            base.abstract_en,
            base.pub_date,
            distance,
            ROUND((1 - distance), 4) as cosine_score
            
            FROM VECTOR_SEARCH(
                TABLE `{project_id}.{dataset_id}.{table_name}`,
                'text_embedding',
                TABLE query_embedding,
                'embedding',
                distance_type => 'COSINE',
                top_k => {top_k},
                options => '{options}'
            )
            ORDER BY distance
        """

### BigQuery Vector index performance
create_vector_index_table = """
    CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.{vector_index_table}`
    (
        test_query STRING,
        search_method STRING,  -- 'brute_force', 'vector_index', 'index_creation'
        run_environment STRING,
        index_name STRING,
        index_usage_mode STRING,
        index_unused_reasons ARRAY<
                                    STRUCT<
                                        code STRING,
                                        message STRING,
                                            base_table STRUCT<
                                            project_id STRING,
                                            dataset_id STRING,
                                            table_id STRING
                                        >
                                    >
                                >,
        ivf_num_lists INT64,
        fraction_lists_searched FLOAT64,
        embedding_time_ms FLOAT64,
        search_time_ms FLOAT64,
        index_creation_time_ms FLOAT64,
        total_time_ms FLOAT64,
        results_count INT64,
        bytes_processed INT64,
        slot_millis INT64,
        cache_hit BOOLEAN,
        recall_vs_brute_force FLOAT64,  -- Accuracy compared to brute force
        min_similarity FLOAT64,
        avg_similarity FLOAT64,
        max_similarity FLOAT64,
        run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    )
    OPTIONS(
        description = "Vector index performance test results comparing brute force vs indexed search"
    )
"""

# Search comparison queries
create_search_comparison_ddl = """
    CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.{search_comparison_table}`
(
    query STRING,
    test_type STRING,  -- 'vector_search' or 'keyword_search'
    run_environment STRING,
    embedding_time_ms FLOAT64,
    search_time_ms FLOAT64,
    total_time_ms FLOAT64,
    total_bytes_processed INT64,
    semantic_results_count INT64,
    keyword_results_count INT64,
    overlap_count INT64,
    semantic_unique_count INT64,
    keyword_unique_count INT64,
    semantic_discovery_rate FLOAT64,
    run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
OPTIONS(
    description = "Search comparison test results between semantic and keyword search"
)
"""

# Metric analysis queries
latency_measurement_query = """
    SELECT 
        test_type,
        run_environment,
        COUNT(*) as query_count,
        MIN(total_time_ms) as min_latency,
        APPROX_QUANTILES(total_time_ms, 100)[OFFSET(50)] as median_latency,
        AVG(total_time_ms) as mean_latency,
        MAX(total_time_ms) as max_latency,
        STDDEV(total_time_ms) as std_dev_latency,
        APPROX_QUANTILES(total_time_ms, 100)[OFFSET(90)] as p90_latency,
        APPROX_QUANTILES(total_time_ms, 100)[OFFSET(95)] as p95_latency,
        APPROX_QUANTILES(total_time_ms, 100)[OFFSET(99)] as p99_latency
    FROM `{project_id}.{dataset_id}.{latency_table}`
    WHERE cache_hit = False
    GROUP BY test_type, run_environment
    ORDER BY test_type, run_environment;
"""

# Semantic versus Keyword search comparison analysis
semantic_vs_keyword_analysis_query = """
    SELECT 
        test_type,
        run_environment,
        COUNT(*) as query_count,
        AVG(total_time_ms) as avg_search_time_ms,
        AVG(semantic_results_count) as avg_semantic_results,
        AVG(keyword_results_count) as avg_keyword_results,
        AVG(overlap_count) as avg_overlap,
        AVG(semantic_unique_count) as avg_semantic_unique,
        AVG(keyword_unique_count) as avg_keyword_unique,
        AVG(semantic_discovery_rate) as avg_discovery_rate
FROM `{project_id}.{dataset_id}.{search_comparison_table}`
GROUP BY test_type, run_environment
ORDER BY test_type, run_environment;
"""

# Discovery rate analysis
discovery_rate_analysis_query = """
    SELECT 
        run_environment,
        test_type,
        AVG(total_time_ms) as avg_time_ms,
        SUM(semantic_results_count) as total_semantic_results,
        SUM(keyword_results_count) as total_keyword_results,
        SUM(overlap_count) as total_overlap,
        SUM(semantic_unique_count) as total_semantic_unique,
        SUM(keyword_unique_count) as total_keyword_unique,
        ROUND(SUM(semantic_unique_count) / SUM(semantic_results_count) * 100, 1) as overall_discovery_rate
FROM `{project_id}.{dataset_id}.{search_comparison_table}`
--WHERE test_type = 'keyword_search'
GROUP BY run_environment, test_type;
"""

# Compare efficiency across date ranges
efficiency_across_partition_query = """
    SELECT
        run_environment,
        test_type,
        ROUND(AVG(search_time_ms / 1000), 2) as avg_search_time_sec,
        ROUND(AVG(bytes_processed / 1024 / 1024 / 1024)) as avg_bytes_processed_gb,
        AVG(results_count) as avg_results
    FROM `{project_id}.{dataset_id}.{partition_pruned_table}`
    WHERE cache_hit = FALSE  -- Only non-cached results
    GROUP BY run_environment, test_type, date_range_months
    ORDER BY date_range_months;
"""

# Calculate bytes used and time reduction percentages
calculate_bytes_and_time_reduction_query = """
    WITH baseline AS (
    SELECT run_environment,
            AVG(bytes_processed) as full_scan_bytes,
            AVG(search_time_ms) as full_scan_time
    FROM `patents_semantic_search.partition_pruning_results`
    WHERE test_type = 'full_scan' AND cache_hit = FALSE
    GROUP BY run_environment
    )
    SELECT
        p.run_environment,
        p.test_type,
        ROUND((1 - AVG(p.bytes_processed) / b.full_scan_bytes) * 100, 1) as bytes_reduction_pct,
        ROUND((1 - AVG(p.search_time_ms) / b.full_scan_time) * 100, 1) as time_reduction_pct
    FROM `{project_id}.{dataset_id}.{partition_pruned_table}` p
    JOIN baseline b ON p.run_environment = b.run_environment
    WHERE p.cache_hit = FALSE
    GROUP BY p.run_environment, p.test_type, date_range_months, b.full_scan_bytes, b.full_scan_time
    ORDER BY date_range_months
"""