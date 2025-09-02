"""Contains various BigQuery sql queries"""
import os
import pandas as pd
import streamlit as st
# from dotenv import load_dotenv
# load_dotenv()

# project_id = os.getenv("project_id") or st.secrets["google"]["project_id"]
# dataset_id = os.getenv("dataset_id") or st.secrets["google"]["dataset_id"]


# Exploratory analysis queries
"""Dataset size"""
dataset_size_qry = """
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
"""

"""Country-wise breakdown of patent publications"""
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

"""Top country each month"""
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

"""Year on Year English-language publications growth rate"""
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

"""Year on Year growth rate of top 10 countries"""
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

"""Citation pattern by top countries"""
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

"""Top n CPCs"""
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

"""Technology area analysis by main CPC classes"""
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

"""Patent publication flow"""
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

# For Embedding extraction
"""Fetch patents for a given date range"""
extract_qry = """
    SELECT
        publication_number,
        country_code,
        pub_date,
        title_en,
        abstract_en,
        CONCAT(COALESCE(title_en, ''), ' ', COALESCE(abstract_en, '')) as combined_text
    FROM `{project_id}.{dataset_id}.patents_2017_2025_en`
    WHERE pub_date BETWEEN '{start_date}' AND '{end_date}'
        AND LENGTH(title_en) >= 30
        AND LENGTH(abstract_en) >= 100
    ORDER BY pub_date DESC
"""

create_embedding_ddl = """
    CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.{table_name}`
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
    CLUSTER BY country_code, publication_number
"""