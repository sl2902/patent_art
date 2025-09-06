-- Compare efficiency across date ranges
SELECT 
  test_type,
  AVG(search_time_ms) as avg_search_time,
  AVG(bytes_processed) as avg_bytes_processed,
  AVG(results_count) as avg_results
FROM `your_project.your_dataset.partition_pruning_results`
WHERE cache_hit = FALSE  -- Only non-cached results
GROUP BY test_type
ORDER BY date_range_months;

-- Calculate reduction percentages
WITH baseline AS (
  SELECT AVG(bytes_processed) as full_scan_bytes,
         AVG(search_time_ms) as full_scan_time
  FROM `your_project.your_dataset.partition_pruning_results`
  WHERE test_type = 'full_scan' AND cache_hit = FALSE
)
SELECT 
  p.test_type,
  ROUND((1 - AVG(p.bytes_processed) / b.full_scan_bytes) * 100, 1) as bytes_reduction_pct,
  ROUND((1 - AVG(p.search_time_ms) / b.full_scan_time) * 100, 1) as time_reduction_pct
FROM `your_project.your_dataset.partition_pruning_results` p
CROSS JOIN baseline b
WHERE p.cache_hit = FALSE
GROUP BY p.test_type, b.full_scan_bytes, b.full_scan_time;