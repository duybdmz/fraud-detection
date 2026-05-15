-- Databricks notebook source
-- MAGIC %md
-- MAGIC # 03 — BI & Data Warehousing
-- MAGIC
-- MAGIC Analytics queries trên `fraud_catalog.gold.gold_fraud_training`.

-- COMMAND ----------
USE CATALOG fraud_catalog;
USE SCHEMA gold;

-- COMMAND ----------
-- MAGIC %md ## 1. Transaction Volume & Fraud Rate by Month

SELECT
    DATE_FORMAT(ts, 'yyyy-MM')                        AS year_month,
    COUNT(*)                                           AS total_tx,
    SUM(is_fraud)                                      AS fraud_count,
    ROUND(SUM(is_fraud) / COUNT(*) * 100, 4)           AS fraud_rate_pct,
    ROUND(SUM(abs_amount), 2)                          AS total_amount,
    ROUND(AVG(abs_amount), 2)                          AS avg_amount
FROM gold_fraud_training
GROUP BY year_month
ORDER BY year_month;

-- COMMAND ----------
-- MAGIC %md ## 2. Fraud Rate by Transaction Type
-- MAGIC
-- MAGIC Swipe 52.4% | Chip 35.9% | Online 11.7%

SELECT
    use_chip,
    COUNT(*)                                           AS total_tx,
    SUM(is_fraud)                                      AS fraud_count,
    ROUND(SUM(is_fraud) / COUNT(*) * 100, 4)           AS fraud_rate_pct,
    ROUND(AVG(abs_amount), 2)                          AS avg_amount
FROM gold_fraud_training
GROUP BY use_chip
ORDER BY fraud_rate_pct DESC;

-- COMMAND ----------
-- MAGIC %md ## 3. Top 20 MCC Categories by Fraud Count

SELECT
    mcc,
    mcc_name,
    COUNT(*)                                           AS total_tx,
    SUM(is_fraud)                                      AS fraud_count,
    ROUND(SUM(is_fraud) / COUNT(*) * 100, 4)           AS fraud_rate_pct,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN abs_amount END), 2) AS avg_fraud_amount
FROM gold_fraud_training
WHERE is_fraud = 1
GROUP BY mcc, mcc_name
ORDER BY fraud_count DESC
LIMIT 20;

-- COMMAND ----------
-- MAGIC %md ## 4. Fraud Rate by Hour of Day

SELECT
    hour_of_day,
    COUNT(*)                                           AS total_tx,
    SUM(is_fraud)                                      AS fraud_count,
    ROUND(SUM(is_fraud) / COUNT(*) * 100, 4)           AS fraud_rate_pct
FROM gold_fraud_training
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- COMMAND ----------
-- MAGIC %md ## 5. Fraud by State (Top 15)

SELECT
    merchant_state,
    COUNT(*)                                           AS total_tx,
    SUM(is_fraud)                                      AS fraud_count,
    ROUND(SUM(is_fraud) / COUNT(*) * 100, 4)           AS fraud_rate_pct
FROM gold_fraud_training
WHERE merchant_state IS NOT NULL AND merchant_state != ''
GROUP BY merchant_state
ORDER BY fraud_count DESC
LIMIT 15;

-- COMMAND ----------
-- MAGIC %md ## 6. Amount Distribution: Fraud vs Normal

SELECT
    is_fraud,
    COUNT(*)                                           AS count,
    ROUND(MIN(abs_amount), 2)                          AS min_amount,
    ROUND(AVG(abs_amount), 2)                          AS avg_amount,
    ROUND(PERCENTILE(abs_amount, 0.5), 2)              AS median_amount,
    ROUND(PERCENTILE(abs_amount, 0.95), 2)             AS p95_amount,
    ROUND(MAX(abs_amount), 2)                          AS max_amount
FROM gold_fraud_training
GROUP BY is_fraud;

-- COMMAND ----------
-- MAGIC %md ## 7. Error Type vs Fraud Rate

SELECT
    COALESCE(errors, 'No Error')                       AS error_type,
    COUNT(*)                                           AS total_tx,
    SUM(is_fraud)                                      AS fraud_count,
    ROUND(SUM(is_fraud) / COUNT(*) * 100, 4)           AS fraud_rate_pct
FROM gold_fraud_training
GROUP BY error_type
ORDER BY fraud_rate_pct DESC;

-- COMMAND ----------
-- MAGIC %md ## 8. High-Risk Customers

SELECT
    client_id,
    COUNT(*)                                           AS total_tx,
    SUM(is_fraud)                                      AS fraud_count,
    ROUND(SUM(is_fraud) / COUNT(*) * 100, 2)           AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN abs_amount ELSE 0 END), 2) AS total_fraud_amount
FROM gold_fraud_training
GROUP BY client_id
HAVING fraud_count > 0
ORDER BY fraud_count DESC
LIMIT 20;

-- COMMAND ----------
-- MAGIC %md ## 9. Card Brand vs Fraud Rate

SELECT
    card_brand,
    card_type,
    COUNT(*)                                           AS total_tx,
    SUM(is_fraud)                                      AS fraud_count,
    ROUND(SUM(is_fraud) / COUNT(*) * 100, 4)           AS fraud_rate_pct
FROM gold_fraud_training
GROUP BY card_brand, card_type
ORDER BY fraud_rate_pct DESC;

-- COMMAND ----------
-- MAGIC %md ## 10. Refund vs Fraud Correlation

SELECT
    is_refund,
    COUNT(*)                                           AS total_tx,
    SUM(is_fraud)                                      AS fraud_count,
    ROUND(SUM(is_fraud) / COUNT(*) * 100, 4)           AS fraud_rate_pct
FROM gold_fraud_training
GROUP BY is_refund;
