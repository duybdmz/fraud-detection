-- Databricks notebook source
-- MAGIC %md
-- MAGIC # 02 — Unity Catalog: Data Governance & ACLs

-- COMMAND ----------
-- MAGIC %md ## Create Catalog & Schemas

CREATE CATALOG IF NOT EXISTS fraud_catalog;
USE CATALOG fraud_catalog;

CREATE SCHEMA IF NOT EXISTS fraud_catalog.bronze;
CREATE SCHEMA IF NOT EXISTS fraud_catalog.silver;
CREATE SCHEMA IF NOT EXISTS fraud_catalog.gold;
CREATE SCHEMA IF NOT EXISTS fraud_catalog.ml;

-- COMMAND ----------
-- MAGIC %md ## Grant Permissions by Role

-- data-engineers: full access bronze/silver
GRANT USE CATALOG ON CATALOG fraud_catalog TO `data-engineers`;
GRANT USE SCHEMA, SELECT, MODIFY ON SCHEMA fraud_catalog.bronze TO `data-engineers`;
GRANT USE SCHEMA, SELECT, MODIFY ON SCHEMA fraud_catalog.silver TO `data-engineers`;

-- ml-team: read gold, write ml
GRANT USE SCHEMA, SELECT ON SCHEMA fraud_catalog.gold TO `ml-team`;
GRANT USE SCHEMA, SELECT, MODIFY ON SCHEMA fraud_catalog.ml TO `ml-team`;

-- analysts: read gold only
GRANT USE SCHEMA, SELECT ON SCHEMA fraud_catalog.gold TO `analysts`;

-- model-serving: read ml for inference
GRANT USE SCHEMA, SELECT ON SCHEMA fraud_catalog.ml TO `model-serving`;

-- COMMAND ----------
-- MAGIC %md ## Tag Tables for Discovery

ALTER TABLE fraud_catalog.gold.gold_fraud_features
SET TAGS ('domain' = 'fraud-detection', 'layer' = 'gold', 'use' = 'analytics');

ALTER TABLE fraud_catalog.gold.gold_fraud_training
SET TAGS ('domain' = 'fraud-detection', 'layer' = 'gold', 'use' = 'ml-training');

-- COMMAND ----------
-- MAGIC %md ## View Data Lineage

SELECT
    source_table_full_name,
    target_table_full_name,
    created_at
FROM system.access.table_lineage
WHERE target_table_full_name LIKE 'fraud_catalog.%'
ORDER BY created_at DESC
LIMIT 20;

-- COMMAND ----------
-- MAGIC %md ## Verify Table Counts

SELECT 'bronze_transactions' AS tbl, COUNT(*) AS rows FROM fraud_catalog.bronze.bronze_transactions
UNION ALL
SELECT 'bronze_cards',        COUNT(*) FROM fraud_catalog.bronze.bronze_cards
UNION ALL
SELECT 'bronze_users',        COUNT(*) FROM fraud_catalog.bronze.bronze_users
UNION ALL
SELECT 'bronze_fraud_labels', COUNT(*) FROM fraud_catalog.bronze.bronze_fraud_labels
UNION ALL
SELECT 'bronze_mcc_codes',    COUNT(*) FROM fraud_catalog.bronze.bronze_mcc_codes
UNION ALL
SELECT 'silver_transactions', COUNT(*) FROM fraud_catalog.silver.silver_transactions
UNION ALL
SELECT 'gold_fraud_features', COUNT(*) FROM fraud_catalog.gold.gold_fraud_features
UNION ALL
SELECT 'gold_fraud_training', COUNT(*) FROM fraud_catalog.gold.gold_fraud_training;
