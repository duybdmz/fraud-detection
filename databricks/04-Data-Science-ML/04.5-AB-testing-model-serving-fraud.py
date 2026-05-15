# Databricks notebook source
# MAGIC %md
# MAGIC # 04.5 — A/B Testing: Champion vs Challenger
# MAGIC
# MAGIC So sánh XGBoost champion (04.4) vs LightGBM challenger (04.2)
# MAGIC dựa trên inference logs từ `system.serving.request_logs`.

# COMMAND ----------
# MAGIC %run ../config

# COMMAND ----------
import mlflow

mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

# COMMAND ----------
# MAGIC %md ## Set Champion / Challenger Aliases
# MAGIC
# MAGIC Chạy cell này để cập nhật traffic split khi có model version mới.

versions     = client.search_model_versions(f"name='{MODEL_NAME}'")
all_versions = sorted([int(v.version) for v in versions])

if len(all_versions) >= 2:
    champion_v   = all_versions[-2]
    challenger_v = all_versions[-1]
    client.set_registered_model_alias(MODEL_NAME, "champion",   champion_v)
    client.set_registered_model_alias(MODEL_NAME, "challenger", challenger_v)
    print(f"Champion:   v{champion_v}")
    print(f"Challenger: v{challenger_v}")
    print("Re-run 04.3 to update endpoint traffic split (90/10)")
else:
    print("Need at least 2 model versions for A/B testing")

# COMMAND ----------
# MAGIC %md ## Compare Metrics via Inference Logs
# MAGIC
# MAGIC `system.serving.request_logs` — Databricks tự log mọi request/response.
# MAGIC Không cần tự implement logging.

# COMMAND ----------
# MAGIC %sql
SELECT
    model_name,
    model_version,
    COUNT(*)                                                          AS total_requests,
    ROUND(AVG(CAST(response:predictions[0] AS DOUBLE)), 4)            AS avg_fraud_score,
    SUM(CASE WHEN CAST(response:predictions[0] AS DOUBLE) > 0.5
             THEN 1 ELSE 0 END)                                       AS fraud_flagged,
    ROUND(SUM(CASE WHEN CAST(response:predictions[0] AS DOUBLE) > 0.5
                   THEN 1 ELSE 0 END) / COUNT(*) * 100, 4)            AS fraud_flag_rate_pct,
    ROUND(AVG(execution_duration_ms), 1)                              AS avg_latency_ms,
    ROUND(PERCENTILE(execution_duration_ms, 0.95), 1)                 AS p95_latency_ms
FROM system.serving.request_logs
WHERE endpoint_name = 'fraud-detection-endpoint'
  AND timestamp >= CURRENT_TIMESTAMP - INTERVAL 7 DAYS
GROUP BY model_name, model_version
ORDER BY model_version;

-- COMMAND ----------
-- MAGIC %md ## Score Distribution by Model Version

-- COMMAND ----------
-- MAGIC %sql
SELECT
    model_version,
    CASE
        WHEN CAST(response:predictions[0] AS DOUBLE) < 0.1 THEN '0.0–0.1'
        WHEN CAST(response:predictions[0] AS DOUBLE) < 0.3 THEN '0.1–0.3'
        WHEN CAST(response:predictions[0] AS DOUBLE) < 0.5 THEN '0.3–0.5'
        WHEN CAST(response:predictions[0] AS DOUBLE) < 0.7 THEN '0.5–0.7'
        WHEN CAST(response:predictions[0] AS DOUBLE) < 0.9 THEN '0.7–0.9'
        ELSE '0.9–1.0'
    END AS score_bucket,
    COUNT(*) AS count
FROM system.serving.request_logs
WHERE endpoint_name = 'fraud-detection-endpoint'
  AND timestamp >= CURRENT_TIMESTAMP - INTERVAL 7 DAYS
GROUP BY model_version, score_bucket
ORDER BY model_version, score_bucket;

# COMMAND ----------
# MAGIC %md ## Promote Challenger to Champion (manual gate)
# MAGIC
# MAGIC Uncomment và chạy sau khi review metrics và quyết định challenger tốt hơn.

# challenger_v = int(client.get_model_version_by_alias(MODEL_NAME, "challenger").version)
# client.set_registered_model_alias(MODEL_NAME, "champion", challenger_v)
# client.delete_registered_model_alias(MODEL_NAME, "challenger")
# print(f"v{challenger_v} promoted to champion — re-run 04.3 to update endpoint")
