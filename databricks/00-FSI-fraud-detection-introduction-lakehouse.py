# Databricks notebook source
# MAGIC %md
# MAGIC # Fraud Detection — Lakehouse on AWS + Databricks
# MAGIC
# MAGIC ## Dataset
# MAGIC
# MAGIC | File | Rows | Mô tả |
# MAGIC |------|------|-------|
# MAGIC | `transactions_data.csv` | 13,305,915 | Lịch sử giao dịch 2010–2019 |
# MAGIC | `cards_data.csv` | 6,146 | Thông tin thẻ (Visa 38%, Mastercard 52%) |
# MAGIC | `users_data.csv` | 2,000 | Thông tin khách hàng (avg age 45, avg credit score 710) |
# MAGIC | `train_fraud_labels.json` | 8,914,963 | Nhãn fraud — 0.15% fraud rate |
# MAGIC | `mcc_codes.json` | 109 | Merchant category lookup |
# MAGIC
# MAGIC **Class imbalance:** 1 fraud / 668 giao dịch bình thường → `scale_pos_weight ≈ 668`
# MAGIC
# MAGIC **Unlabeled:** ~4.4M transactions không có label — không dùng để train, dùng cho scoring/demo.
# MAGIC
# MAGIC ## Architecture
# MAGIC
# MAGIC ```
# MAGIC S3 Raw Data (5 files)
# MAGIC         ↓
# MAGIC Lakeflow Pipeline (Python DLT)
# MAGIC         ↓
# MAGIC bronze_transactions / bronze_cards / bronze_users / bronze_fraud_labels / bronze_mcc_codes
# MAGIC         ↓
# MAGIC silver_transactions  (cleaned + joined)
# MAGIC         ↓
# MAGIC gold_fraud_features  (31 features + rolling windows 1h/24h/7d)
# MAGIC gold_fraud_training  (labeled only — 8.9M rows)
# MAGIC         ↓
# MAGIC AutoML baseline  →  XGBoost champion
# MAGIC         ↓
# MAGIC MLflow  →  Unity Catalog Model Registry
# MAGIC         ↓
# MAGIC Databricks Model Serving  (champion 90% / challenger 10%)
# MAGIC ```
# MAGIC
# MAGIC ## Run Order
# MAGIC
# MAGIC | Step | Notebook | Mô tả |
# MAGIC |------|----------|-------|
# MAGIC | 1 | `config` | Shared variables |
# MAGIC | 2 | `01-Data-ingestion/01.1-dlt-python` | Lakeflow pipeline Bronze→Silver→Gold |
# MAGIC | 3 | `02-Data-governance` | Unity Catalog ACLs + lineage |
# MAGIC | 4 | `04-Data-Science-ML/04.1-AutoML` | AutoML baseline |
# MAGIC | 5 | `04-Data-Science-ML/04.4-XGBoost` | XGBoost + imbalance handling |
# MAGIC | 6 | `04-Data-Science-ML/04.3-Model-serving` | Deploy endpoint |
# MAGIC | 7 | `04-Data-Science-ML/04.5-AB-testing` | A/B traffic analysis |
# MAGIC | 8 | `06-Workflow-orchestration` | Orchestrate toàn bộ pipeline |

# COMMAND ----------
# MAGIC %run ./config
