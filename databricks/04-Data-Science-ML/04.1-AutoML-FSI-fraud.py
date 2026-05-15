# Databricks notebook source
# MAGIC %md
# MAGIC # 04.1 — AutoML Baseline
# MAGIC
# MAGIC Databricks AutoML tự xử lý: class imbalance, feature encoding, hyperparameter tuning.
# MAGIC Fraud rate 0.15% → AutoML dùng `primary_metric="f1"` thay vì accuracy.

# COMMAND ----------
# MAGIC %run ../config

# COMMAND ----------
import databricks.automl as automl
from pyspark.sql import functions as F

# COMMAND ----------
# MAGIC %md ## Load Gold Training Table

df = spark.table(f"{CATALOG}.gold.gold_fraud_training")

total = df.count()
fraud = df.filter("is_fraud = 1").count()
print(f"Total rows : {total:,}")
print(f"Fraud      : {fraud:,}  ({fraud/total*100:.3f}%)")
print(f"Normal     : {total-fraud:,}  ({(total-fraud)/total*100:.3f}%)")

# COMMAND ----------
# MAGIC %md ## Run AutoML
# MAGIC
# MAGIC Exclude columns không available tại scoring time:
# MAGIC - IDs: transaction_id, client_id, card_id, merchant_id
# MAGIC - Raw strings đã được encode: use_chip, amount, errors, has_chip, card_on_dark_web
# MAGIC - Không dùng cho model: mcc_name, merchant_city, merchant_state, zip, gender, lat/lon

EXCLUDE_COLS = [
    "transaction_id", "ts", "client_id", "card_id", "merchant_id",
    "use_chip", "amount", "errors", "has_chip", "card_on_dark_web",
    "mcc_name", "merchant_city", "merchant_state", "zip",
    "gender", "latitude", "longitude",
]

summary = automl.classify(
    dataset=df,
    target_col="is_fraud",
    primary_metric="f1",
    timeout_minutes=60,
    exclude_cols=EXCLUDE_COLS,
    experiment_name="/fraud-demo/automl-baseline",
)

print(f"Best trial : {summary.best_trial.model_path}")
print(f"F1 score   : {summary.best_trial.metrics['val_f1_score']:.4f}")
print(f"Experiment : {summary.experiment.experiment_id}")

# COMMAND ----------
# MAGIC %md ## Register Best Model as Challenger

import mlflow
mlflow.set_registry_uri("databricks-uc")

best_run_id = summary.best_trial.mlflow_run_id

mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name=MODEL_NAME,
)

client   = mlflow.MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest   = max(int(v.version) for v in versions)
client.set_registered_model_alias(MODEL_NAME, "challenger", latest)

print(f"Registered: {MODEL_NAME} v{latest} → alias 'challenger'")
print("Next: run 04.4 to train XGBoost champion")
