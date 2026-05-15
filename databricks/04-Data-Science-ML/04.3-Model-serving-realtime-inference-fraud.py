# Databricks notebook source
# MAGIC %md
# MAGIC # 04.3 — Model Serving & Realtime Inference
# MAGIC
# MAGIC Deploy champion model lên Databricks Model Serving.
# MAGIC Output: `fraud_score` (float 0–1) — không block giao dịch.

# COMMAND ----------
# MAGIC %run ../config

# COMMAND ----------
import mlflow
import requests
import json
import time

mlflow.set_registry_uri("databricks-uc")

DATABRICKS_HOST  = spark.conf.get("spark.databricks.workspaceUrl")
DATABRICKS_TOKEN = (
    dbutils.notebook.entry_point
    .getDbutils().notebook().getContext()
    .apiToken().get()
)
HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type":  "application/json",
}

# COMMAND ----------
# MAGIC %md ## Get Champion / Challenger

client   = mlflow.MlflowClient()
champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
print(f"Champion: {MODEL_NAME} v{champion.version}")

try:
    challenger = client.get_model_version_by_alias(MODEL_NAME, "challenger")
    print(f"Challenger: v{challenger.version}")
except mlflow.exceptions.MlflowException:
    challenger = None
    print("No challenger — deploying champion at 100% traffic")

# COMMAND ----------
# MAGIC %md ## Create / Update Endpoint

served_models = [{
    "name":                  "champion",
    "model_name":            MODEL_NAME,
    "model_version":         champion.version,
    "workload_size":         "Small",
    "scale_to_zero_enabled": True,
    "traffic_percentage":    100 if not challenger else 90,
}]

if challenger:
    served_models.append({
        "name":                  "challenger",
        "model_name":            MODEL_NAME,
        "model_version":         challenger.version,
        "workload_size":         "Small",
        "scale_to_zero_enabled": True,
        "traffic_percentage":    10,
    })

existing = [
    e["name"] for e in
    requests.get(f"https://{DATABRICKS_HOST}/api/2.0/serving-endpoints", headers=HEADERS)
    .json().get("endpoints", [])
]

if ENDPOINT in existing:
    resp = requests.put(
        f"https://{DATABRICKS_HOST}/api/2.0/serving-endpoints/{ENDPOINT}/config",
        headers=HEADERS,
        json={"served_models": served_models},
    )
    print(f"Updated '{ENDPOINT}'  →  {resp.status_code}")
else:
    resp = requests.post(
        f"https://{DATABRICKS_HOST}/api/2.0/serving-endpoints",
        headers=HEADERS,
        json={"name": ENDPOINT, "config": {"served_models": served_models}},
    )
    print(f"Created '{ENDPOINT}'  →  {resp.status_code}")

# COMMAND ----------
# MAGIC %md ## Wait for READY

for _ in range(30):
    state = (
        requests.get(f"https://{DATABRICKS_HOST}/api/2.0/serving-endpoints/{ENDPOINT}", headers=HEADERS)
        .json().get("state", {}).get("ready", "NOT_READY")
    )
    print(f"State: {state}")
    if state == "READY":
        break
    time.sleep(30)

# COMMAND ----------
# MAGIC %md ## Test Inference
# MAGIC
# MAGIC Payload dùng đúng FEATURE_COLS từ config — phải match training schema.
# MAGIC
# MAGIC Sample dựa trên thống kê thực tế của dataset:
# MAGIC - Normal: amount ~$29 (median), Chip, grocery store, giờ bình thường
# MAGIC - High-risk: amount $9,500, Online, 2am, nhiều giao dịch trong 1h

normal_tx = {
    "abs_amount": 28.99,    "is_refund": 0,
    "is_online": 0,         "is_chip": 1,           "is_swipe": 0,
    "mcc": 5411,            "has_error": 0,          # Grocery store
    "hour_of_day": 14,      "day_of_week": 3,        "month": 6,
    "has_chip_flag": 1,     "credit_limit": 12594.0,
    "num_cards_issued": 2,  "year_pin_last_changed": 2018,  "is_dark_web": 0,
    "current_age": 45,      "yearly_income": 59696.0,
    "total_debt": 12000.0,  "per_capita_income": 29278.0,
    "credit_score": 710,    "num_credit_cards": 3,   "debt_to_income": 0.20,
    "tx_count_1h": 1,       "tx_amount_1h": 28.99,
    "tx_count_24h": 3,      "tx_amount_24h": 85.0,   "avg_amount_24h": 28.33,
    "std_amount_24h": 5.2,  "unique_merchants_24h": 3, "unique_states_24h": 1,
    "tx_count_7d": 12,      "tx_amount_7d": 340.0,
}

high_risk_tx = {
    "abs_amount": 9500.0,   "is_refund": 0,
    "is_online": 1,         "is_chip": 0,            "is_swipe": 0,
    "mcc": 5732,            "has_error": 1,           # Electronics, Bad PIN
    "hour_of_day": 2,       "day_of_week": 7,         "month": 11,
    "has_chip_flag": 1,     "credit_limit": 12000.0,
    "num_cards_issued": 1,  "year_pin_last_changed": 2010,  "is_dark_web": 0,
    "current_age": 32,      "yearly_income": 45000.0,
    "total_debt": 38000.0,  "per_capita_income": 22000.0,
    "credit_score": 580,    "num_credit_cards": 5,   "debt_to_income": 0.84,
    "tx_count_1h": 8,       "tx_amount_1h": 42000.0,
    "tx_count_24h": 15,     "tx_amount_24h": 68000.0, "avg_amount_24h": 4533.0,
    "std_amount_24h": 3200.0, "unique_merchants_24h": 12, "unique_states_24h": 5,
    "tx_count_7d": 22,      "tx_amount_7d": 95000.0,
}

resp = requests.post(
    f"https://{DATABRICKS_HOST}/serving-endpoints/{ENDPOINT}/invocations",
    headers=HEADERS,
    json={"dataframe_records": [normal_tx, high_risk_tx]},
)

for label, score in zip(["normal", "high_risk"], resp.json()["predictions"]):
    s = score if isinstance(score, float) else score.get("fraud_probability", score)
    print(f"[{label:10s}]  fraud_score={s:.4f}  →  {'FRAUD' if s > 0.5 else 'NORMAL'}")
