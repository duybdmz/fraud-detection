# Databricks notebook source

CATALOG    = "fraud_catalog"
S3_BUCKET  = "fraud-demo-434481793703"
RAW_BASE   = f"s3://{S3_BUCKET}/raw"
MODEL_NAME = f"{CATALOG}.ml.fraud_classifier"
ENDPOINT   = "fraud-detection-endpoint"

# Feature columns — match training and serving schema exactly
FEATURE_COLS = [
    # transaction
    "abs_amount", "is_refund",
    "is_online", "is_chip", "is_swipe",
    "mcc", "has_error",
    "hour_of_day", "day_of_week", "month",
    # card
    "has_chip_flag", "credit_limit",
    "num_cards_issued", "year_pin_last_changed",
    "is_dark_web",
    # user
    "current_age",
    "yearly_income", "total_debt", "per_capita_income",
    "credit_score", "num_credit_cards",
    "debt_to_income",
    # rolling windows
    "tx_count_1h",  "tx_amount_1h",
    "tx_count_24h", "tx_amount_24h", "avg_amount_24h",
    "std_amount_24h", "unique_merchants_24h", "unique_states_24h",
    "tx_count_7d",  "tx_amount_7d",
]

print(f"Catalog  : {CATALOG}")
print(f"S3       : s3://{S3_BUCKET}")
print(f"Model    : {MODEL_NAME}")
print(f"Endpoint : {ENDPOINT}")
print(f"Features : {len(FEATURE_COLS)}")
