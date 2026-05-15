# Databricks notebook source
# DLT pipeline — Bronze layer
# Deploy as a Lakeflow Spark Declarative Pipeline, NOT run directly.

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    LongType, IntegerType, DoubleType, StringType, MapType,
)

S3_BUCKET = spark.conf.get("pipeline.s3_bucket", "fraud-demo-434481793703")
RAW_BASE  = f"s3://{S3_BUCKET}/raw"

# COMMAND ----------
# MAGIC %md ## Bronze — Transactions
# MAGIC
# MAGIC Source: `transactions_data.csv` — 13,305,915 rows
# MAGIC
# MAGIC | Column | Type | Note |
# MAGIC |--------|------|------|
# MAGIC | id | BIGINT | Transaction ID |
# MAGIC | date | STRING | "2010-01-01 00:01:00" |
# MAGIC | client_id | BIGINT | |
# MAGIC | card_id | BIGINT | |
# MAGIC | amount | STRING | "$42.98" hoặc "$-77.00" — cleaned in silver |
# MAGIC | use_chip | STRING | "Swipe Transaction" / "Chip Transaction" / "Online Transaction" |
# MAGIC | merchant_id | BIGINT | |
# MAGIC | merchant_city | STRING | |
# MAGIC | merchant_state | STRING | |
# MAGIC | zip | STRING | |
# MAGIC | mcc | INT | Merchant Category Code |
# MAGIC | errors | STRING | null = no error |

TX_SCHEMA = StructType([
    StructField("id",             LongType(),    True),
    StructField("date",           StringType(),  True),
    StructField("client_id",      LongType(),    True),
    StructField("card_id",        LongType(),    True),
    StructField("amount",         StringType(),  True),
    StructField("use_chip",       StringType(),  True),
    StructField("merchant_id",    LongType(),    True),
    StructField("merchant_city",  StringType(),  True),
    StructField("merchant_state", StringType(),  True),
    StructField("zip",            StringType(),  True),
    StructField("mcc",            IntegerType(), True),
    StructField("errors",         StringType(),  True),
])

@dp.table(
    name="bronze_transactions",
    comment="Raw transactions from S3 — 13.3M rows, 2010-2019",
    table_properties={"quality": "bronze"},
)
def bronze_transactions():
    return (
        spark.read
        .option("header", True)
        .schema(TX_SCHEMA)
        .csv(f"{RAW_BASE}/transactions/")
        .withColumn("_ingestion_timestamp", F.current_timestamp())
        .withColumn("_source_file", F.input_file_name())
    )

# COMMAND ----------
# MAGIC %md ## Bronze — Cards
# MAGIC
# MAGIC Source: `cards_data.csv` — 6,146 rows
# MAGIC
# MAGIC | Column | Type | Note |
# MAGIC |--------|------|------|
# MAGIC | id | BIGINT | Card ID |
# MAGIC | client_id | BIGINT | |
# MAGIC | card_brand | STRING | Visa / Mastercard / Discover / Amex |
# MAGIC | card_type | STRING | Debit / Credit / Debit (Prepaid) |
# MAGIC | has_chip | STRING | "YES" / "NO" |
# MAGIC | credit_limit | STRING | "$14,347" — cleaned in silver |
# MAGIC | year_pin_last_changed | INT | |
# MAGIC | card_on_dark_web | STRING | "Yes" / "No" — 0% in this dataset |

CARDS_SCHEMA = StructType([
    StructField("id",                    LongType(),    True),
    StructField("client_id",             LongType(),    True),
    StructField("card_brand",            StringType(),  True),
    StructField("card_type",             StringType(),  True),
    StructField("card_number",           StringType(),  True),
    StructField("expires",               StringType(),  True),
    StructField("cvv",                   StringType(),  True),
    StructField("has_chip",              StringType(),  True),
    StructField("num_cards_issued",      IntegerType(), True),
    StructField("credit_limit",          StringType(),  True),
    StructField("acct_open_date",        StringType(),  True),
    StructField("year_pin_last_changed", IntegerType(), True),
    StructField("card_on_dark_web",      StringType(),  True),
])

@dp.table(
    name="bronze_cards",
    comment="Raw card profiles — 6,146 rows (Visa 38%, Mastercard 52%, Discover 3%, Amex 7%)",
    table_properties={"quality": "bronze"},
)
def bronze_cards():
    return (
        spark.read
        .option("header", True)
        .schema(CARDS_SCHEMA)
        .csv(f"{RAW_BASE}/cards/")
        .withColumn("_ingestion_timestamp", F.current_timestamp())
    )

# COMMAND ----------
# MAGIC %md ## Bronze — Users
# MAGIC
# MAGIC Source: `users_data.csv` — 2,000 rows
# MAGIC
# MAGIC | Column | Type | Note |
# MAGIC |--------|------|------|
# MAGIC | id | BIGINT | User ID |
# MAGIC | current_age | INT | 18–101, avg 45 |
# MAGIC | gender | STRING | Male / Female |
# MAGIC | yearly_income | STRING | "$59,696" — cleaned in silver |
# MAGIC | total_debt | STRING | "$127,613" — cleaned in silver |
# MAGIC | credit_score | INT | 480–850, avg 710 |

USERS_SCHEMA = StructType([
    StructField("id",                LongType(),    True),
    StructField("current_age",       IntegerType(), True),
    StructField("retirement_age",    IntegerType(), True),
    StructField("birth_year",        IntegerType(), True),
    StructField("birth_month",       IntegerType(), True),
    StructField("gender",            StringType(),  True),
    StructField("address",           StringType(),  True),
    StructField("latitude",          DoubleType(),  True),
    StructField("longitude",         DoubleType(),  True),
    StructField("per_capita_income", StringType(),  True),
    StructField("yearly_income",     StringType(),  True),
    StructField("total_debt",        StringType(),  True),
    StructField("credit_score",      IntegerType(), True),
    StructField("num_credit_cards",  IntegerType(), True),
])

@dp.table(
    name="bronze_users",
    comment="Raw user profiles — 2,000 users, avg age 45, avg credit score 710",
    table_properties={"quality": "bronze"},
)
def bronze_users():
    return (
        spark.read
        .option("header", True)
        .schema(USERS_SCHEMA)
        .csv(f"{RAW_BASE}/users/")
        .withColumn("_ingestion_timestamp", F.current_timestamp())
    )

# COMMAND ----------
# MAGIC %md ## Bronze — Fraud Labels
# MAGIC
# MAGIC Source: `train_fraud_labels.json` — 8,914,963 labeled transactions
# MAGIC
# MAGIC Format: `{"target": {"transaction_id": "Yes/No", ...}}`
# MAGIC
# MAGIC | Label | Count | % |
# MAGIC |-------|-------|---|
# MAGIC | No (normal) | 8,901,631 | 99.85% |
# MAGIC | Yes (fraud) | 13,332 | 0.15% |
# MAGIC
# MAGIC ~4.4M transactions không có label (unlabeled test set) — không dùng để train.

@dp.table(
    name="bronze_fraud_labels",
    comment="Fraud labels — 8.9M labeled, 0.15% fraud rate",
    table_properties={"quality": "bronze"},
)
def bronze_fraud_labels():
    return (
        spark.read
        .option("wholetext", True)
        .text(f"{RAW_BASE}/fraud_labels/")
        .select(
            F.explode(
                F.from_json(
                    F.get_json_object(F.col("value"), "$.target"),
                    MapType(StringType(), StringType()),
                )
            ).alias("transaction_id", "label")
        )
        .withColumn("transaction_id", F.col("transaction_id").cast(LongType()))
        .withColumn("_ingestion_timestamp", F.current_timestamp())
    )

# COMMAND ----------
# MAGIC %md ## Bronze — MCC Codes
# MAGIC
# MAGIC Source: `mcc_codes.json` — 109 merchant categories
# MAGIC
# MAGIC Format: `{"5812": "Eating Places and Restaurants", ...}` — flat dict

@dp.table(
    name="bronze_mcc_codes",
    comment="Merchant category code lookup — 109 categories",
    table_properties={"quality": "bronze"},
)
def bronze_mcc_codes():
    return (
        spark.read
        .option("wholetext", True)
        .text(f"{RAW_BASE}/mcc_codes/")
        .select(
            F.explode(
                F.from_json(F.col("value"), MapType(StringType(), StringType()))
            ).alias("mcc_code", "mcc_name")
        )
        .withColumn("mcc_code", F.col("mcc_code").cast(IntegerType()))
        .withColumn("_ingestion_timestamp", F.current_timestamp())
    )
