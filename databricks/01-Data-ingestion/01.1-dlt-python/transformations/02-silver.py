# Databricks notebook source
# DLT pipeline — Silver layer
# Must be in the same Lakeflow pipeline as 01-bronze.py

from pyspark import pipelines as dp
from pyspark.sql import functions as F

def strip_currency(col_name):
    """Strip '$' and ',' then cast to double. Handles: '$42.98', '$-77.00', '$127,613'"""
    return F.regexp_replace(F.col(col_name), r"[\$,]", "").cast("double")

# COMMAND ----------
# MAGIC %md ## Silver — Transactions
# MAGIC
# MAGIC Join 5 bronze tables thành 1 enriched table.
# MAGIC
# MAGIC **Data quality rules:**
# MAGIC - DROP rows thiếu transaction_id, card_id, client_id
# MAGIC - WARN (giữ lại) rows thiếu amount — có thể là refund edge case
# MAGIC
# MAGIC **Xử lý đặc biệt:**
# MAGIC - `amount`: strip `$` → double (âm = refund, ~660K rows)
# MAGIC - `errors`: empty string → null
# MAGIC - `is_fraud`: null = unlabeled (~4.4M rows) — giữ lại, không drop

@dp.table(
    name="silver_transactions",
    comment="Cleaned + enriched transactions — joined with cards, users, MCC, labels",
    table_properties={"quality": "silver"},
)
@dp.expect_or_drop("valid_transaction_id", "id IS NOT NULL")
@dp.expect_or_drop("valid_card_id",        "card_id IS NOT NULL")
@dp.expect_or_drop("valid_client_id",      "client_id IS NOT NULL")
@dp.expect("valid_amount",                 "amount IS NOT NULL")
def silver_transactions():
    tx     = dp.read("bronze_transactions")
    cards  = dp.read("bronze_cards")
    users  = dp.read("bronze_users")
    mcc    = dp.read("bronze_mcc_codes").withColumnRenamed("mcc_code", "_mcc_code")
    labels = dp.read("bronze_fraud_labels")

    cards_clean = cards.select(
        F.col("id").alias("_card_id"),
        "card_brand", "card_type", "has_chip",
        strip_currency("credit_limit").alias("credit_limit"),
        "num_cards_issued",
        "year_pin_last_changed",
        "card_on_dark_web",
    )

    users_clean = users.select(
        F.col("id").alias("_user_id"),
        "current_age", "gender", "latitude", "longitude",
        strip_currency("yearly_income").alias("yearly_income"),
        strip_currency("total_debt").alias("total_debt"),
        strip_currency("per_capita_income").alias("per_capita_income"),
        "credit_score",
        "num_credit_cards",
    )

    # "Yes" → 1, "No" → 0, missing → null (unlabeled)
    labels_clean = labels.select(
        F.col("transaction_id").alias("_label_tx_id"),
        F.when(F.col("label") == "Yes", 1).otherwise(0).alias("is_fraud"),
    )

    return (
        tx
        .withColumn("amount", strip_currency("amount"))
        .withColumn("ts", F.to_timestamp("date", "yyyy-MM-dd HH:mm:ss"))
        # empty string errors → null
        .withColumn("errors", F.when(F.trim(F.col("errors")) == "", None).otherwise(F.col("errors")))
        .join(labels_clean, tx.id == F.col("_label_tx_id"), how="left")
        .join(cards_clean,  tx.card_id == F.col("_card_id"), how="left")
        .join(users_clean,  tx.client_id == F.col("_user_id"), how="left")
        .join(mcc,          tx.mcc == F.col("_mcc_code"), how="left")
        .select(
            tx.id.alias("transaction_id"),
            "ts",
            tx.client_id,
            "card_id",
            "amount",
            "use_chip",
            "merchant_id", "merchant_city", "merchant_state", "zip",
            "mcc", "mcc_name",
            "errors",
            # card
            "card_brand", "card_type", "has_chip", "credit_limit",
            "num_cards_issued", "year_pin_last_changed", "card_on_dark_web",
            # user
            "current_age", "gender", "latitude", "longitude",
            "yearly_income", "total_debt", "per_capita_income",
            "credit_score", "num_credit_cards",
            # label (null = unlabeled)
            "is_fraud",
        )
    )
