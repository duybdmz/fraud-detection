# Databricks notebook source
# DLT pipeline — Gold layer
# Must be in the same Lakeflow pipeline as 01-bronze.py and 02-silver.py

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------
# MAGIC %md ## Gold — Feature Engineering
# MAGIC
# MAGIC Tạo 31 features từ silver_transactions:
# MAGIC
# MAGIC | Nhóm | Features |
# MAGIC |------|----------|
# MAGIC | Transaction | abs_amount, is_refund, is_online/chip/swipe, mcc, has_error |
# MAGIC | Time | hour_of_day, day_of_week, month |
# MAGIC | Card | has_chip_flag, credit_limit, num_cards_issued, year_pin_last_changed, is_dark_web |
# MAGIC | User | current_age, yearly_income, total_debt, per_capita_income, credit_score, num_credit_cards, debt_to_income |
# MAGIC | Window 1h | tx_count_1h, tx_amount_1h |
# MAGIC | Window 24h | tx_count_24h, tx_amount_24h, avg_amount_24h, std_amount_24h, unique_merchants_24h, unique_states_24h |
# MAGIC | Window 7d | tx_count_7d, tx_amount_7d |

@dp.table(
    name="gold_fraud_features",
    comment="ML-ready feature table — 31 features + rolling windows 1h/24h/7d",
    table_properties={"quality": "gold"},
)
def gold_fraud_features():
    # Partition by card_id, order by unix timestamp for range-based windows
    w_1h  = Window.partitionBy("card_id").orderBy(F.col("ts").cast("long")).rangeBetween(-3_600,    0)
    w_24h = Window.partitionBy("card_id").orderBy(F.col("ts").cast("long")).rangeBetween(-86_400,   0)
    w_7d  = Window.partitionBy("card_id").orderBy(F.col("ts").cast("long")).rangeBetween(-604_800,  0)

    return (
        dp.read("silver_transactions")

        # --- Time features ---
        .withColumn("hour_of_day", F.hour("ts"))
        .withColumn("day_of_week", F.dayofweek("ts"))
        .withColumn("month",       F.month("ts"))

        # --- Transaction type flags ---
        # use_chip: "Swipe Transaction" 52.4% | "Chip Transaction" 35.9% | "Online Transaction" 11.7%
        .withColumn("is_online", (F.col("use_chip") == "Online Transaction").cast("int"))
        .withColumn("is_chip",   (F.col("use_chip") == "Chip Transaction").cast("int"))
        .withColumn("is_swipe",  (F.col("use_chip") == "Swipe Transaction").cast("int"))

        # --- Amount features ---
        # ~660K negative amounts = refunds
        .withColumn("is_refund",  (F.col("amount") < 0).cast("int"))
        .withColumn("abs_amount", F.abs("amount"))

        # --- Error flag ---
        # Insufficient Balance 130K | Bad PIN 32K | Technical Glitch 26K | ...
        .withColumn("has_error", F.col("errors").isNotNull().cast("int"))

        # --- Card risk flags ---
        # card_on_dark_web = 0% in this dataset — keep as feature, may differ in prod
        .withColumn("is_dark_web",   (F.col("card_on_dark_web") == "Yes").cast("int"))
        .withColumn("has_chip_flag", (F.col("has_chip") == "YES").cast("int"))

        # --- User risk ratio ---
        .withColumn("debt_to_income",
            F.when(F.col("yearly_income") > 0, F.col("total_debt") / F.col("yearly_income"))
            .otherwise(None)
        )

        # --- Rolling window 1h ---
        .withColumn("tx_count_1h",  F.count("transaction_id").over(w_1h))
        .withColumn("tx_amount_1h", F.sum("abs_amount").over(w_1h))

        # --- Rolling window 24h ---
        .withColumn("tx_count_24h",         F.count("transaction_id").over(w_24h))
        .withColumn("tx_amount_24h",        F.sum("abs_amount").over(w_24h))
        .withColumn("avg_amount_24h",       F.avg("abs_amount").over(w_24h))
        .withColumn("std_amount_24h",       F.stddev("abs_amount").over(w_24h))
        .withColumn("unique_merchants_24h", F.countDistinct("merchant_id").over(w_24h))
        .withColumn("unique_states_24h",    F.countDistinct("merchant_state").over(w_24h))

        # --- Rolling window 7d ---
        .withColumn("tx_count_7d",  F.count("transaction_id").over(w_7d))
        .withColumn("tx_amount_7d", F.sum("abs_amount").over(w_7d))
    )

# COMMAND ----------
# MAGIC %md ## Gold — Training Table
# MAGIC
# MAGIC Chỉ giữ rows có label (8,914,963 / 13,305,915).
# MAGIC ~4.4M unlabeled rows bị drop — không dùng cho supervised training.

@dp.table(
    name="gold_fraud_training",
    comment="Labeled subset — 8.9M rows, 0.15% fraud rate, dùng cho training",
    table_properties={"quality": "gold"},
)
@dp.expect_or_drop("has_label",    "is_fraud IS NOT NULL")
@dp.expect_or_drop("valid_label",  "is_fraud IN (0, 1)")
def gold_fraud_training():
    return dp.read("gold_fraud_features").filter(F.col("is_fraud").isNotNull())
