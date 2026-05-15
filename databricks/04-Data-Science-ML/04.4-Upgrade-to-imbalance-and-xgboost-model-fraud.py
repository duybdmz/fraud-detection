# Databricks notebook source
# MAGIC %md
# MAGIC # 04.4 — XGBoost + Imbalance Handling (Champion)
# MAGIC
# MAGIC AutoML baseline không tối ưu cho fraud rate 0.15%.
# MAGIC XGBoost với `scale_pos_weight` và time-based split cho kết quả tốt hơn.

# COMMAND ----------
# MAGIC %run ../config

# COMMAND ----------
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
from pyspark.sql import functions as F
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    classification_report,
)

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
# MAGIC %md ## Time-based Split
# MAGIC
# MAGIC Dataset: 2010-01 → 2019-10 (gần 10 năm)
# MAGIC
# MAGIC **Không dùng random split** — gây time leakage, model sẽ "thấy tương lai" khi train.
# MAGIC
# MAGIC | Split | Period | Mục đích |
# MAGIC |-------|--------|----------|
# MAGIC | Train | 2010-01 → 2017-12 | Học pattern |
# MAGIC | Val   | 2018-01 → 2018-12 | Tune + early stopping |
# MAGIC | Test  | 2019-01 → 2019-10 | Đánh giá cuối |

df = spark.table(f"{CATALOG}.gold.gold_fraud_training")

train_df = df.filter(F.col("ts") <  "2018-01-01")
val_df   = df.filter((F.col("ts") >= "2018-01-01") & (F.col("ts") < "2019-01-01"))
test_df  = df.filter(F.col("ts") >= "2019-01-01")

print(f"Train : {train_df.count():,}")
print(f"Val   : {val_df.count():,}")
print(f"Test  : {test_df.count():,}")

def to_pandas(sdf):
    pdf = sdf.select(FEATURE_COLS + ["is_fraud"]).toPandas()
    pdf[FEATURE_COLS] = pdf[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    return pdf

train_pd = to_pandas(train_df)
val_pd   = to_pandas(val_df)
test_pd  = to_pandas(test_df)

X_train, y_train = train_pd[FEATURE_COLS], train_pd["is_fraud"].astype(int)
X_val,   y_val   = val_pd[FEATURE_COLS],   val_pd["is_fraud"].astype(int)
X_test,  y_test  = test_pd[FEATURE_COLS],  test_pd["is_fraud"].astype(int)

# Class imbalance: ~1 fraud per 668 normal transactions
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"\nscale_pos_weight = {scale_pos_weight:.1f}  (neg={neg:,}, pos={pos:,})")

# COMMAND ----------
# MAGIC %md ## Train XGBoost

params = {
    "objective":        "binary:logistic",
    "eval_metric":      ["logloss", "aucpr"],
    "scale_pos_weight": scale_pos_weight,   # ~668
    "max_depth":        6,
    "learning_rate":    0.05,
    "n_estimators":     500,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "tree_method":      "hist",
    "random_state":     42,
}

with mlflow.start_run(run_name="xgboost_imbalance") as run:
    mlflow.log_params(params)
    mlflow.log_param("train_period",      "2010-01 → 2017-12")
    mlflow.log_param("val_period",        "2018-01 → 2018-12")
    mlflow.log_param("test_period",       "2019-01 → 2019-10")
    mlflow.log_param("scale_pos_weight",  round(scale_pos_weight, 1))
    mlflow.log_param("n_train",           len(X_train))
    mlflow.log_param("n_val",             len(X_val))
    mlflow.log_param("n_test",            len(X_test))

    model = xgb.XGBClassifier(**params, early_stopping_rounds=20)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    # Evaluate on val + test
    for split, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        proba = model.predict_proba(X)[:, 1]
        pred  = (proba >= 0.5).astype(int)
        mlflow.log_metrics({
            f"{split}_f1":        f1_score(y, pred),
            f"{split}_precision": precision_score(y, pred),
            f"{split}_recall":    recall_score(y, pred),
            f"{split}_roc_auc":   roc_auc_score(y, proba),
            f"{split}_pr_auc":    average_precision_score(y, proba),
        })
        print(f"\n=== {split.upper()} ===")
        print(classification_report(y, pred, target_names=["normal", "fraud"]))

    # Log model with signature
    signature = mlflow.models.infer_signature(X_train, model.predict_proba(X_train)[:, 1])
    mlflow.xgboost.log_model(
        model,
        artifact_path="fraud_model",
        registered_model_name=MODEL_NAME,
        signature=signature,
        input_example=X_train.head(3),
    )

    # Promote to champion
    client   = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest   = max(int(v.version) for v in versions)
    client.set_registered_model_alias(MODEL_NAME, "champion", latest)
    print(f"\nModel v{latest} → alias 'champion'")
