# Databricks notebook source
# MAGIC %md
# MAGIC # 04.2 — AutoML Generated Notebook (LightGBM Best Trial)
# MAGIC
# MAGIC Notebook này mô phỏng output AutoML tự generate sau khi chạy 04.1.
# MAGIC Best trial thường là LightGBM với `class_weight="balanced"`.
# MAGIC Set alias "challenger" để so sánh A/B với XGBoost champion.

# COMMAND ----------
# MAGIC %run ../config

# COMMAND ----------
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import pandas as pd
from pyspark.sql import functions as F
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    classification_report,
)

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
# MAGIC %md ## Load Data — Same Time-based Split as 04.4

df = spark.table(f"{CATALOG}.gold.gold_fraud_training")

train_df = df.filter(F.col("ts") <  "2018-01-01")
val_df   = df.filter((F.col("ts") >= "2018-01-01") & (F.col("ts") < "2019-01-01"))

def to_pandas(sdf):
    pdf = sdf.select(FEATURE_COLS + ["is_fraud"]).toPandas()
    pdf[FEATURE_COLS] = pdf[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    return pdf

train_pd = to_pandas(train_df)
val_pd   = to_pandas(val_df)

X_train, y_train = train_pd[FEATURE_COLS], train_pd["is_fraud"].astype(int)
X_val,   y_val   = val_pd[FEATURE_COLS],   val_pd["is_fraud"].astype(int)

# COMMAND ----------
# MAGIC %md ## AutoML Best Trial — LightGBM
# MAGIC
# MAGIC `class_weight="balanced"` = AutoML's default imbalance strategy.
# MAGIC Berbeda dengan XGBoost yang dùng `scale_pos_weight` explicit.

best_params = {
    "colsample_bytree":  0.5,
    "lambda_l1":         0.01,
    "lambda_l2":         10.0,
    "learning_rate":     0.05,
    "max_bin":           511,
    "max_depth":         7,
    "min_child_samples": 20,
    "n_estimators":      400,
    "num_leaves":        100,
    "subsample":         0.8,
    "class_weight":      "balanced",
    "random_state":      42,
    "verbose":           -1,
}

with mlflow.start_run(run_name="automl_lgbm_best_trial") as run:
    mlflow.log_params(best_params)
    mlflow.log_param("train_period", "2010-01 → 2017-12")
    mlflow.log_param("val_period",   "2018-01 → 2018-12")

    model = lgb.LGBMClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred  = (val_proba >= 0.5).astype(int)

    mlflow.log_metrics({
        "val_f1":        f1_score(y_val, val_pred),
        "val_precision": precision_score(y_val, val_pred),
        "val_recall":    recall_score(y_val, val_pred),
        "val_roc_auc":   roc_auc_score(y_val, val_proba),
        "val_pr_auc":    average_precision_score(y_val, val_proba),
    })

    print(classification_report(y_val, val_pred, target_names=["normal", "fraud"]))

    signature = mlflow.models.infer_signature(X_train, model.predict_proba(X_train)[:, 1])
    mlflow.lightgbm.log_model(
        model,
        artifact_path="fraud_model",
        registered_model_name=MODEL_NAME,
        signature=signature,
        input_example=X_train.head(3),
    )

    client   = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest   = max(int(v.version) for v in versions)
    client.set_registered_model_alias(MODEL_NAME, "challenger", latest)
    print(f"\nModel v{latest} → alias 'challenger'")
    print("Run 04.5 to set up A/B traffic split")
