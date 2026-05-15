# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Workflow Orchestration
# MAGIC
# MAGIC Tạo Databricks Workflows job orchestrate toàn bộ pipeline:
# MAGIC
# MAGIC ```
# MAGIC fraud-demo-pipeline
# MAGIC ├── Task 1: dlt_pipeline      (Bronze → Silver → Gold)
# MAGIC ├── Task 2: automl_training   [depends on dlt_pipeline]
# MAGIC ├── Task 3: xgboost_upgrade   [depends on automl_training]
# MAGIC └── Task 4: deploy_serving    [depends on xgboost_upgrade]
# MAGIC ```

# COMMAND ----------
# MAGIC %run ../config

# COMMAND ----------
import requests
import json

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
# MAGIC %md ## Parameters
# MAGIC
# MAGIC Điền DLT pipeline ID và cluster ID trước khi chạy.
# MAGIC - DLT pipeline ID: lấy từ Databricks UI → Delta Live Tables → pipeline ID
# MAGIC - Cluster ID: lấy từ Compute → cluster ID

dbutils.widgets.text("dlt_pipeline_id", "", "DLT Pipeline ID")
dbutils.widgets.text("cluster_id",      "", "Cluster ID")
dbutils.widgets.text("workspace_path",  "/fraud-demo", "Workspace Notebook Path")

DLT_PIPELINE_ID = dbutils.widgets.get("dlt_pipeline_id")
CLUSTER_ID      = dbutils.widgets.get("cluster_id")
WS_PATH         = dbutils.widgets.get("workspace_path")

# COMMAND ----------
# MAGIC %md ## Create Workflow Job

job_config = {
    "name": "fraud-demo-pipeline",
    "tasks": [
        {
            "task_key": "dlt_pipeline",
            "pipeline_task": {"pipeline_id": DLT_PIPELINE_ID},
        },
        {
            "task_key":            "automl_training",
            "depends_on":          [{"task_key": "dlt_pipeline"}],
            "existing_cluster_id": CLUSTER_ID,
            "notebook_task": {
                "notebook_path": f"{WS_PATH}/04-Data-Science-ML/04.1-AutoML-FSI-fraud",
            },
        },
        {
            "task_key":            "xgboost_upgrade",
            "depends_on":          [{"task_key": "automl_training"}],
            "existing_cluster_id": CLUSTER_ID,
            "notebook_task": {
                "notebook_path": f"{WS_PATH}/04-Data-Science-ML/04.4-Upgrade-to-imbalance-and-xgboost-model-fraud",
            },
        },
        {
            "task_key":            "deploy_serving",
            "depends_on":          [{"task_key": "xgboost_upgrade"}],
            "existing_cluster_id": CLUSTER_ID,
            "notebook_task": {
                "notebook_path": f"{WS_PATH}/04-Data-Science-ML/04.3-Model-serving-realtime-inference-fraud",
            },
        },
    ],
    "tags": {"project": "fraud-detection-demo"},
}

resp   = requests.post(
    f"https://{DATABRICKS_HOST}/api/2.1/jobs/create",
    headers=HEADERS,
    json=job_config,
)
job_id = resp.json().get("job_id")
print(f"Job created : {job_id}")
print(f"View at     : https://{DATABRICKS_HOST}/#job/{job_id}")

# COMMAND ----------
# MAGIC %md ## Trigger Run

run_resp = requests.post(
    f"https://{DATABRICKS_HOST}/api/2.1/jobs/run-now",
    headers=HEADERS,
    json={"job_id": job_id},
)
run_id = run_resp.json().get("run_id")
print(f"Run started : {run_id}")
print(f"View at     : https://{DATABRICKS_HOST}/#job/{job_id}/run/{run_id}")
