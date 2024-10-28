# Databricks notebook source
import requests

API_URL = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
EMAIL = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
USER_NAME = EMAIL.split('.')[0]
URL = f"{API_URL}/api/2.0/clusters/create"


CLUSTER_CONFIG = {
    "cluster_name": f"{USER_NAME}'s LLM Cluster",
    "spark_version": "14.3.x-cpu-ml-scala2.12",
    "spark_conf": {
        "spark.master": "local[*, 4]",
        "spark.databricks.cluster.profile": "singleNode"
    },
    "azure_attributes": {
        "first_on_demand": 1,
        "availability": "ON_DEMAND_AZURE",
        "spot_bid_max_price": -1
    },
    "node_type_id": "Standard_D32ads_v5",
    "driver_node_type_id": "Standard_D32ads_v5",
    "custom_tags": {
        "ResourceClass": "SingleNode"
    },
    "autotermination_minutes": 120,
    "enable_elastic_disk": True,
    "single_user_name": f"{EMAIL}",
    "enable_local_disk_encryption": False,
    "data_security_mode": "SINGLE_USER",
    "runtime_engine": "STANDARD",
    "effective_spark_version": "15.4.x-cpu-ml-scala2.12",
    "num_workers": 0,
    "apply_policy_default_values": False
}

response = requests.post(URL, json=CLUSTER_CONFIG, headers={"Authorization": f"Bearer {TOKEN}"})


if response.status_code == 200:
    print("Cluster created successfully")
else:
    print("Failed to create cluster:", response.json())
