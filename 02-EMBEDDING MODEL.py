# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # DOWNLOAD AND SERVE EMBEDDING MODEL

# COMMAND ----------

# MAGIC %md
# MAGIC Estimated Duration: ~20 to 30 minutes
# MAGIC
# MAGIC - The embedding model is what will be used to convert our PDF text into vector embeddings. 
# MAGIC - These vector embeddings will be loaded into a Vector Search Index and allow for fast "Similarity Search". 
# MAGIC - This is a very important part of the RAG architecture. 
# MAGIC
# MAGIC - In this notebook we will be using the [gte-large](https://huggingface.co/thenlper/gte-large) open source embedding model from hugging face. 

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow-skinny[databricks]"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./RESOURCES/INIT

# COMMAND ----------

import json
import pandas as pd
import requests
import time
from sentence_transformers import SentenceTransformer # SentenceTransformers is a Python framework for state-of-the-art sentence, text and image embeddings
from mlflow.utils.databricks_utils import get_databricks_host_creds
creds = get_databricks_host_creds()

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# downloading the embedding model from Hugging Face
source_model_name = 'thenlper/gte-large' 
model = SentenceTransformer(source_model_name)

# COMMAND ----------

# Test the model, just to show it works.
sentences = ["Checking if this works", "Each sentence is converted"]
embeddings = model.encode(sentences)
print(embeddings)

# COMMAND ----------

# Compute input/output schema
signature = mlflow.models.signature.infer_signature(sentences, embeddings)
print(signature)

# COMMAND ----------

# start mlflow client
mlflow_client = mlflow.MlflowClient()

# COMMAND ----------

requirements = """
mlflow==2.12.1
accelerate==0.25.0
astunparse==1.6.3
boto3==1.24.28
cffi==1.15.1
defusedxml==0.7.1
dill==0.3.6
google-cloud-storage==2.11.0
ipython==8.14.0
opt-einsum==3.3.0
pydantic==1.10.6
sentence-transformers>=2.3.0
torch==2.0.1
transformers==4.36.1
huggingface-hub==0.25.2
"""
with open("requirements.txt", "w") as f:
    f.write(requirements)

# COMMAND ----------

# register model into UC
model_info = mlflow.sentence_transformers.log_model(
  model,
  artifact_path="model",
  signature=signature,
  input_example=sentences,
  registered_model_name=registered_embedding_model_name,
  pip_requirements="requirements.txt")

# write a model description
mlflow_client.update_registered_model(
  name=f"{registered_embedding_model_name}",
  description="https://huggingface.co/thenlper/gte-large"
)

# COMMAND ----------

# get latest version of model
def get_latest_model_version(mlflow_client, model_name):
  model_version_infos = mlflow_client.search_model_versions("name = '%s'" % model_name)
  return max([int(model_version_info.version) for model_version_info in model_version_infos])

model_version=get_latest_model_version(mlflow_client, registered_embedding_model_name)

print(model_version)

# COMMAND ----------

#GPU Serving of embedding model
served_models = [
    {
      "name": embedding_model_name,
      "model_name": registered_embedding_model_name,
      "model_version": model_version,
      "workload_size": "Medium",
      "workload_type": "GPU_LARGE",
      "scale_to_zero_enabled": False
    }
]
traffic_config = {"routes": [{"served_model_name": embedding_model_name, "traffic_percentage": "100"}]}

# Create or update model serving endpoint
if not endpoint_exists(embedding_endpoint_name):
  create_endpoint(embedding_endpoint_name, served_models)
else:
  update_endpoint(embedding_endpoint_name, served_models)

# COMMAND ----------

# Prepare data for query
# Query endpoint (once ready)
sentences = ['Hello world', 'Good morning']
ds_dict = {'dataframe_split': pd.DataFrame(pd.Series(sentences)).to_dict(orient='split')}
data_json = json.dumps(ds_dict, allow_nan=True)
print(data_json)

# COMMAND ----------

# testing endpoint
invoke_headers = {'Authorization': f'Bearer {creds.token}', 'Content-Type': 'application/json'}
invoke_url = f'{workspace_url}/serving-endpoints/{embedding_endpoint_name}/invocations'
print(invoke_url)

start = time.time()
invoke_response = requests.request(method='POST', headers=invoke_headers, url=invoke_url, data=data_json, timeout=360)
end = time.time()
print(f'time in seconds: {end-start}')

if invoke_response.status_code != 200:
  raise Exception(f'Request failed with status {invoke_response.status_code}, {invoke_response.text}')

print(invoke_response.text)
