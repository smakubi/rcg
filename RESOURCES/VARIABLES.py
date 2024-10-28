# Databricks notebook source
#GENERAL

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get().split(".")[0]
catalog="llm_workshop" 
schema=f"{user}_rcggs"
volume_name="pdf"

# HUGGING FACE TOKEN
hf_token = "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"

workspace_url = "https://" + spark.conf.get("spark.databricks.workspaceUrl") 
base_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

# COMMAND ----------

# CREATE STORAGE

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
spark.sql(f"USE SCHEMA {schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_name}") 

# COMMAND ----------

#DATA PREP

volume_path=f"/Volumes/{catalog}/{schema}/pdf" #create a volume with this structure
chunk_size=500
chunk_overlap=50

# COMMAND ----------

#EMBEDDING MODEL

embedding_endpoint_name = f"{user}-gte-large"
embedding_model_name=f'{user}-gte-large'
registered_embedding_model_name = f'{catalog}.{schema}.{embedding_model_name}'

# COMMAND ----------

#VECTOR SEARCH

vs_endpoint_name=f'{user}_vs_endpoint'
vs_index = f"{user}_vs_documents_index"
vs_index_fullname = f"{catalog}.{schema}.{vs_index}"
sync_table_name = f"{user}_rcg_documents_sync"
sync_table_fullname = f"{catalog}.{schema}.{sync_table_name}"

# COMMAND ----------

#LLM SERVING

llm_model_name=f'{user}-llama-2-7b-hf-chat'
registered_llm_model_name=f'{catalog}.{schema}.{llm_model_name}'
llm_endpoint_name = f'{user}-llama-2-7b-hf-chat'
