# Databricks notebook source
# MAGIC %md
# MAGIC # DATABRICKS VECTOR SEARCH

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Estimated Duration:  ~20 minutes
# MAGIC In this notebook we will create a Vector Search Index on top of our Delta Lake table
# MAGIC
# MAGIC We now have our knowledge base ready, and saved as a Delta Lake table within Unity Catalog (including permission, lineage, audit logs and all UC features).
# MAGIC
# MAGIC Typically, deploying a production-grade Vector Search index on top of your knowledge base is a difficult task. You need to maintain a process to capture table changes, index the model, provide a security layer, and all sorts of advanced search capabilities.
# MAGIC
# MAGIC Databricks Vector Search removes those painpoints.
# MAGIC
# MAGIC Databricks Vector Search is a new production-grade service that allows you to store a vector representation of your data, including metadata. It will automatically sync with the source Delta table and keep your index up-to-date without you needing to worry about underlying pipelines or clusters. 
# MAGIC
# MAGIC It makes embeddings highly accessible. You can query the index with a simple API to return the most similar vectors, and can optionally include filters or keyword-based queries.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### WHAT'S REQUIRED FOR OUR VECTOR SEARCH INDEX
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/databricks-vector-search-type.png?raw=true" style="float: center" width="800px">
# MAGIC
# MAGIC Databricks provide multiple types of vector search indexes:
# MAGIC
# MAGIC - **Managed embeddings**: you provide a text column and endpoint name and Databricks synchronizes the index with your Delta table 
# MAGIC - **Self Managed embeddings**: you compute the embeddings and save them as a field of your Delta Table, Databricks will then synchronize the index
# MAGIC - **Direct index**: when you want to use and update the index without having a Delta Table.
# MAGIC
# MAGIC In this workshop, we will show you how to setup a **Self-managed Embeddings** index. 
# MAGIC
# MAGIC To do so, we will have to first compute the embeddings of our chunks and save them as a Delta Lake table field as `array&ltfloat&gt`

# COMMAND ----------

# MAGIC %md
# MAGIC ### INSTALL EXTERNAL LIBRARIES

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### LOAD VARIABLES

# COMMAND ----------

# MAGIC %run ./resources/00-init

# COMMAND ----------

# MAGIC %md
# MAGIC ### IMPORT LIBRARIES

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T
from databricks.sdk.runtime import *

from langchain.text_splitter import TokenTextSplitter
from typing import Iterator
import pandas as pd
from pyspark.sql.functions import rand,when

from transformers import LlamaTokenizer
from typing import Iterator, List, Dict
import pandas as pd
from random import randint

# COMMAND ----------

# MAGIC %md
# MAGIC ### CREATE VECTOR SEARCH ENDPOINT

# COMMAND ----------

# import Vector Search and initiate the class
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

# COMMAND ----------

# Create a Vector Search Index

if not vs_endpoint_exists(vsc, vs_endpoint_name):
    vsc.create_endpoint(name=vs_endpoint_name, endpoint_type="STANDARD")
    wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name)
else:
    print(f"Endpoint named {vs_endpoint_name} is ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### COMPUTING THE CHUNK EMBEDDINGS AND SAVING TO DELTA TABLE 
# MAGIC
# MAGIC The last step is to now compute an embedding for all our documentation chunks. Let's create an udf to compute the embeddings using the embedding model endpoint.
# MAGIC
# MAGIC *Note that this part would typically be setup as a production-grade job, running as soon as a new documentation page is updated. <br/> This could be setup as a **Delta Live Table pipeline to incrementally consume updates**.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### UDF TO COMPUTE CHUNK EMBEDDINGS

# COMMAND ----------

@F.pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    import mlflow.deployments

    deploy_client = mlflow.deployments.get_deploy_client("databricks")

    def get_embeddings(batch):
        # Note: this will fail if an exception is thrown during embedding creation (add try/except if needed)
        response = deploy_client.predict(
            endpoint=embedding_endpoint_name,
            inputs={"inputs": batch}
        )
        return response['predictions']

    # EMBEDDING MODEL TAKES AT MOST 150 INPUTS PER REQUEST
    max_batch_size = 150
    
    all_embeddings = []
    for i in range(0, len(contents), max_batch_size):
        batch = contents.iloc[i:i+max_batch_size].tolist()
        embeddings = get_embeddings(batch)
        all_embeddings.extend(embeddings)

    # Ensure we return the same number of embeddings as input rows
    assert len(all_embeddings) == len(contents), f"Mismatch in number of embeddings: got {len(all_embeddings)}, expected {len(contents)}"
    
    return pd.Series(all_embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC ## SAVE CHUNK EMBEDDINGS TO DELTA TABLE

# COMMAND ----------

# in order to take advantage of the delta sync, we need to ensure that delta.enableChangeDataFeed is enabled.
# create the sync table with an ID column and enableChangeDataFeed

spark.sql(f"CREATE TABLE IF NOT EXISTS {sync_table_fullname} (id BIGINT GENERATED BY DEFAULT AS IDENTITY,url STRING,content STRING,embedding ARRAY < FLOAT >)TBLPROPERTIES (delta.enableChangeDataFeed = true)");

# COMMAND ----------

# calculate the embeddings and write to the sync table
(
    spark.readStream.table(f"{catalog}.{schema}.pdf_raw_chunks")
    .withColumn("embedding", get_embedding("content"))
    .selectExpr("path as url", "content", "embedding")

    .writeStream.trigger(availableNow=True)
    .option("checkpointLocation", f"dbfs:{volume_path}/checkpoints/pdf_chunk")
    .table(f"{sync_table_fullname}")
    .awaitTermination()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## PREVIEW CHUNK EMBEDDINGS

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {sync_table_fullname} WHERE url like '%.pdf' LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### CREATE SELF-MANAGED VECTOR SEARCH INDEX USING ENDPOINT

# COMMAND ----------

# MAGIC %md
# MAGIC Estimated Duration: 5 mins

# COMMAND ----------

# create a vector search sync with a delta table. This will create a serverless DLT job that will manage creating the embeddings of any new documents that are added to the delta table.
vsc.create_delta_sync_index_and_wait(
  endpoint_name=vs_endpoint_name,
  source_table_name=sync_table_fullname,
  index_name=vs_index_fullname,
  pipeline_type='TRIGGERED',
  primary_key="id",
  embedding_source_column="content",
  embedding_model_endpoint_name=embedding_endpoint_name
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## SEARCHING FOR SIMILAR CONTENT
# MAGIC
# MAGIC That's all we have to do. Databricks will automatically capture and synchronize new entries in your Delta Lake Table.
# MAGIC
# MAGIC Note that depending on your dataset size and model size, index creation can take a few seconds to start and index your embeddings.
# MAGIC
# MAGIC Let's give it a try and search for similar content.
# MAGIC
# MAGIC *Note: `similarity_search` also supports a filters parameter. This is useful to add a security layer to your RAG system: you can filter out some sensitive content based on who is doing the call (for example filter on a specific department based on the user preference).*

# COMMAND ----------

# Once our Vector Search Index is created (may take some time depending on how many documents are synced), lets do a similarity search

results = vsc.get_index(index_name=vs_index_fullname, endpoint_name=vs_endpoint_name).similarity_search(
  query_text="What does RCG do?",
  columns=['id','url','content'],
  num_results=3
  )
  
results
