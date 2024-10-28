# Databricks notebook source
# MAGIC %md
# MAGIC # INGEST PDF FROM GITHUB

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Estimated Duration: ~2 mins
# MAGIC
# MAGIC In this notebook we will be ingesting our data. 
# MAGIC
# MAGIC Before running this notebook, there needs to be PDF files in a public Github repo. Here, we'll use my Github repo where I have uploaded RCGGS PDFs.

# COMMAND ----------

# MAGIC %md
# MAGIC ### LOAD VARIABLES AND HELPER FUNCTIONS

# COMMAND ----------

# MAGIC %run ./RESOURCES/INIT

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
# MAGIC
# MAGIC ##  BRONZE DATA LAYER
# MAGIC
# MAGIC
# MAGIC We will be ingesting the files from a Unity Catalog Volume
# MAGIC
# MAGIC Volumes are Unity Catalog objects that enable governance over non-tabular datasets. Volumes represent a logical volume of storage in a cloud object storage location. Volumes provide capabilities for accessing, storing, governing, and organizing files.
# MAGIC
# MAGIC While tables provide governance over tabular datasets, volumes add governance over non-tabular datasets. You can use volumes to store and access files in any format, including structured, semi-structured, and unstructured data.
# MAGIC
# MAGIC Databricks recommends using volumes to govern access to all non-tabular data. Like tables, volumes can be managed or external

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC <img src="https://github.com/smakubi/rcg/blob/main/images/image1.jpeg?raw=true" style="float: center;  margin-left: 100px">

# COMMAND ----------

# MAGIC %md
# MAGIC ### LOAD PDF INTO VOLUME

# COMMAND ----------

# download PDF files from Github

upload_pdfs_to_volume(volume_path)
display(dbutils.fs.ls(volume_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## INGEST PDF INTO DELTA AS BINARY USING AUTOLOADER

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## What is Databricks Auto Loader?
# MAGIC
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/autoloader/autoloader-edited-anim.gif" style="float:right; margin-left: 10px" />
# MAGIC
# MAGIC [Databricks Auto Loader](https://docs.databricks.com/ingestion/auto-loader/index.html) lets you scan a cloud storage folder (S3, ADLS, GS) and only ingest the new data that arrived since the previous run.
# MAGIC
# MAGIC This is called **incremental ingestion**.
# MAGIC
# MAGIC Auto Loader can be used in a near real-time stream or in a batch fashion, e.g., running every night to ingest daily data.
# MAGIC
# MAGIC Auto Loader provides a strong gaurantee when used with a Delta sink (the data will only be ingested once).
# MAGIC
# MAGIC ### How Auto Loader simplifies data ingestion
# MAGIC
# MAGIC Ingesting data at scale from cloud storage can be really hard at scale. Auto Loader makes it easy, offering these benefits:
# MAGIC
# MAGIC
# MAGIC * **Incremental** & **cost-efficient** ingestion (removes unnecessary listing or state handling)
# MAGIC * **Simple** and **resilient** operation: no tuning or manual code required
# MAGIC * Scalable to **billions of files**
# MAGIC   * Using incremental listing (recommended, relies on filename order)
# MAGIC   * Leveraging notification + message queue (when incremental listing can't be used)
# MAGIC * **Schema inference** and **schema evolution** are handled out of the box for most formats (csv, json, avro, images...)
# MAGIC
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&aip=1&t=event&ec=dbdemos&ea=VIEW&dp=%2F_dbdemos%2Fdata-engineering%2Fauto-loader%2F01-Auto-loader-schema-evolution-Ingestion&cid=1444828305810485&uid=6103594171624722">

# COMMAND ----------

# READ PDF AS BINARY USING AUTOLOADER
df = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "BINARYFILE")
    .option("pathGlobFilter", "*.pdf")
    .load('dbfs:'+volume_path)
)

# WRITE THE DATA INTO A DELTA TABLE 
(
    df.writeStream.trigger(availableNow=True)
    .option("checkpointLocation", f"dbfs:{volume_path}/checkpoints/raw_docs")
    .table(f"{catalog}.{schema}.pdf_raw")
    .awaitTermination()
)

# COMMAND ----------

# LETS QUERY DATA IN THE DELTA TABLE
display(spark.sql(f"SELECT * FROM {catalog}.{schema}.pdf_raw LIMIT 10"))
