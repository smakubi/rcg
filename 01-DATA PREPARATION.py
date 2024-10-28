# Databricks notebook source
# MAGIC %md
# MAGIC # CLEAN UP INGESTED PDF DATA

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Estimated Duration: ~2 mins
# MAGIC
# MAGIC In this notebook we will be preparing our downloaded PDF data ready for consumption. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### INSTALL REQUIRED EXTERNAL LIBRARIES 

# COMMAND ----------

# MAGIC %pip install pypdf==4.1.0 llama-index==0.10.43
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### LOAD VARIABLES

# COMMAND ----------

# MAGIC %run ./RESOURCES/INIT

# COMMAND ----------

print(user)
print(f'{catalog}.{schema}')
print(volume_path)

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
from random import randint

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## EXTRACT PDF CONTENT AS TEXT CHUNKS
# MAGIC
# MAGIC We need to convert the **PDF documents bytes** to **text**, and extract **chunks** from their content.
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-self-managed-2.png?raw=true" style="float: center" width="500px">
# MAGIC
# MAGIC
# MAGIC
# MAGIC If your PDFs were saved as images, you will need an **OCR to extract** the text.
# MAGIC
# MAGIC Using the **`pypdf`** library within a **Spark UDF** makes it easy to extract text. 
# MAGIC
# MAGIC
# MAGIC
# MAGIC <br style="clear: both">
# MAGIC
# MAGIC ### SPLITTING BIG DOCUMENTATION PAGE IN SMALLER CHUNKS
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/chunk-window-size.png?raw=true" style="float: center" width="700px">
# MAGIC
# MAGIC In this workshop, some PDFs can be very large, with a lot of text.
# MAGIC
# MAGIC We'll extract the content and then use **llama_index `SentenceSplitter`**, and ensure that each chunk isn't bigger **than 500 tokens**. 
# MAGIC
# MAGIC
# MAGIC The chunk size and chunk overlap depend on the use case and the PDF files. 
# MAGIC
# MAGIC Remember that your **prompt + answer should stay below your model max window size (4096 for llama2)**. 
# MAGIC
# MAGIC
# MAGIC <br/>
# MAGIC <br style="clear: both">
# MAGIC <div style="background-color: #def2ff; padding: 15px;  border-radius: 30px; ">
# MAGIC   <strong>Information</strong><br/>
# MAGIC   Remember that the following steps are specific to your dataset. This is a critical part to building a successful RAG assistant.
# MAGIC   <br/> Always take time to review the chunks created and ensure they make sense and contain relevant information.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## EXTRACT BINARY DATA INTO TEXT FROM PDF

# COMMAND ----------

import warnings
import io
import re
from pypdf import PdfReader
import requests

def parse_bytes_pypdf(raw_doc_contents_bytes: bytes):
    try:
        pdf = io.BytesIO(raw_doc_contents_bytes)
        reader = PdfReader(pdf)
        
        def clean_text(text):
            # REPLACE MULTIPLE SPACES OR NEW LINES WITH A SINGLE SPACE 
            text = re.sub(r'\s+', ' ', text)
            # HANDLE HYPHENATED WORDS AT THE END OF LINES 
            text = re.sub(r'-\s+', '', text)
            # REMOVE SPACES BEFORE PUNCTUATION 
            text = re.sub(r'\s+([.,;:!?])', r'\1', text)
            return text.strip()
        
        parsed_content = [clean_text(page.extract_text()) for page in reader.pages]
        return "\n\n".join(parsed_content)  # USE DOUBLE NEWLINE TO SEPARATE PAGES 
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### TEST TEXT EXTRACTION FUNCTION

# COMMAND ----------

# test text extraction

with requests.get("https://github.com/smakubi/pbs/blob/main/pdfs/Case-Study-Mitsui.pdf?raw=true") as pdf:
    doc = parse_bytes_pypdf(pdf.content)
    print(doc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## TEXT SPLITTER PANDAS UDF

# COMMAND ----------

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document, set_global_tokenizer
from transformers import AutoTokenizer
from typing import Iterator
import io

# REDUCE ARROW BATCH SIZE AS PDF CAN BE BIG IN MEMOERY 
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)

# PANDAS UDF
@F.pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:

    # SET LLAMA2 AS TOKENIZER TO MATCH OUR MODEL SIZE (WILL STAY BELOW GTE 1024 LIMIT) 
    set_global_tokenizer(
        AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    )
    # SENTENCE SPLITTER FROM LLAMA_INDEX TO SPLIT ON SENTENCES 
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=10)

    def extract_and_split(b):
        txt = parse_bytes_pypdf(b)
        if txt is None:
            return []
        nodes = splitter.get_nodes_from_documents([Document(text=txt)])
        return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)

# COMMAND ----------

# MAGIC %md
# MAGIC ### APPLY THE TEXT SPLITTER UDF

# COMMAND ----------

df = spark.table(f"{catalog}.{schema}.pdf_raw").withColumn("content", F.explode(read_as_chunk("content")))
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### SAVE TABLE AS CHUNKS

# COMMAND ----------

# write to delta table

# in this step we are overwriting the table with the new PDF data, however in a production scenario we would append to the table in order to keep the historical data and add any new data

df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.pdf_raw_chunks")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {catalog}.{schema}.pdf_raw_chunks"))
