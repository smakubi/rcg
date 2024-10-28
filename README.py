# Databricks notebook source
# MAGIC %md
# MAGIC ## STEP BY STEP INSTRUCTIONS

# COMMAND ----------

# MAGIC %md
# MAGIC 1. **CREATE A CLUSTER WITH 14.3 LTS ML DATABRICKS RUNTIME AND USE THAT CLUSTER TO RUN NOTEBOOKS 00-05**: Navigate to **RESOURCES -> COMPUTE**. On the top right connect to Serverless Compute and click **Run All** to create a Single User Cluster.
# MAGIC
# MAGIC 2. **OPTIONALLY UPDATE HUGGING FACE TOKEN**: Go to **RESOURCES -> VARIABLES** and update the hf_token with your own HuggingFace token. While on this notebook, connect to your **first_name's LLM Cluster** then go ahead and **Run All**. 
# MAGIC
# MAGIC 3. **ENSURE CATALOG IS CREATED:** On the left pane of your workspace, go to Catalog and ensure that a catalog named **llm_workshop** was created.
# MAGIC
# MAGIC
# MAGIC 4. **RUN ALL NOTEBOOKS** : Go ahead and Run All notebooks 00-05 in order.
# MAGIC
# MAGIC 5. **MODEL SERVING NOTE:** Note that there are 2 different model serving notebooks (02 and 04). One serves the embeddign model and another serves a chat Large Language Model. These will each take anywhere from 25-45 minutes to complete. you can check the status of the endpoint under **Machine Learning --> Serving** tab on the left pane of your wirkspace.
# MAGIC
# MAGIC 6. **VECTOR SEARCH ENDPOINT NOTE**: Note that the deployment of the Vector Search Endpoint will take 10-15 minutes. After that a Vector Search Index will be created on that endpoint. It can be monitored there.
