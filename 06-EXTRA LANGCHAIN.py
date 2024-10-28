# Databricks notebook source
# MAGIC %md
# MAGIC ## LIBRARY INSTALLATION

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch
# MAGIC
# MAGIC %pip install mlflow==2.9.0 lxml==4.9.3 langchain==0.0.344 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.12.0 cloudpickle==2.2.1 pydantic==2.5.2
# MAGIC
# MAGIC %pip install pip mlflow[databricks]==2.9.0
# MAGIC
# MAGIC %pip install google-search-results numexpr
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## IMPORT LIBRARIES

# COMMAND ----------

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from langchain.chains import RetrievalQA
import os

from operator import itemgetter

from databricks.vector_search.client import VectorSearchClient
from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough, RunnableLambda

# from langchain_core.messages import HumanMessage
from langchain.memory import ChatMessageHistory, ConversationSummaryMemory, ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationChain, LLMChain, SequentialChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.agents import load_tools
from langchain.agents import initialize_agent


# COMMAND ----------

# MAGIC %run ./RESOURCES/INIT

# COMMAND ----------

# MAGIC %md
# MAGIC ## LCEL

# COMMAND ----------

prompt = PromptTemplate(input_variables=["topic"], template="Tell me a short joke about {topic}")

llm = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

chain.invoke({'topic':'cats'})

# COMMAND ----------

# MAGIC %md
# MAGIC ## MULTI-CHAIN CHAINS

# COMMAND ----------

prompt1 = ChatPromptTemplate.from_template("what is the city {person} is from?")
prompt2 = ChatPromptTemplate.from_template(
    "what country is the city {city} in? respond in {language}"
)

llm= ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)

chain1 = prompt1 | llm | StrOutputParser()

chain2 = (
    {"city": chain1, "language": itemgetter("language")}
    | prompt2
    | llm
    | StrOutputParser()
)

chain2.invoke({"person": "obama", "language": "spanish"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## AGENTS

# COMMAND ----------

os.environ["SERPAPI_API_KEY"] ='4cbdc6699f3110c59a0e1189869009cbd2b2846758ce00a591cb24135df724e5'

llm = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, 
                         llm, 
                         agent="zero-shot-react-description", 
                         verbose=True,
                         handle_parsing_errors=True)

# agent.run("Who is the United States President? What will be their age in 5 years?")
# agent.run("What 350 raised to the power of 2?")
agent.run("Who is the current leader of Japan? What is their age multiplied by 2? Answer both questions")

# COMMAND ----------

tools[0]

# COMMAND ----------

tools[1]

# COMMAND ----------

# MAGIC %md
# MAGIC ## RUNNABLES

# COMMAND ----------

# MAGIC %md
# MAGIC #### RunnableParallel

# COMMAND ----------

llm = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)

joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | llm

poem_chain = (
    ChatPromptTemplate.from_template("write a 2-line poem about {topic}") | llm
)

map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)

map_chain.invoke({"topic": "bear"})

# COMMAND ----------

# MAGIC %md
# MAGIC #### RunnablePassthrough

# COMMAND ----------

runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    modified=lambda x: x["num"] + 1,
)

runnable.invoke({"num": 1})

# COMMAND ----------

# MAGIC %md
# MAGIC #### RunnableLambda

# COMMAND ----------

def length_function(text):
    return len(text)


def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)


def multiple_length_function(_dict):
    return _multiple_length_function(_dict["text1"], _dict["text2"])


prompt = ChatPromptTemplate.from_template("what is {a} + {b}")
llm = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)


chain = (
    {
        "a": itemgetter("foo") | RunnableLambda(length_function),
        "b": {"text1": itemgetter("foo"), "text2": itemgetter("bar")}
        | RunnableLambda(multiple_length_function),
    }
    | prompt
    | llm
)

chain.invoke({"foo": "test", "bar": "testing"})

# COMMAND ----------

# MAGIC %md
# MAGIC #### RunnableBranch

# COMMAND ----------

llm = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)


chain = (
    PromptTemplate.from_template(
        """Given the user question below, classify it as either being about `Langchain`, `Cooking`, or `Other`.

Do not respond with more than one word.

<question>
{question}
</question>

Classification:"""
    )
    | llm
    | StrOutputParser()
)

response=chain.invoke({"question": "how do i unclog a drain"})

# COMMAND ----------

response.lower().strip()

# COMMAND ----------



langchain_chain = (
    PromptTemplate.from_template(
        """You are an expert in langchain. \
Always answer questions starting with "I'm the best at Langchain...". \
Respond to the following question:

Question: {question}
Answer:"""
    )
    | llm
)


chef_chain = (
    PromptTemplate.from_template(
        """You are an expert in cooking. \
Always answer questions starting with "Mama mia, let's get cookin....". \
Respond to the following question:

Question: {question}
Answer:"""
    )
    | llm
)

general_chain = (
    PromptTemplate.from_template(
        """Respond to the following question:

Question: {question}
Answer:"""
    )
    | llm
)



branch = RunnableBranch(
    (lambda x: "langchain" in x["topic"].lower().strip(), langchain_chain),
    (lambda x: "cooking" in x["topic"].lower().strip(), chef_chain),
    general_chain,
)


full_chain = {"topic": chain, "question": lambda x: x["question"]} | branch


full_chain.invoke({"question": "how do I make meatloaf?"})   

## Can also do same logic with Runnable Lambda
# def route(info):
#     if "langchain" in info["topic"].lower():
#         return langchain_chain
#     elif "cook" in info["topic"].lower():
#         return chef_chain
#     else:
#         return general_chain
      


# COMMAND ----------

# MAGIC %md
# MAGIC #MEMORY

# COMMAND ----------

llm = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

conversation.invoke('hi there')


# COMMAND ----------

conversation.invoke('what color is the sky and why?')
