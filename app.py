import logging
import sys
from sqlalchemy import create_engine, MetaData, text
from llama_index.core import Settings
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import SQLDatabase
from llama_index.core.query_pipeline import QueryPipeline as QP
from llama_index.core.indices.struct_store.sql_query import (
    SQLTableRetrieverQueryEngine
)
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import SQLRetriever
from typing import List
from llama_index.core.query_pipeline import FnComponent
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_pipeline import FnComponent
from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.core.storage import StorageContext
import os
from pathlib import Path
from typing import Dict
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.query_pipeline import FnComponent
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
    CustomQueryComponent,
)
import streamlit as st

logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

engine = create_engine("sqlite:///bankdeposit.db")

table_details = {
    "table1_name": "table1 description",
    "table2_name": "table2 description"
}

data = {}
data["table1_name"] = {
    "column1_name": "column1 description",
    "column2_name": "column2 description"
}

data["table2_name"] = {
    "column1_name": "column1 description",
    "column2_name": "column2 description"
}

metadata = MetaData()
metadata.reflect(bind=engine)

for k, v in data.items():
    table = metadata.tables[k]
    for k, v in v.items():
        table.columns[k].comment = v

from llama_index.core import SQLDatabase
sql_database = SQLDatabase(engine, sample_rows_in_table_info=100, include_tables=["table1_name", "table2_name"], metadata=metadata)

llm = Gemini(model_name="models/gemini-1.5-pro", api_key="GEMINI API KEY", temperature=0, max_tokens=1048576)

embeddings = GeminiEmbedding(model="models/textembedding-gecko@004", api_key="GEMINI API KEY")

qp = QP(verbose=True)

Settings.llm = llm
Settings.embed_model = embeddings
Settings._callback_manager = qp.callback_manager

table_node_mapping = SQLTableNodeMapping(sql_database)

table_schema_objs = [
    (SQLTableSchema(table_name="table1_name", context_str=table_details["table1_name"])),
    (SQLTableSchema(table_name="table2_name", context_str=table_details["table2_name"]))
]

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
    embed_model=embeddings
)

obj_retriever = obj_index.as_retriever(similarity_top_k=3)

query_engine = SQLTableRetrieverQueryEngine(
    sql_database, obj_index.as_retriever(similarity_top_k=3),
    service_context = Settings
)

sql_retriever = SQLRetriever(sql_database)

def get_table_context_str(table_schema_objs: List[SQLTableSchema]):
    """Get table context string."""
    context_strs = []
    for table_schema_obj in table_schema_objs:
        table_info = sql_database.get_single_table_info(
            table_schema_obj.table_name
        )

        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context

        context_strs.append(table_info)

    return "\n\n".join(context_strs)

table_parser_component = FnComponent(fn=get_table_context_str)

def parse_response_to_sql(response: llm) -> str:
    """Parse response to SQL"""
    response = str(response).replace("assistant: ","").replace("\n"," ")

    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]

        if response.startswith("SQLQuery:"):
            response = response[len("SQLQuery:"):]

    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]
        st.write(f"**SQL Query**: {response}")

    return response.strip().strip("```").strip()

file = open('fewshots.txt', 'r')
content = file.read()
lines = content.split('###')

sql_parser_component = FnComponent(fn=parse_response_to_sql)

text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
    dialect=engine.dialect.name
)
text2sql_prompt.template = text2sql_prompt.template + '''\n\n

INSTRUCTIONS TO BE FOLLOWED:

1.Please generate an SQL query in one line without adding any special characters like \n or \r. Do not include words like "sql" or "assistant" at the beginning.

2.When generating SQL queries, use the LIKE statement for searching text data in the WHERE clauseto filter data based on conditions.

3.Convert both the column value and the value being compared to LOWERCASE when the comparing text data.

Use the following example as a reference:

{lines} 
'''

sql_retriever = SQLRetriever(sql_database)

response_synthesis_prompt_str = (
    "Given an input questiob, synthesize a response from the query results.\n"
    "Query: {query_str}\n"
    "SQL: {sql_query}\n"
    "SQL Response: {context_str}\n"
    "Response: "
)

response_synthesis_prompt = PromptTemplate(
    response_synthesis_prompt_str
)

qp.add_modules({
    "input": InputComponent(),
    "table_retriever": obj_retriever,
    "table_output_parser": table_parser_component,
    "text2sql_prompt": text2sql_prompt,
    "text2sql_llm": llm,
    "sql_output_parser": sql_parser_component,
    "sql_retriever": sql_retriever,
    "response_synthesis_prompt": response_synthesis_prompt,
    "response_synthesis_llm": llm,
})

qp.add_link("input", "table_retriever")
qp.add_link("input", "table_output_parser", dest_key="query_str")
qp.add_link(
    "table_retriever", "table_output_parser", dest_key="table_schema_objs"
)
qp.add_link("input", "text2sql_prompt", dest_key="query_str")
qp.add_link("table_output_parser", "text2sql_prompt", dest_key="schema")
qp.add_chain(
    ["text2sql_prompt", "text2sql_llm", "sql_output_parser", "sql_retriever"]
)
qp.add_link(
    "sql_output_parser", "response_synthesis_prompt", dest_key="sql_query"
)
qp.add_link(
    "sql_retriever", "response_synthesis_prompt", dest_key="context_str"
)
qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
qp.add_link("response_synthesis_prompt", "response_synthesis_llm")

st.title("NL2SQL Bot")
user_input = st.text_input("Enter your query in natural language:")

if user_input:
    response = qp.run(
        query = user_input
    )
    st.write(f"**Response**: {response}")
