#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate

'''
通过SQLDatabaseChain，自动生成SQL，并获取数据
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    get_api_key()

    db = SQLDatabase.from_uri("sqlite://lanchain07.db")
    llm = OpenAI(temperature=0)

    _DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Use the following format:

    Question: "Question here"
    SQLQuery: "SQL Query to run"
    SQLResult: "Result of the SQLQuery"
    Answer: "Final answer here"

    Only use the following tables:

    {table_info}

    If someone asks for the table foobar, they really mean the employee table.

    Question: {input}"""

    PROMPT = PromptTemplate(
        input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
    )

    db_chain = SQLDatabaseChain(llm=llm, database=db, prompt=PROMPT, verbose=True)

    db_chain.run("How many employees are there in the foobar table?")


"""
> Entering new SQLDatabaseChain chain...
How many employees are there in the foobar table? 
SQLQuery: SELECT COUNT(*) FROM Employee;
SQLResult: [(8,)]
Answer: There are 8 employees in the foobar table.
> Finished chain.
"""
