from fastapi import FastAPI, HTTPException
from langchain_community.llms.ollama import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains.sql_database.query import create_sql_query_chain
from database import DATABASE_URL
from dotenv import load_dotenv
from pydantic import BaseModel
import os

app = FastAPI()
db = SQLDatabase.from_uri(DATABASE_URL)
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = Ollama(model="llama3.2", temperature=0)
chain = create_sql_query_chain(llm, db)

class QueryRequest(BaseModel):
    question: str
    execute: bool = False

@app.post("/query")
async def text_to_sql(request: QueryRequest):
    response = chain.invoke({"question": request.question})

    if "SQLQuery:" in response:
        sql = response.split('SQLQuery:')[1].strip()
    else:
        sql = response.strip()
    result = {"question": request.question, "sql":sql}

    if request.execute:
        try:
            query_result = db.run(sql)
            result["result"] = query_result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return result