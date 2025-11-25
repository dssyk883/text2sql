from fastapi import FastAPI, HTTPException
from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains.sql_database.query import create_sql_query_chain
from database import DATABASE_URL
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

from pydantic import BaseModel
import os
import re

top_k = 5

examples = [
    {
        "input": "Show me all artists",
        "query": "SELECT name FROM artist;"
    },
    {
        "input": "How many albums are there?",
        "query": "SELECT COUNT(*) FROM album;"
    },
    {
        "input": "What Rock albums exist?",
        "query": "SELECT a.title FROM album a JOIN track t ON a.album_id = t.album_id JOIN genre g ON t.genre_id = g.genre_id WHERE g.name = 'Rock' LIMIT 5;"
    }
]

app = FastAPI()
db = SQLDatabase.from_uri(DATABASE_URL)
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = OllamaLLM(model="llama3.2", temperature=0)

psql_prompt = PromptTemplate(
    input_variables = ["input","query"],
    template = "Question: {input}\nSQL:{query}"
)

fshot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=psql_prompt,
    prefix="""Given an input quesiton, create a syntatically correct PostgreSQL query.
Critical Rules:
1. Return ONLY the sql auery, no explanations, no other lines.
2. Do NOT wrap in '''sql''' blocks
3. Add "LIMIT {top_k}" at the end unless COUNT/SUM/AVG is used
4. Only use columns from: {table_info}
Examples:""",
    suffix="Question:{input}\nSQL:",
    input_variables=["input","top_k","table_info"]
)

chain = create_sql_query_chain(
    llm=llm,
    db=db,
    k=top_k,
    prompt=fshot_prompt
    )

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def text_to_sql(request: QueryRequest):
    response = chain.invoke({"question": request.question})
    sql = extract_sql(response.strip())
    result = {"question": request.question, "sql":sql}
    try:
        query_result = db.run(sql)
        result["result"] = query_result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result

def extract_sql(text: str) -> str:
    """자연어가 섞인 응답 출력의 경우 SQL 쿼리만 추출"""
    match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return text.strip()
