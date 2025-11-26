from fastapi import FastAPI, HTTPException
from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains.sql_database.query import create_sql_query_chain
from database import DATABASE_URL
from models import generate_sql
from pydantic import BaseModel


app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def text_to_sql(request: QueryRequest):
    try:
        sql, query_result = generate_sql(request.question, DATABASE_URL)
        return {
            "question": request.question,
            "sql": sql,
            "result": query_result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import sys
    # 벤치마크 실행 시
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        from evaluation.benchmark import run_spider_benchmark
        metrics = run_spider_benchmark()
        print(f"Total: {metrics['total']}, Success: {metrics['success']}")
    # Ollama 실행용
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)