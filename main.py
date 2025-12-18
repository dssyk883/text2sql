from fastapi import FastAPI, HTTPException
from database import DATABASE_URL
from models import generate_sql
from pydantic import BaseModel
from evaluation.benchmark import run_spider_benchmark
import argparse

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
    parser = argparse.ArgumentParser(description='NL2SQL Few-Shot Benchmark')
    parser.add_argument('-m', '--mode', choices=['benchmark', 'app'],
                        default='benchmark', help='Execution mode')
    parser.add_argument('-s', '--strategy', choices=['random', 'rag', 'ic', 'jacc'],
                        default='random', help='Few-shot retrieval strategy')
    parser.add_argument('--model', choices=['qwen', 'mistral', 'sonnet'],
                        default='qwen', help='LLM Model')
    parser.add_argument('-b', '--batch', type=int, default=100,
                        help='Batch size: number of examples to evaluate')    
    parser.add_argument('-k', '--k-examples', type=int, default=5,
                        help='Number of few-shot examples')
    parser.add_argument('-c', '--cluster', type=int, default=1,
                        help='Number of clusters in intent-clustering')
    parser.add_argument('--use-limit', action='store_true', help='Add LIMIT clause to SQL')
    
    args = parser.parse_args()

    if args.mode == 'benchmark':
        run_spider_benchmark(args)

    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)