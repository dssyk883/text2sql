from database import DATABASE_URL
from models import generate_sql
from pydantic import BaseModel
from evaluation.benchmark import run_spider_benchmark
from evaluation.agent_benchmark import run_spider_agent_benchmark
import argparse

def main():
    parser = argparse.ArgumentParser(description='NL2SQL Few-Shot Benchmark')
    parser.add_argument('-m', '--mode', choices=['benchmark', 'app', 'agent'],
                        default='benchmark', help='Execution mode')
    parser.add_argument('-s', '--strategy', choices=['random', 'rag', 'ic', 'jacc'],
                        default='random', help='Few-shot retrieval strategy')
    parser.add_argument('--model', choices=['qwen', 'mistral', 'sonnet'],
                        default='qwen', help='LLM Model')
    parser.add_argument('-i', '--max-iterations', type=int,
                        default=10, help='Max iterations for agent mode')
    parser.add_argument('-r', '--max-refinements',type=int,
                        default=3, help='Max refinements for agent mode')
    parser.add_argument('-b', '--batch', type=int,
                        default=100, help='Batch size: number of examples to evaluate')    
    parser.add_argument('-k', '--k-examples', type=int,
                        default=5,help='Number of few-shot examples')
    parser.add_argument('-c', '--cluster', type=int,
                        default=1, help='Number of clusters in intent-clustering')
    parser.add_argument('--use-limit', action='store_true', help='Add LIMIT clause to SQL')
    
    args = parser.parse_args()

    if args.mode == 'benchmark':
        run_spider_benchmark(args)
    
    if args.mode == 'agent':
        run_spider_agent_benchmark(args)

if __name__ == "__main__":
    main()