import json
from pathlib import Path
import random
import time

from agent import NL2SQLAgent, AgentConfig

text2sql_path = Path(__file__).parent.parent
spider_db_dir_path = text2sql_path.parent / "spider" / "database"
spider_dir_path = text2sql_path.parent / "spider"

examples_path = spider_dir_path / "evaluation_examples" / "examples"
dev_json_path = examples_path  / "dev.json"
tables_json_path = examples_path / "tables.json"


def run_spider_agent_benchmark(args):
    agent = NL2SQLAgent(
        model_type = args.model,
        k = args.k_examples,
        config = AgentConfig(
            max_iterations=args.max_iterations,
            max_refinements=args.max_refinements
        )
    )

    if not dev_json_path.exists():
        raise FileNotFoundError(f"Spider dev.json not found at {dev_json_path}")
    if not tables_json_path.exists():
        raise FileNotFoundError(f"Spider tables.json not found at {tables_json_path}")
    
    with open(dev_json_path, "r") as f:
        dev_data = json.load(f)
    
    predictions = []
    results = []

    print(f"Starting Spider benchmark on {args.batch} examples .... ")
    start_time = time.time()
    random.seed(88)
    batch = random.sample(dev_data, args.batch)

    for idx, example in enumerate(batch, 1):
        question = example["question"]
        db_id = example["db_id"]
        gold_sql = example["query"]
        db_path = spider_db_dir_path / db_id / f"{db_id}.sqlite"

        if not db_path.exists():
            print(f"Warning: DB db_id = {db_id} not found")
            continue

        result = agent.run(question, db_id, db_path)
        if result['success']:
            predictions.append(result['sql'])
        else:
            predictions.append(', '.join(result['sql']))
        
        results.append({
            "question": question,
            "schema": result['schema'],
            "predicted_sql": predictions[-1],
            "predicted_result": result.get('result', None),
            "gold_sql": gold_sql,
            "db_id": db_id,
            "iterations": result.get('iterations', 0),
            "refinements": result.get('refinements', 0),
            "error": result.get('error', None),
            "error_type": result.get('error_type', None),
            "success": result['success'],

        })

        print(f"Success: {idx} / {args.batch}")
        if idx % 10 == 0:
            print(f"Progress: {idx} / {args.batch}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    output_dir = Path(__file__).parent.parent / "output" / f"{args.model}_{args.batch}_k-{args.k_examples}"
    pred_file = output_dir / f"pred-{args.strategy}.sql"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_file, "w") as f:
        f.write("\n".join(predictions))
    
    results_file = output_dir / f"predictions-{args.strategy}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark complete!")
    print(f"Predictions saved to {pred_file}")
    print(f"Detailed results saved to {results_file}")
    
    # 단순 sql 실행 성공률 (정확성 XX)
    # 정확도는 여기서 만들어진 sql 문으로
    success_count = sum(1 for r in results if r["success"])
    print(f"Success rate: {success_count}/{args.batch} ({success_count/args.batch*100:.1f}%)")
    print(f"Total Execution Time: {int(elapsed_time//60)}분 {elapsed_time%60:.2f}초")
    
    return {
        "total": args.batch,
        "success": success_count,
        "failed": args.batch - success_count,
        "results": results
    }