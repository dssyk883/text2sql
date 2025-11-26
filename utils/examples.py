import json
from pathlib import Path

spider_path = Path("/home/sykwon/spider")
train_path = spider_path / "train_spider.json"
result_example_path = spider_path / "evaluation_example" / "eval_result_example.txt"

SQL_comp1 = ["WHERE", "GROUP BY", "ORDER BY", "LIMIT", "JOIN", "OR", "LIKE", "HAVING"]
SQL_comp2 = ["EXCEPT", "UNION", "INTERSECT", "NESTED"]
"""
Others: number of agg > 1,
number of select columns > 1,
number of where conditions > 1,
number of group by clauses > 1,
number of group by clauses > 1 (no consider col1-col2 math equations etc.)
"""
# with open(train_path, "r") as f:
#     train_data = json.load(f)

with open(result_example_path, "r") as f:
    example_data = f.readlines()
# https://github.com/taoyds/spider/tree/master/evaluation_examples
# (venv) sykwon@asus-skwon:~/spider/evaluation_examples$ vi eval_result_example.txt
def classify_hardness(query: str) -> str:
    comp1_count, comp2_count = 0, 0

for idx, data in enumerate(example_data, 1):
    line = data.split(":")
    if "pred" in line[0]:
        continue
