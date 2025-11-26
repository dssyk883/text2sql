import json
from pathlib import Path
import re

spider_path = Path("/home/sykwon/spider")
train_path = spider_path / "train_spider.json"
result_example_path = spider_path / "evaluation_examples" / "eval_result_example.txt"
wrong_example_path = spider_path / "evaluation_examples" / "wrong_example.txt"

SQL_comp1 = ["WHERE", "GROUP BY", "ORDER BY", "LIMIT", "JOIN", "OR", "LIKE"]
SQL_comp2 = ["EXCEPT", "UNION", "INTERSECT"] # and sub query
SQL_aggs = ["COUNT", "MIN", "MAX", "SUM", "AVG"]
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

# (venv) sykwon@asus-skwon:~/spider/evaluation_examples$ vi eval_result_example.txt

def classify_hardness(query: str) -> str:
    comp1_count = sum(1 for keyword in SQL_comp1 if re.search(rf'\b{keyword}\b', query) and keyword != 'JOIN') 
    comp1_count += len(re.findall(r'\bJOIN\b', query))
    
    agg_count = sum(1 for keyword in SQL_aggs if re.search(rf'\b{keyword}\b', query))
    where_match = re.search(r'WHERE\s+(.+?)(?=\s+GROUP BY|\s+ORDER BY|\s+LIMIT|$)', query)
    where_count = 0

    subquery_count = query.count('(SELECT')
    subquery_removed = re.sub(r'\(SELECT.*?\)', '', query, flags=re.DOTALL)
    comp2_count = sum(1 for keyword in SQL_comp2 if re.search(rf'\b{keyword}\b', subquery_removed))
    
    if subquery_count > 0:
        comp2_count += 1
        
    if where_match:
        where_part = where_match.group(1)
        where_count = 1 + len(re.findall(r'\b(?:AND|OR)\b', where_part))
        where_len = len(re.findall(r'\b(?:AND|OR)\b', where_match.group(1)))
        print(f"WHERE part: '{where_match.group(1)}'")
        print(f"AND count: {where_len}")
    groupby_match = re.search(r'GROUP BY\s+(.+?)(?=\s+(?:HAVING|ORDER BY|LIMIT|$))', query)
    col_match = re.search(r'SELECT\s+(.*?)\s+FROM', query)

    num_cols = 0
    if col_match:
        col_part = col_match.group(1).strip()
        num_cols = len(col_part.split(','))

    num_groupby = 0
    if groupby_match:
        groupby_part = groupby_match.group(1).strip()
        num_groupby = len(groupby_part.split(','))

    others_count = sum([
        agg_count > 1,
        num_cols > 1,
        where_count > 1,
        num_groupby > 1
    ])

    print(f"comp1: {comp1_count}, comp2: {comp2_count}, others: {others_count}")

    # easy: 0 - 1 from comp1, none from comp2, no condition from others satisfied
    if comp1_count < 2 and comp2_count == 0 and others_count == 0:
        return "easy"
    
    # medium: Others < 3, comp1 < 2 , 0 in comp2,
    # OR
    # comp1 = 2, Others < 2 , and 0 from comp2
    if ((others_count < 3 and comp1_count < 2 and comp2_count == 0)
        or (others_count < 2 and comp1_count == 2 and comp2_count == 0)):
        return "medium"

    # hard: > 2 others, comp1 < 3 , 0 in comp2
    # OR
    # 2 < comp1 <= 3, Others < 3 , 0 in comp2
    # OR
    # comp1 < 2, 0 in others, comp2 = 1
    if ((others_count > 2 and comp1_count < 3 and comp2_count == 0)
        or (2 < comp1_count <= 3 and others_count < 3 and comp2_count == 0)
        or (comp1_count < 2 and others_count == 0 and comp2_count == 1)):
        return "hard"

    return "extra"


wrong_sql = []
total, wrong = 0, 0
for idx, data in enumerate(example_data, 1):
    line = data.split(":", 1)
    if len(line) < 2 or "pred" in line[0]:
        continue
    
    parts = line[0].strip().split()
    if not parts:
        continue
    truth_hardness = parts[0]
    query = line[1].strip()
    pred_hardness = classify_hardness(query)
    if truth_hardness != pred_hardness:
        wrong += 1
        total += 1
        dg = f'Predicted: {pred_hardness}, Answer: {truth_hardness} - {line[1]}'
        print(dg)
        wrong_sql.append(dg)

    else:
        total += 1

with open(wrong_example_path, "w") as f:
    f.write('\n'.join(wrong_sql))

print(f"Total of {total}: Wrong: {wrong}")