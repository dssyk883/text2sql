import json
from pathlib import Path

PRJ_ROOT = Path(__file__).parent.parent
DATA_DIR = PRJ_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"

train_path = DATA_DIR / "train_spider.json"
jaccard_matrix_file = INDEX_DIR / "jaccard_matrix.npy"
questions_file = INDEX_DIR / "questions.pkl"
sqls_file = INDEX_DIR / "sqls.pkl"

def jaccard_similarity(q1, q2):
    set1 = set(q1.lower().split())
    set2 = set(q2.lower().split())
    return len(set1 & set2) / len(set1 | set2)

train_questions = []
train_sqls = []

def load_train_questions():
    global train_questions, train_sqls
    with open(train_path, "r") as f:
        train_data = json.load(f)
    train_questions = [item['question'] for item in train_data]
    train_sqls = [item['query'] for item in train_data]
    

def retrieve_jaccard_examples(question, k=5):    
    """
    Jaccard 유사도 기반 k개의 예제 반환
    
    :param question: input string
    :param k: 예제 개수
    """
    if len(train_questions) == 0:
        load_train_questions()
    
    scores = []
    for i, train_q in enumerate(train_questions):
        sim = jaccard_similarity(question, train_q)
        scores.append((sim, i))
    
    scores.sort(reverse=True)

    examples = []
    for _, idx in scores[:k]:
        examples.append({
            "input": train_questions[idx],
            "query": train_sqls[idx]
        })
    
    return examples