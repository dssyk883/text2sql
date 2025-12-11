import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from paths import DATA_DIR, INDEX_DIR
import faiss
import numpy as np
import re

embeddings_file = INDEX_DIR / "embeddings.npy"
db_ids_file = INDEX_DIR / "db_ids.pkl"
questions_file = INDEX_DIR / "questions.pkl"
sqls_file = INDEX_DIR / "sqls.pkl"
faiss_index_file = INDEX_DIR / "faiss.index"

embedder = None
train_questions = []
train_sqls = []
faiss_index = None

def load_index():
    global embedder, embeddings, train_questions, train_sqls, faiss_index

    print("*** Loading embedder...")
    embedder = SentenceTransformer('BAAI/bge-base-en-v1.5')

    print("*** Loading embeddings...")
    embeddings = np.load(embeddings_file)

    print("*** Loading questions...")
    with open(questions_file, 'rb') as f:
        train_questions = pickle.load(f)

    print("*** Loading sqls...")
    with open(sqls_file, 'rb') as f:
        train_sqls = pickle.load(f)

    print("*** Loading FAISS index...")
    faiss_index = faiss.read_index(str(faiss_index_file))

    print(f"*** Load {len(train_questions)} vectors on CPU")

def extract_tables(schema: str) -> set:
    """스키마에서 테이블명 추출"""
    tables = re.findall(r'CREATE TABLE ["\']?(\w+)["\']?', schema, re.IGNORECASE)
    return set(t.lower() for t in tables)


def extract_tables_from_sql(sql: str) -> set:
    """Question SQL에서 사용된 테이블명 추출"""
    # FROM, JOIN 뒤의 테이블명
    tables = re.findall(r'(?:FROM|JOIN)\s+["\']?(\w+)["\']?', sql, re.IGNORECASE)
    return set(t.lower() for t in tables)


def table_overlap_score(schema_tables: set, sql_tables: set) -> float:
    """테이블 overlap 점수 계산"""
    if not sql_tables:
        return 0.0
    intersection = schema_tables & sql_tables
    # print(f"=== Table overlap Score: {len(intersection) / len(sql_tables)} ===")
    return len(intersection) / len(sql_tables)


def retrieve_RAG_examples(question: str, schema: str, k: int = 5) -> list:
    if faiss_index is None:
        load_index()
       
    query_embedding = embedder.encode([question], convert_to_numpy=True)
    query_embedding = query_embedding.astype('float32')
    faiss.normalize_L2(query_embedding)
    
    distances, indices = faiss_index.search(query_embedding, k*3)

    schema_tables = extract_tables(schema)    
    scored = []
    for idx, dist in zip(indices[0], distances[0]):
        sql_tables = extract_tables_from_sql(train_sqls[idx])
        overlap = table_overlap_score(schema_tables, sql_tables)
        mix_score = dist * 0.7 + overlap * 0.3
        scored.append((idx, mix_score))
    
    final = sorted(scored, key=lambda x: x[1], reverse=True)[:k]
    examples = [{"input": train_questions[i], "query": train_sqls[i]} for i, _ in final]

    return examples