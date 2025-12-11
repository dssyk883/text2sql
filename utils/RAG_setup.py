import json
import pickle
from paths import DATA_DIR, INDEX_DIR, SPIDER_DIR
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.utilities import SQLDatabase

train_path = DATA_DIR / "train_spider.json"
spider_db_dir = SPIDER_DIR / "database"

embedding_file = INDEX_DIR / "embeddings.npy"
db_ids_file = INDEX_DIR / "db_ids.pkl"
questions_file = INDEX_DIR / "questions.pkl"
sqls_file = INDEX_DIR / "sqls.pkl"
faiss_index_file = INDEX_DIR / "faiss.index"

def summarize_schema(schema):
    import re
    # print("=== RAW Schema ===")
    # print(schema)
    # CREATE TABLE 파싱
    tables = re.findall(
        r'CREATE TABLE ["\']?(\w+)["\']?\s*\((.*?)\n\)',
        schema,
        re.DOTALL | re.IGNORECASE
    )
    
    tables_info = []
    fk_info = []
    
    for table_name, table_def in tables:
        # 모든 "컬럼명" 또는 컬럼명 (따옴표 유무 상관없이) 추출
        # INTEGER, CHAR, TEXT, FLOAT 등 데이터 타입 앞에 있는 단어
        col_pattern = r'["\']?(\w+)["\']?\s+(?:INTEGER|INT|CHAR|VARCHAR|TEXT|FLOAT|REAL|DOUBLE|DECIMAL|NUMERIC|BLOB|BOOLEAN|DATE|TIME|DATETIME)'
        
        cols = re.findall(col_pattern, table_def, re.IGNORECASE)
        
        # 중복 제거 (순서 유지)
        seen = set()
        unique_cols = []
        for col in cols:
            if col not in seen:
                seen.add(col)
                unique_cols.append(col)
        
        # FK 추출
        fk_pattern = r'FOREIGN KEY\s*\(["\']?(\w+)["\']?\)\s+REFERENCES\s+(\w+)\s*\(["\']?(\w+)["\']?\)'
        fk_matches = re.findall(fk_pattern, table_def, re.IGNORECASE)
        
        for from_col, to_table, to_col in fk_matches:
            fk_info.append(f"{table_name}.{from_col} = {to_table}.{to_col}")
        
        if unique_cols:
            tables_info.append(f"{table_name}({', '.join(unique_cols)})")
    
    result = "Tables:\n" + "\n".join(tables_info)
    
    if fk_info:
        result += "\n\nForeign Keys:\n" + "\n".join(fk_info)
    
    # print("=== Summarized Schema ===")
    # print(result)

    return result

def get_schema_safe(db_id):
    
    try:
        db_path = spider_db_dir / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            return ""
        
        db = SQLDatabase.from_uri(
            f"sqlite:///{str(db_path)}",
            sample_rows_in_table_info=0
        )
        return db.table_info
        
    except Exception as e:
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
            schemas = [row[0] for row in cursor.fetchall() if row[0]]
            conn.close()
            return "\n\n".join(schemas)
        except:
            print(f"Warning: Failed to load schema for {db_id}")
            return ""

def build_save_index():

    with open(train_path, "r") as f:
        train_data = json.load(f)

    db_ids = [item['db_id'] for item in train_data]
    questions = [item['question'] for item in train_data]
    sqls = [item['query'] for item in train_data]

    model = SentenceTransformer('BAAI/bge-base-en-v1.5')

    combined_texts = []
    for idx, item in enumerate(train_data):
        question = item['question']
        combined_texts.append(question)

    embeddings = model.encode(
        combined_texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)

    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)

    np.save(embedding_file, embeddings)        

    with open(db_ids_file, "wb") as f:
        pickle.dump(db_ids, f)
    with open(questions_file, "wb") as f:
        pickle.dump(questions, f)
    with open(sqls_file, "wb") as f:
        pickle.dump(sqls, f)
    faiss.write_index(index, str(faiss_index_file))

    print(f"Index built successfully with {len(questions)} examples!")

if __name__ == "__main__":
    build_save_index()