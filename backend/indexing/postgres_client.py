import os
import time
import json
import psycopg2
from psycopg2 import sql
import torch
from psycopg2.extras import execute_values, Json
from backend.models.embedding_client import embed_queries, embed_sparse
from sentence_transformers import CrossEncoder
from backend.core.config_loader import settings

class SearchResult:
    def __init__(self, id, payload, score):
        self.id = id
        self.payload = payload
        self.score = score

class PostgresVectorDB:
    def __init__(self):
        if not settings:
            raise ValueError("❌ Settings not loaded. Check config_loader.py")

        self.host = settings.retrieval.postgres_host
        self.user = settings.retrieval.postgres_user
        self.password = settings.retrieval.postgres_password or "mysecretpassword"
        self.dbname = settings.retrieval.postgres_db
        self.port = str(settings.retrieval.postgres_port)
        
        print(f"🔌 Connecting to Postgres at {self.host}:{self.port}...")
        self.conn = psycopg2.connect(
            host=self.host, user=self.user, password=self.password, dbname=self.dbname, port=self.port
        )
        self.conn.autocommit = True
        self.table_name = settings.retrieval.vector_store_collection

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=2048, device="cuda")
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            query = sql.SQL("""
                CREATE TABLE IF NOT EXISTS {table} (
                    id bigserial PRIMARY KEY,
                    content text,
                    metadata jsonb,
                    dense_vector vector(1024),
                    sparse_vector sparsevec(250002)
                );
            """).format(table=sql.Identifier(self.table_name))
            cur.execute(query)

    def _format_sparse(self, indices, values, dim=250002):
        elements = [f"{i}:{v}" for i, v in zip(indices, values)]
        return "{" + ",".join(elements) + "}/" + str(dim)

    # --- NEW: Filter Logic ported from Target Code ---
    def _build_filter_clause(self, filters: dict):
        """
        Builds the WHERE clause dynamically based on the target code's logic.
        Returns: (sql_snippet, list_of_params)
        """
        if not filters:
            return sql.SQL("TRUE"), []

        conditions = []
        args = []

        for key, value in filters.items():
            # 1. Handle Lists (JSONB contains any): field ?| ['a', 'b']
            if isinstance(value, list):
                if not value: continue # Skip empty lists
                conditions.append(sql.SQL("metadata->%s ?| %s"))
                args.extend([key, value]) # Pass key as string, value as list
            
            # 2. Handle Dicts (Nested JSONB): field->key = value
            elif isinstance(value, dict):
                sub_conditions = []
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, list):
                        sub_conditions.append(sql.SQL("metadata->%s->%s ?| %s"))
                        args.extend([key, sub_key, sub_val])
                    else:
                        # Use json.dumps for primitives inside JSONB to ensure correct types
                        sub_conditions.append(sql.SQL("metadata->%s->%s = %s"))
                        args.extend([key, sub_key, json.dumps(sub_val)])
                
                if sub_conditions:
                    conditions.append(sql.SQL("(") + sql.SQL(" AND ").join(sub_conditions) + sql.SQL(")"))

            # 3. Handle Primitives (Strings, Ints, Bools)
            elif isinstance(value, (str, int, float, bool)):
                conditions.append(sql.SQL("metadata->>%s = %s"))
                args.extend([key, str(value)]) # ->> operator returns text

        if not conditions:
             return sql.SQL("TRUE"), []

        return sql.SQL(" AND ").join(conditions), args

    def upsert(self, points: list):
        data_to_insert = []
        for p in points:
            if hasattr(p, 'payload'):
                payload = p.payload
                dense = p.vector['dense']
                sparse_obj = p.vector['sparse']
                sparse_str = self._format_sparse(sparse_obj['indices'], sparse_obj['values'])
            else:
                payload = p.get('payload', {})
                dense = p.get('vector', {}).get('dense')
                sparse_obj = p.get('vector', {}).get('sparse')
                sparse_str = self._format_sparse(sparse_obj['indices'], sparse_obj['values'])

            content = payload.get("search_content") or payload.get("text", "")
            data_to_insert.append((content, Json(payload), dense, sparse_str))

        with self.conn.cursor() as cur:
            query = sql.SQL("""
                INSERT INTO {table} (content, metadata, dense_vector, sparse_vector)
                VALUES %s
            """).format(table=sql.Identifier(self.table_name))
            execute_values(cur, query, data_to_insert)

    # --- UPDATED: Hybrid Search with Filters ---
    def search(self, query_text: str, limit: int = 5, filter: dict = None):
        # 0. Embed Query
        query_dense = embed_queries([query_text])[0]
        sparse_output = embed_sparse([query_text])[0]
        query_sparse_str = self._format_sparse(
            list(int(k) for k in sparse_output.keys()), 
            list(float(v) for v in sparse_output.values())
        )

        # 1. Build Dynamic Filter Clause
        where_sql, filter_args = self._build_filter_clause(filter)

        # 2. Hybrid Search SQL
        # We use a CTE 'filtered_docs' to narrow down candidates FIRST, then rank.
        initial_limit = 15
        
        query_sql = sql.SQL("""
        WITH filtered_docs AS (
            SELECT id, dense_vector, sparse_vector, content, metadata
            FROM {table}
            WHERE {where_clause}
        ),
        dense_search AS (
            SELECT id, RANK() OVER (ORDER BY dense_vector <=> %s::vector) as dense_rank
            FROM filtered_docs
            LIMIT %s
        ),
        sparse_search AS (
            SELECT id, RANK() OVER (ORDER BY sparse_vector <#> %s::sparsevec) as sparse_rank
            FROM filtered_docs
            LIMIT %s
        )
        SELECT 
            COALESCE(d.id, s.id) as id,
            doc.content,
            doc.metadata,
            (1.0 / (60 + COALESCE(d.dense_rank, 0))) + (1.0 / (60 + COALESCE(s.sparse_rank, 0))) as rrf_score
        FROM dense_search d
        FULL OUTER JOIN sparse_search s ON d.id = s.id
        JOIN filtered_docs doc ON doc.id = COALESCE(d.id, s.id)
        ORDER BY rrf_score DESC
        LIMIT %s;
        """).format(
            table=sql.Identifier(self.table_name),
            where_clause=where_sql
        )
        
        # Combine arguments: [Filter Args] + [Dense Vec, Limit] + [Sparse Vec, Limit] + [Final Limit]
        full_args = filter_args + [query_dense, initial_limit, query_sparse_str, initial_limit, initial_limit]

        candidates = []
        with self.conn.cursor() as cur:
            cur.execute(query_sql, full_args)
            rows = cur.fetchall()
            
            for row in rows:
                candidates.append(SearchResult(
                    id=row[0],
                    payload=row[2],
                    score=row[3]
                ))

        if not candidates:
            return []

        # 3. Re-Ranking (Cross-Encoder)
        pairs = []
        for hit in candidates:
            doc_text = hit.payload.get("search_content")
            if not doc_text:
                summary = hit.payload.get("context_summary", "")
                raw_text = hit.payload.get("text", "")
                doc_text = f"{summary}\n{raw_text}" if summary else raw_text
            pairs.append([query_text, doc_text])

        start_rerank = time.time()
        scores = self.reranker.predict(pairs)
        print(f"📊 Re-ranking took {time.time() - start_rerank:.4f}s")
        for i, hit in enumerate(candidates):
            hit.score = float(scores[i])

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:limit]

# --- TEST BLOCK ---
if __name__ == "__main__":
    print("--- TEST: Hybrid + Filter ---")
    try:
        db = PostgresVectorDB()
        
        # Insert Data with Metadata
        test_text_1 = "Python is great for backend."
        test_text_2 = "Javascript is great for frontend."
        
        # Embed
        dense_1 = embed_queries([test_text_1])[0]
        sparse_1 = embed_sparse([test_text_1])[0]
        
        dense_2 = embed_queries([test_text_2])[0]
        sparse_2 = embed_sparse([test_text_2])[0]

        # Upsert
        p1 = {
            "payload": {"text": test_text_1, "metadata": {"lang": "python", "type": "backend"}},
            "vector": {
                "dense": dense_1, 
                "sparse": {"indices": list(sparse_1.keys()), "values": list(sparse_1.values())}
            }
        }
        p2 = {
            "payload": {"text": test_text_2, "metadata": {"lang": "js", "type": "frontend"}},
            "vector": {
                "dense": dense_2, 
                "sparse": {"indices": list(sparse_2.keys()), "values": list(sparse_2.values())}
            }
        }
        
        db.upsert([p1, p2])
        
        # Search WITH Filter
        print("\nSearching for 'great code' with filter lang='python'...")
        results = db.search("great code", filter={"lang": "python"}, limit=5)
        
        for res in results:
            print(f"Match: {res.payload['text']} | Metadata: {res.payload['metadata']}")

    except Exception as e:
        print(f"Error: {e}")