import faiss
import numpy as np
import orjson
from sentence_transformers import SentenceTransformer

INDEX_PATH = r"C:\rag_project\data\faiss_index\test_index.faiss"
META_PATH = r"C:\rag_project\data\wikipedia_processed\meta_test.jsonl"

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

index = faiss.read_index(INDEX_PATH)

metadata = []
with open(META_PATH, "rb") as f:
    for line in f:
        metadata.append(orjson.loads(line))


query = input("Ask a question: ")

query_vec = model.encode([query], normalize_embeddings=True)

scores, ids = index.search(np.array(query_vec), k=3)

print("\n🔎 Top Results:\n")

for i in ids[0]:
    result = metadata[i]
    print(f"Title: {result['title']}")
    print(f"Section: {result['section']}")
    print(result['text'][:400])
    print("\n-----------------\n")
