import faiss
import numpy as np

EMB_PATH = r"C:\rag_project\data\wikipedia_processed\embeddings_test.npy"
INDEX_PATH = r"C:\rag_project\data\faiss_index\test_index.faiss"

embeddings = np.load(EMB_PATH)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)  
# Inner Product = cosine similarity because we normalized vectors

index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

print(f"✅ FAISS index created with {index.ntotal} vectors")
