import orjson
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

INPUT_FILE = r"C:\rag_project\data\wikipedia_processed\clean_chunks_0.jsonl"
OUTPUT_EMB = r"C:\rag_project\data\wikipedia_processed\embeddings_test.npy"
OUTPUT_META = r"C:\rag_project\data\wikipedia_processed\meta_test.jsonl"

MODEL_NAME = r"C:\Users\Dinesh karthik\.cache\huggingface\hub\models--BAAI--bge-small-en-v1.5\snapshots\5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

model = SentenceTransformer(MODEL_NAME)

BATCH_SIZE = 64
MAX_CHUNKS = 10000   # 🔴 SAFETY LIMIT


def stream_chunks():

    with open(INPUT_FILE, "rb") as f:
        for i, line in enumerate(f):

            if i >= MAX_CHUNKS:
                break

            yield orjson.loads(line)


def main():

    texts = []
    meta = []
    embeddings = []

    for chunk in tqdm(stream_chunks()):

        texts.append(chunk["text"])
        meta.append({
            "title": chunk["title"],
            "section": chunk["section"],
            "text": chunk["text"]
        })

        if len(texts) == BATCH_SIZE:

            emb = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            embeddings.append(emb)

            texts = []

    # last batch
    if texts:
        emb = model.encode(texts, normalize_embeddings=True)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)

    np.save(OUTPUT_EMB, embeddings)

    with open(OUTPUT_META, "wb") as f:
        for m in meta:
            f.write(orjson.dumps(m) + b"\n")

    print(f"\n✅ Created {len(embeddings)} embeddings")


if __name__ == "__main__":
    main()
