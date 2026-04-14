import re
import orjson
from tqdm import tqdm

INPUT_FILE = r"C:\rag_project\data\wikipedia_processed\chunks_0.jsonl"
OUTPUT_FILE = r"C:\rag_project\data\wikipedia_processed\clean_chunks_0.jsonl"


citation_pattern = re.compile(r"\[\d+\]")
multi_space_pattern = re.compile(r"\s+")


def clean_text(text):

    # Remove citations like [1], [23]
    text = citation_pattern.sub("", text)

    # Normalize spaces
    text = multi_space_pattern.sub(" ", text)

    return text.strip()


def process():

    count = 0

    with open(INPUT_FILE, "rb") as f, open(OUTPUT_FILE, "wb") as out:

        for line in tqdm(f):

            chunk = orjson.loads(line)

            cleaned = clean_text(chunk["text"])

            # Skip if cleaning destroyed it
            if len(cleaned) < 120:
                continue

            chunk["text"] = cleaned

            out.write(orjson.dumps(chunk) + b"\n")

            count += 1

    print(f"\n✅ Clean chunks: {count}")


if __name__ == "__main__":
    process()
