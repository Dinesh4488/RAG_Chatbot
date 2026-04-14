import orjson
from tqdm import tqdm
import os

INPUT_FILE = r"C:\rag_project\data\wikipedia_raw\unpacked\enwiki_namespace_0_0.jsonl"
OUTPUT_FILE = r"C:\rag_project\data\wikipedia_processed\chunks_0.jsonl"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


def extract_paragraphs(article):

    title = article.get("name", "")
    sections = article.get("sections", [])

    chunks = []

    for sec in sections:

        section_name = sec.get("name", "Unknown")

        if "has_parts" not in sec:
            continue

        for part in sec["has_parts"]:

            if part.get("type") != "paragraph":
                continue

            text = part.get("value", "").strip()

            # 🔴 Skip garbage
            if len(text) < 120:
                continue

            chunk = {
                "title": title,
                "section": section_name,
                "text": text
            }

            chunks.append(chunk)

    return chunks


def process_file():

    count = 0

    with open(INPUT_FILE, "rb") as f, open(OUTPUT_FILE, "wb") as out:

        for line in tqdm(f):

            article = orjson.loads(line)

            paragraphs = extract_paragraphs(article)

            for p in paragraphs:
                out.write(orjson.dumps(p) + b"\n")
                count += 1

    print(f"\n✅ Total chunks created: {count}")


if __name__ == "__main__":
    process_file()
