from datasets import load_from_disk
import os
from pathlib import Path
import json
import re
from unidecode import unidecode

RAW_PATH = "C:/rag_project/data/wikipedia_raw/"
OUT_PATH = "C:/rag_project/data/wikipedia_processed/"

os.makedirs(OUT_PATH, exist_ok=True)

wiki = load_from_disk(RAW_PATH)

def clean(text):
    text = unidecode(text)
    text = re.sub(r'\[\d+\]', '', text)  # remove citation markers
    return text.strip()

doc_id = 0

for row in wiki:
    title = row["title"]
    text = clean(row["text"])
    
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    
    for p in paragraphs:
        out = {
            "id": f"wiki_{doc_id}",
            "title": title,
            "paragraph": p
        }
        with open(f"{OUT_PATH}/{doc_id}.json", "w", encoding="utf-8") as f:
            json.dump(out, f)
        doc_id += 1

print(f"Processed {doc_id} paragraphs.")
