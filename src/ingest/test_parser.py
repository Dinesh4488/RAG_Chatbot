import orjson

file_path = r"C:\rag_project\data\wikipedia_raw\unpacked\enwiki_namespace_0_0.jsonl"

count = 0

with open(file_path, "rb") as f:
    for line in f:
        article = orjson.loads(line)

        name = article.get("name")
        sections = article.get("sections", [])

        print(f"\nArticle: {name}")

        for sec in sections:
            if "has_parts" in sec:
                for part in sec["has_parts"]:
                    if part.get("type") == "paragraph":
                        text = part.get("value")
                        print("Paragraph:", text[:200])

        count += 1

        if count == 3:
            break
