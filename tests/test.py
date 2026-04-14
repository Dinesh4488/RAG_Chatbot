# import json

# file_path = "C:/rag_project/data/wikipedia_raw/unpacked/enwiki_namespace_0_0.jsonl"
# with open(file_path, 'r', encoding='utf-8') as f:
#     line = f.readline()
#     sample = json.loads(line)
#     print("Description: ",sample.get("description"))
#     print("Section: ",[s['name'] for s in sample['sections'][:5]])
    # print(list(sample.keys()))
    # print(sample['sections'][0].keys())

# print("Description:",sample.get("description"))

# for sec in sample['sections'][:10]:
#     print(sec['name'],sec.keys())

# has_parts_sections=[sec for sec in sample['sections'] if 'has_parts' in sec]
# print("sections with text:",len(has_parts_sections))
# if has_parts_sections:
#     print(has_parts_sections[0])

# for part in sample['sections'][0]:
#     print(part)

# from sentence_transformers import SentenceTransformer
# model=SentenceTransformer("BAAI/bge-small-en-v1.5")
# print("loaded")

