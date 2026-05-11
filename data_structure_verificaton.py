import json

with open(caption_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=== TOP-LEVEL TYPE ===")
print(type(data))

print("\n=== FIRST 3 ENTRIES ===")
for i, entry in enumerate(data[:3]):
    print(f"\n--- Entry {i} ---")
    print(entry)
    print(f"  Keys: {entry.keys() if isinstance(entry, dict) else 'NOT A DICT'}")
    print(f"  Filename: {entry.get('filename', 'KEY NOT FOUND')}")
    cap = entry.get('caption', 'KEY NOT FOUND')
    print(f"  Caption type: {type(cap)}")
    print(f"  Caption value: {cap}")
