import json

count = 0
total = 0

with open("data/processed/math.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        total += 1
        item = json.loads(line)

        if "You are an AI assistant" in item["text"]:
            count += 1

print("Total examples:", total)
print("Examples containing leaked instruction text:", count)