import json

input_file = "vision_aid_1000.jsonl"
output_file = "vision_aid_for_ft.jsonl"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        instruction = data["instruction"]
        input_data = data["input"]
        output_data = data["output"]

        prompt = f"<s>[INST] {instruction}\n{input_data} [/INST] {output_data}</s>"

        json.dump({"text": prompt}, outfile, ensure_ascii=False)
        outfile.write("\n")
