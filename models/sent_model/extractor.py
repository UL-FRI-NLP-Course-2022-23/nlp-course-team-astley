import csv
import json

input_file_path = 'output.json'
output_file_path = 'output_updated.json'
encoding = 'ISO-8859-1'
scores_path = "./models/sets/traits_scores.csv"
trait_dict = {}

with open(scores_path, 'r', encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        key = row["Trait"]
        value = row["Score"]
        trait_dict[key] = float(value)

# Read JSON data from input file
with open(input_file_path, 'r', encoding=encoding) as json_file:
    data = json.load(json_file)
count_f = 0
for obj in data:
    score = 0
    count = 0
    sentence = obj["sentence"].lower()
    
    words = sentence.split()
    for word in words:
        if word in trait_dict:
            count += 1
            score += trait_dict[word]
    obj["score"] = score
    obj["trust"] = count
    obj["tagged_by"] = "automatic"
# with open(output_file_path, 'w', encoding=encoding) as json_output_file:
#     json.dump(data, json_output_file, ensure_ascii=False)