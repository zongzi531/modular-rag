import csv
import json

csvfile = open("result.csv", "r")
jsonfile = open("result.json", "w")

fieldnames = ("index", "question", "contexts", "ground_truth", "evolution_type", "metadata", "episode_done", "answer", "context_precision", "faithfulness", "answer_relevancy", "context_recall", "context_entity_recall", "answer_similarity", "answer_correctness")
reader = csv.DictReader(csvfile, fieldnames)
rows = list(reader)
totalrows = len(rows)

jsonfile.write("[\n")

for i, row in enumerate(rows):
    if i == 0:
        continue
    json.dump(row, jsonfile, ensure_ascii=False, indent=2)
    if (i < totalrows - 1):
        jsonfile.write(",")
    jsonfile.write("\n")

jsonfile.write("]\n")
