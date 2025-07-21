import json

datasets = [""]

ROOT = "./prepare/data"
max_length = 0
for dataset in datasets:
    sentences = []
    datafile = f"{ROOT}/{dataset}.json"
    data = json.load(open(datafile, "r", encoding="utf-8"))
    for i in range(len(data)):
        max_length = max(len(data[i]["conversations"][0]["value"]), max_length)
print(max_length)
