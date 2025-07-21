import os

import pandas as pd

# 读取单个 Parquet 文件
dataset = os.path.basename(__file__)[:-3].split("process_")[1]
import json

choices_ = ["A", "B"]
for split in ["train", "test"]:
    df = pd.read_parquet(f"mmr1_training_datas/{dataset}/{split}.parquet")
    new_data = []
    system = "you are a helpful AI assistant, and you are going to find the better solution to a specific problem from the given 2 solutions. Answer the chapital character of the better solution directly. You'll only need to answer by a single [ans] (ans is A,B,C,D or True/False)"
    for i in range(len(df)):
        choices = "\n".join([choices_[j] + ": " + df.iloc[i][f"sol{j+1}"] for j in range(len(choices_))])
        question = df.iloc[i]["goal"] + "\n" + choices
        answer = f'[{choices_[int(df.iloc[i]["label"])-1]}]'
        new_data.append({"prompt": question, "response": answer, "system": system})

    json.dump(new_data, open(f"{dataset}_{split}.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)
