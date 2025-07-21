import os

import pandas as pd

# 读取单个 Parquet 文件
dataset = os.path.basename(__file__)[:-3].split("process_")[1]
import json

choices_ = ["A", "B", "C", "D"]
for split in ["train", "validation"]:
    df = pd.read_parquet(f"mmr1_training_datas/{dataset}/{split}.parquet")
    new_data = []
    system = "you are a helpful AI assistant, and you are going to finsh the sentence by picking one sentence from the given choices. Answer the chapital character of the choice directly. Answer true or false directly. You'll only need to answer by a single [ans] (ans is A,B,C,D or True/False)"
    for i in range(len(df)):
        # print(df.iloc[r]["ctx"])
        # if df.iloc[i]["label"]=="0" or df.iloc[i]["label"]=="4": print(df.iloc[i]["label"])
        # print(len(df.iloc[i]["endings"]))
        # break
        choices = "\n".join([choices_[j] + ": " + df.iloc[i]["endings"][j] for j in range(len(choices_))])
        question = (
            "Which solution is most appropriate for finishing the following sentence? Sentence:"
            + df.iloc[i]["ctx"]
            + "Choices:"
            + choices
        )
        answer = f'[{choices_[int(df.iloc[i]["label"])]}]'
        new_data.append({"prompt": question, "response": answer, "system": system})
    json.dump(new_data, open(f"{dataset}_{split}.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)

for split in ["test"]:
    df = pd.read_parquet(f"mmr1_training_datas/{dataset}/{split}.parquet")
    new_data = []
    system = "you are a helpful AI assistant, and you are going to finsh the sentence by picking one sentence from the given choices. Answer the chapital character of the choice directly."
    for i in range(len(df)):
        # print(df.iloc[r]["ctx"])
        # if df.iloc[i]["label"]=="0" or df.iloc[i]["label"]=="4": print(df.iloc[i]["label"])
        # print(len(df.iloc[i]["endings"]))
        # break
        choices = "\n".join([choices_[j] + ": " + df.iloc[i]["endings"][j] for j in range(len(choices_))])
        question = (
            "Which solution is most appropriate for finishing the following sentence? Sentence:"
            + df.iloc[i]["ctx"]
            + "Choices:"
            + choices
        )
        # answer = choices_[int(df.iloc[i]["label"])]
        new_data.append(
            {
                "prompt": question,
                # "response":answer,
                "system": system,
            }
        )
    json.dump(new_data, open(f"{dataset}_{split}.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)
