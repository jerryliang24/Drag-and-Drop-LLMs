import os

import pandas as pd

# 读取单个 Parquet 文件
dataset = os.path.basename(__file__)[:-3].split("process_")[1]
import json

for split in ["train", "test"]:
    df = pd.read_parquet(f"mmr1_training_datas/{dataset}/{split}.parquet")
    new_data = []
    system = "you are a helpful AI assistant, and you are going to answer the question of the user by picking one answer among the given choices. Answer the chapital character of the choice directly. You'll only need to answer by a single [ans] (ans is A,B,C,D or True/False)"
    for i in range(len(df)):
        choices = "\n".join(
            [
                df.iloc[i]["choices"]["label"][j] + ": " + df.iloc[i]["choices"]["text"][j]
                for j in range(len(df.iloc[i]["choices"]["label"]))
            ]
        )
        question = df.iloc[i]["question_stem"] + "\n" + choices
        answer = f'[{df.iloc[i]["answerKey"]}]'
        new_data.append({"prompt": question, "response": answer, "system": system})

    json.dump(new_data, open(f"{dataset}_{split}.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)
