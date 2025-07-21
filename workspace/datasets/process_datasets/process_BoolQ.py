import os

import pandas as pd

# 读取单个 Parquet 文件
dataset = os.path.basename(__file__)[:-3].split("process_")[1]
import json

for split in ["train", "validation"]:
    df = pd.read_parquet(f"mmr1_training_datas/{dataset}/{split}.parquet")
    new_data = []
    system = "you are a helpful AI assistant, and you are going to answer the question of the user by determining the statement is true or false. Answer true or false directly. You'll only need to answer by a single [ans] (ans is A,B,C,D or True/False)"
    for i in range(len(df)):
        question = df.iloc[i]["question"] + "?"
        answer = f'[{str(df.iloc[i]["answer"])}]'

        new_data.append({"prompt": question, "response": answer, "system": system})

    json.dump(new_data, open(f"{dataset}_{split}.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)
