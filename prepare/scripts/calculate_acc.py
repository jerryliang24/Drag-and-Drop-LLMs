import json
import re

import fire

datasets = ["ARC-c", "ARC-e", "PIQA", "OBQA", "HellaSwag", "WinoGrande", "BoolQ"]
numbers = ["1", "2", "3", "4", "["]
options = ["A", "B", "C", "D"]
logic = ["True", "False", "false", "true"]


def extract_answer(response: str):
    pattern = r"([A-Za-z]+):"
    matches = re.findall(pattern, response)
    return matches


def extract_answer_base(response: str):
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, response)
    return matches


def check_validity(file: str):
    f = open(file, "r")
    count, valid, acc = 0, 0.0, 0.0
    for line in f:
        count += 1

        answer_dict = json.loads(line.strip())
        response = answer_dict["predict"][:20]

        if "BoolQ" in file:
            label = extract_answer_base(answer_dict["label"])[0]
            pred = response.split("ns: ")[-1].split("\n")[0]
            if len(pred) > 5:
                pred = [e for e in logic if e in pred]
                pred = pred[0] if len(pred) > 0 else ""
            if pred in logic:
                valid += 1
                if pred.lower() == label.lower():
                    acc += 1
        else:
            label = answer_dict["label"][1]
            if len(response) < 2:
                if len(response) == 0:
                    continue
                pred = response[0]

            # if False: pass
            else:
                if response[1] in numbers + options:
                    valid += 1
                    if response[1] in options:
                        pred = response[1]
                    elif response[1] == "[":
                        try:
                            pred = extract_answer_base(response)[0]
                        except:
                            pred = ""
                    else:
                        pred = options[numbers.index(response[1])]
                    # print(pred,label)
                    if pred == label:
                        acc += 1
    print(f"valid:{valid/count}, acc:{acc/count}")


if __name__ == "__main__":
    fire.Fire(check_validity)
