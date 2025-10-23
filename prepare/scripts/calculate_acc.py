import json
import re

import fire

datasets = ["ARC-c", "ARC-e", "PIQA", "OBQA", "HellaSwag", "WinoGrande", "BoolQ", "science-dataset"]
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


def extract_option_keywords(prompt: str):
    """extract each choice's key words from prompts"""
    keywords = {}
    lines = prompt.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if line.startswith(('A:', 'B:', 'C:', 'D:')):
            option = line[0]
            content = line[3:].strip()
            
            keywords[option] = content
    return keywords


def match_option_by_keyword_count(response: str, keywords: dict):
    response_lower = response.lower()
    option_counts = {}
    
    for option, words in keywords.items():
        count = 0
        for word in words:
            count += response_lower.count(word)
        option_counts[option] = count
    
    # return the choice that occurs most often
    if option_counts and max(option_counts.values()) > 0:
        return max(option_counts, key=option_counts.get)
    return None


def calculate_acc(file: str):
    
    test_dataset = [d for d in datasets if d in file][0]
    test_data = {}
    
    if "science-dataset" in file :
        test_dataset = "science-dataset"
        with open(f"data/{test_dataset}.json", 'r') as f:
            test_data_list = json.load(f)
            for i, item in enumerate(test_data_list):
                test_data[i] = item
                
    else:   
        with open(f"data/{test_dataset}_test.json", 'r') as f:
            test_data_list = json.load(f)
            for i, item in enumerate(test_data_list):
                test_data[i] = item
                
    f = open(file, "r")
    count, valid, acc = 0, 0.0, 0.0
    for line in f:
        count += 1

        answer_dict = json.loads(line.strip())
        response = answer_dict["predict"][:20]

        if "BoolQ" in file and "science-dataset" not in file:
            label = extract_answer_base(answer_dict["label"])[0]
            pred = response
            
            found_logic = None
            for logic_word in logic:
                if logic_word in pred:
                    found_logic = logic_word
                    break
            
            if found_logic:
                valid += 1
                if found_logic.lower() == label.lower():
                    acc += 1
        else:
            label = answer_dict["label"][1]
            pred = ""
            
            if len(response) == 0:
                continue
            elif len(response) == 1:
                # check whether the first char is valid
                if response[0] in options:
                    pred = response[0]
                    valid += 1
            else:
                # check whether the second char is valid 
                if response[1] in numbers + options:
                    valid += 1
                    if response[1] in options:
                        pred = response[1]
                    elif response[1] == "[":
                        try:
                            extracted = extract_answer_base(response)
                            if extracted:  # 确保列表不为空
                                pred = extracted[0]
                            else:
                                pred = ""
                        except:
                            pred = ""
                    else:
                        # 数字转换为选项
                        try:
                            pred = options[numbers.index(response[1])]
                        except (ValueError, IndexError):
                            pred = ""
                else:
                    # directly pairing ABCD
                    for option in options:
                        if option in response:
                            pred = option
                            valid += 1
                            break
                        
                    
                    if not pred and (count-1) in test_data:
                        prompt = test_data[count-1].get("prompt", "")
                        if prompt:
                            keywords = extract_option_keywords(prompt)
                            
                            if keywords:
                                # use full response instead of partial
                                full_response = answer_dict["predict"]
                                matched_option = match_option_by_keyword_count(full_response, keywords)
                                
                                if matched_option:
                                    pred = matched_option
                                    valid += 1
            
            if pred and pred == label:
                acc += 1
    # print(f"valid:{valid/count}, acc:{acc/count}")
    return valid/count, acc/count


if __name__ == "__main__":
    fire.Fire(calculate_acc)
