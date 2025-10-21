cd ./workspace/main

#for 5-2 dataset arrangements

python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset 5train_2test --test_dataset ARC-c
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset 5train_2test --test_dataset OBQA

#for 4-3 dataset arrangements

python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset 4train_3test --test_dataset ARC-c
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset 4train_3test --test_dataset OBQA
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset 4train_3test --test_dataset WinoGrande

# for 3-4 dataset arrangements

python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset 3train_4test --test_dataset ARC-c
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset 3train_4test --test_dataset OBQA
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset 3train_4test --test_dataset WinoGrande
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset 3train_4test --test_dataset BoolQ

# for 2-5 dataset arrangements

python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset 2train_5test --test_dataset BoolQ
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset 2train_5test --test_dataset ARC-c
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset 2train_5test --test_dataset OBQA
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset 2train_5test --test_dataset HellaSwag
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset 2train_5test --test_dataset WinoGrande
