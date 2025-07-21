cd ./workspace/main

#for 5-2 dataset arrangements

python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 5train_2test1000 --test_dataset ARC-c
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 5train_2test2000 --test_dataset ARC-c
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 5train_2test3000 --test_dataset ARC-c

python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 5train_2test1000 --test_dataset OBQA
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 5train_2test2000 --test_dataset OBQA
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 5train_2test3000 --test_dataset OBQA

#for 4-3 dataset arrangements

python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 4train_3test1000 --test_dataset ARC-c
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 4train_3test2000 --test_dataset ARC-c
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 4train_3test3000 --test_dataset ARC-c

python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 4train_3test1000 --test_dataset OBQA
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 4train_3test2000 --test_dataset OBQA
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 4train_3test3000 --test_dataset OBQA

python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 4train_3test1000 --test_dataset WinoGrande
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 4train_3test2000 --test_dataset WinoGrande
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 4train_3test3000 --test_dataset WinoGrande

# for 3-4 dataset arrangements

python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 3train_4test1000 --test_dataset ARC-c
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 3train_4test2000 --test_dataset ARC-c
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 3train_4test3000 --test_dataset ARC-c

python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 3train_4test1000 --test_dataset OBQA
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 3train_4test2000 --test_dataset OBQA
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 3train_4test3000 --test_dataset OBQA

python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 3train_4test1000 --test_dataset WinoGrande
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 3train_4test2000 --test_dataset WinoGrande
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 3train_4test3000 --test_dataset WinoGrande

python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 3train_4test1000 --test_dataset BoolQ
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 3train_4test2000 --test_dataset BoolQ
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 3train_4test3000 --test_dataset BoolQ

python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 2train_5test1000 --test_dataset BoolQ
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 2train_5test2000 --test_dataset BoolQ
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 2train_5test3000 --test_dataset BoolQ

python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 2train_5test1000 --test_dataset ARC-c
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 2train_5test2000 --test_dataset ARC-c
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 2train_5test3000 --test_dataset ARC-c

python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 2train_5test1000 --test_dataset OBQA
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 2train_5test2000 --test_dataset OBQA
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 2train_5test3000 --test_dataset OBQA

python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 2train_5test1000 --test_dataset HellaSwag
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 2train_5test2000 --test_dataset HellaSwag
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 2train_5test3000 --test_dataset HellaSwag

python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 2train_5test1000 --test_dataset WinoGrande
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 2train_5test2000 --test_dataset WinoGrande
python generate/qwen0.5lora_generation_for_fullBERT.py --eval_dataset 2train_5test3000 --test_dataset WinoGrande
