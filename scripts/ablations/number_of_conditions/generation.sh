cd ./workspace/main

python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_128-1281000 --test_dataset ARC-c
python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_128-1282000 --test_dataset ARC-c
python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_128-1283000 --test_dataset ARC-c

python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_256-1281000 --test_dataset ARC-c
python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_256-1282000 --test_dataset ARC-c
python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_256-1283000 --test_dataset ARC-c

python ablation/number_of_conditions/qwen0.5lora_generation_for_256-256.py --eval_dataset cond256_2561000 --test_dataset ARC-c
python ablation/number_of_conditions/qwen0.5lora_generation_for_256-256.py --eval_dataset cond256_2562000 --test_dataset ARC-c
python ablation/number_of_conditions/qwen0.5lora_generation_for_256-256.py --eval_dataset cond256_2563000 --test_dataset ARC-c

python ablation/number_of_conditions/qwen0.5lora_generation_for_512-512.py --eval_dataset cond512_5121000 --test_dataset ARC-c
python ablation/number_of_conditions/qwen0.5lora_generation_for_512-512.py --eval_dataset cond512_5122000 --test_dataset ARC-c
python ablation/number_of_conditions/qwen0.5lora_generation_for_512-512.py --eval_dataset cond512_5123000 --test_dataset ARC-c


python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_512-1281000 --test_dataset ARC-c
python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_512-1282000 --test_dataset ARC-c
python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_512-1283000 --test_dataset ARC-c

python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_1024-1281000 --test_dataset ARC-c
python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_1024-1282000 --test_dataset ARC-c
python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_1024-1283000 --test_dataset ARC-c

python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_2048-1281000 --test_dataset ARC-c
python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_2048-1282000 --test_dataset ARC-c
python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_2048-1283000 --test_dataset ARC-c
