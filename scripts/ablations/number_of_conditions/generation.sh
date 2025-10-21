cd ./workspace/main

python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_128-128 --test_dataset ARC-c
python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_256-128 --test_dataset ARC-c
python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_512-128 --test_dataset ARC-c
python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_1024-128 --test_dataset ARC-c
python ablation/number_of_conditions/qwen0.5lora_generation_for_128-128.py --eval_dataset cond_2048-128 --test_dataset ARC-c

python ablation/number_of_conditions/qwen0.5lora_generation_for_256-256.py --eval_dataset cond256_256 --test_dataset ARC-c
python ablation/number_of_conditions/qwen0.5lora_generation_for_512-512.py --eval_dataset cond512_512 --test_dataset ARC-c

