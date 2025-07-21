cd ./workspace/main

#for Q+A condition arrangements

python ablation/condition_forms/qwen0.5lora_generation_cond_QA.py --eval_dataset Cond_Q_A1000 --test_dataset ARC-c
python ablation/condition_forms/qwen0.5lora_generation_cond_QA.py --eval_dataset Cond_Q_A2000 --test_dataset ARC-c
python ablation/condition_forms/qwen0.5lora_generation_cond_QA.py --eval_dataset Cond_Q_A3000 --test_dataset ARC-c

# #for Mix condition arrangements

python ablation/condition_forms/qwen0.5lora_generation_cond_Mix.py --eval_dataset Cond_Mix1000 --test_dataset ARC-c
python ablation/condition_forms/qwen0.5lora_generation_cond_Mix.py --eval_dataset Cond_Mix2000 --test_dataset ARC-c
python ablation/condition_forms/qwen0.5lora_generation_cond_Mix.py --eval_dataset Cond_Mix3000 --test_dataset ARC-c

python ablation/condition_forms/qwen1.5Math_generation_cond_A.py --eval_dataset Cond_A1000 --test_dataset GSM8K
python ablation/condition_forms/qwen1.5Math_generation_cond_A.py --eval_dataset Cond_A2000 --test_dataset GSM8K
python ablation/condition_forms/qwen1.5Math_generation_cond_A.py --eval_dataset Cond_A3000 --test_dataset GSM8K
