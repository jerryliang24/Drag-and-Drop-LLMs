cd ./workspace/main

bash launch_multi.sh ablation/condition_forms/train_qwen0.5lora_Q+A.py 4
bash launch_multi.sh ablation/condition_forms/train_qwen0.5lora_Mix.py 4
bash launch_multi.sh ablation/condition_forms/train_qwen1.5Math_A.py 8
