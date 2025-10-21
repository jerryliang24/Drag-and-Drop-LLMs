cd ./workspace/main


bash launch_multi.sh ablation/number_of_conditions/train_qwen0.5lora_128-128.py 4
bash launch_multi.sh ablation/number_of_conditions/train_qwen0.5lora_256-128.py 4
bash launch_multi.sh ablation/number_of_conditions/train_qwen0.5lora_512-128.py 4
bash launch_multi.sh ablation/number_of_conditions/train_qwen0.5lora_1024-128.py 4
bash launch_multi.sh ablation/number_of_conditions/train_qwen0.5lora_2048-128.py 4


bash launch_multi.sh ablation/number_of_conditions/train_qwen0.5lora_256-256.py 4
bash launch_multi.sh ablation/number_of_conditions/train_qwen0.5lora_512-512.py 4
