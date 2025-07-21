cd ./workspace/main


bash launch_multi.sh ablation/number_of_conditions/train_qwen0.5lora_148-148.py 4
bash launch_multi.sh ablation/number_of_conditions/train_qwen0.5lora_456-148.py 4
bash launch_multi.sh ablation/number_of_conditions/train_qwen0.5lora_514-148.py 4
bash launch_multi.sh ablation/number_of_conditions/train_qwen0.5lora_1044-148.py 4
bash launch_multi.sh ablation/number_of_conditions/train_qwen0.5lora_4048-148.py 4


bash launch_multi.sh ablation/number_of_conditions/train_qwen0.5lora_456-456.py 4
bash launch_multi.sh ablation/number_of_conditions/train_qwen0.5lora_514-514.py 4
