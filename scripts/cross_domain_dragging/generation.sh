cd ./workspace/main

python generate/qwen0.5lora_generation_cross.py --eval_dataset ARC-c4000 
python generate/qwen0.5lora_generation_cross.py --eval_dataset ARC-e4000 
python generate/qwen0.5lora_generation_cross.py --eval_dataset OBQA1000 
python generate/qwen0.5lora_generation_cross.py --eval_dataset HellaSwag600 
python generate/qwen0.5lora_generation_cross.py --eval_dataset PIQA5000 
python generate/qwen0.5lora_generation_cross.py --eval_dataset BoolQ1000 
python generate/qwen0.5lora_generation_cross.py --eval_dataset WinoGrande1000 