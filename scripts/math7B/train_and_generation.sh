cd ./workspace/main
bash launch_multi.sh tasks/math/train_qwen7lora_Math.py 8

python generate/qwen7lora_generation_math.py --eval_dataset Math7B1000 --test_dataset GSM8K
python generate/qwen7lora_generation_math.py --eval_dataset Math7B2000 --test_dataset GSM8K
python generate/qwen7lora_generation_math.py --eval_dataset Math7B3000 --test_dataset GSM8K
python generate/qwen7lora_generation_math.py --eval_dataset Math7B4000 --test_dataset GSM8K
python generate/qwen7lora_generation_math.py --eval_dataset Math7B5000 --test_dataset GSM8K

python generate/qwen7lora_generation_math.py --eval_dataset Math7B1000 --test_dataset MATH
python generate/qwen7lora_generation_math.py --eval_dataset Math7B2000 --test_dataset MATH
python generate/qwen7lora_generation_math.py --eval_dataset Math7B3000 --test_dataset MATH
python generate/qwen7lora_generation_math.py --eval_dataset Math7B4000 --test_dataset MATH
python generate/qwen7lora_generation_math.py --eval_dataset Math7B5000 --test_dataset MATH
