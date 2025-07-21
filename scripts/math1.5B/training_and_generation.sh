cd ./workspace/main
bash launch_multi.sh tasks/math/train_qwen1.5lora_Math.py 8

python generate/qwen1.5lora_generation_math.py --eval_dataset Math1.5B1000 --test_dataset GSM8K
python generate/qwen1.5lora_generation_math.py --eval_dataset Math1.5B2000 --test_dataset GSM8K
python generate/qwen1.5lora_generation_math.py --eval_dataset Math1.5B3000 --test_dataset GSM8K
python generate/qwen1.5lora_generation_math.py --eval_dataset Math1.5B4000 --test_dataset GSM8K
python generate/qwen1.5lora_generation_math.py --eval_dataset Math1.5B5000 --test_dataset GSM8K

python generate/qwen1.5lora_generation_math.py --eval_dataset Math1.5B1000 --test_dataset MATH
python generate/qwen1.5lora_generation_math.py --eval_dataset Math1.5B2000 --test_dataset MATH
python generate/qwen1.5lora_generation_math.py --eval_dataset Math1.5B3000 --test_dataset MATH
python generate/qwen1.5lora_generation_math.py --eval_dataset Math1.5B4000 --test_dataset MATH
python generate/qwen1.5lora_generation_math.py --eval_dataset Math1.5B5000 --test_dataset MATH
