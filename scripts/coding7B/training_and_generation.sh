cd ./workspace/main
bash launch_multi.sh tasks/coding/train_qwen7lora_Coding.py 8

python generate/qwen7lora_generation_coding.py --eval_dataset Coding7B1000 --test_dataset Live-Code-Bench
python generate/qwen7lora_generation_coding.py --eval_dataset Coding7B2000 --test_dataset Live-Code-Bench
python generate/qwen7lora_generation_coding.py --eval_dataset Coding7B3000 --test_dataset Live-Code-Bench
python generate/qwen7lora_generation_coding.py --eval_dataset Coding7B4000 --test_dataset Live-Code-Bench
python generate/qwen7lora_generation_coding.py --eval_dataset Coding7B5000 --test_dataset Live-Code-Bench
