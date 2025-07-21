cd ./workspace/main
bash launch_multi.sh tasks/coding/train_qwen1.5lora_Coding.py 8

python generate/qwen1.5lora_generation_coding.py --eval_dataset Coding1.5B500 --test_dataset Human_Eval
python generate/qwen1.5lora_generation_coding.py --eval_dataset Coding1.5B1000 --test_dataset Human_Eval
python generate/qwen1.5lora_generation_coding.py --eval_dataset Coding1.5B1500 --test_dataset Human_Eval
python generate/qwen1.5lora_generation_coding.py --eval_dataset Coding1.5B2000 --test_dataset Human_Eval
python generate/qwen1.5lora_generation_coding.py --eval_dataset Coding1.5B2500 --test_dataset Human_Eval
