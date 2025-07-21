cd ./workspace/main
bash launch_multi.sh ./tasks/multimodal/train_qwen3vl-lora_multimodal.py 8

python generate/qwen3vl-lora_generation_multimodal.py --eval_dataset multimodal1000 --test_dataset Math-Vision
python generate/qwen3vl-lora_generation_multimodal.py --eval_dataset multimodal2000 --test_dataset Math-Vision
python generate/qwen3vl-lora_generation_multimodal.py --eval_dataset multimodal3000 --test_dataset Math-Vision
python generate/qwen3vl-lora_generation_multimodal.py --eval_dataset multimodal4000 --test_dataset Math-Vision
python generate/qwen3vl-lora_generation_multimodal.py --eval_dataset multimodal5000 --test_dataset Math-Vision

python generate/qwen3vl-lora_generation_multimodal.py --eval_dataset multimodal1000 --test_dataset Math-Vista
python generate/qwen3vl-lora_generation_multimodal.py --eval_dataset multimodal2000 --test_dataset Math-Vista
python generate/qwen3vl-lora_generation_multimodal.py --eval_dataset multimodal3000 --test_dataset Math-Vista
python generate/qwen3vl-lora_generation_multimodal.py --eval_dataset multimodal4000 --test_dataset Math-Vista
python generate/qwen3vl-lora_generation_multimodal.py --eval_dataset multimodal5000 --test_dataset Math-Vista
