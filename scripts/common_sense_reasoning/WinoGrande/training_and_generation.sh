cd ./workspace/main
bash launch_multi.sh tasks/common_sense_reasoning/train_qwen0.5lora_WinoGrande.py 4

export CUDA_VISIBLE_DEVICES=0,1
# Qwen0.5B has 14 attention heads and can only parallel on 2 or 7 GPUs

python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset WinoGrande1000 --test_dataset WinoGrande
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset WinoGrande2000 --test_dataset WinoGrande
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset WinoGrande3000 --test_dataset WinoGrande
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset WinoGrande4000 --test_dataset WinoGrande
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset WinoGrande5000 --test_dataset WinoGrande
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset WinoGrande --test_dataset WinoGrande
