cd ./workspace/main
bash launch_multi.sh tasks/common_sense_reasoning/train_qwen0.5lora_WinoGrande.py 4

python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullWinoGrandeBERT400 --test_dataset WinoGrande
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullWinoGrandeBERT800 --test_dataset WinoGrande
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullWinoGrandeBERT1200 --test_dataset WinoGrande
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullWinoGrandeBERT1600 --test_dataset WinoGrande
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullWinoGrandeBERT2000 --test_dataset WinoGrande
