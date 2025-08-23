export CUDA_VISIBLE_DEVICES=4,5,6,7

cd ./prepare

llamafactory-cli train training_scripts/common_sense_reasoning/ARC-c.yaml
llamafactory-cli train training_scripts/common_sense_reasoning/ARC-e.yaml
llamafactory-cli train training_scripts/common_sense_reasoning/BoolQ.yaml
llamafactory-cli train training_scripts/common_sense_reasoning/OBQA.yaml
llamafactory-cli train training_scripts/common_sense_reasoning/PIQA.yaml
llamafactory-cli train training_scripts/common_sense_reasoning/HellaSwag.yaml
llamafactory-cli train training_scripts/common_sense_reasoning/WinoGrande.yaml

