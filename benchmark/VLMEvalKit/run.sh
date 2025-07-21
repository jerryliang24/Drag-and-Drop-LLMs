LORA_PATH="lora_path"
BASE_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
OUT_PATH="merged_ckpt"

python ../merge_lora.py --lora_model_path $LORA_PATH --base_model_name $BASE_MODEL --output_dir $OUT_PATH

python run.py --config config.json

rm -rf $OUT_PATH
