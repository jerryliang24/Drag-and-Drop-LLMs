LORA_PATH="lora_path"
MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUT_PATH="merged_ckpt"

cd LiveCodeBench

python ../merge_lora.py --lora_model_path $LORA_PATH --base_model_name $MODEL_PATH --output_dir $OUT_PATH

python -m lcb_runner.runner.main \
 --model $MODEL_PATH \
 --scenario codegeneration \
 --local_model_path $OUT_PATH

rm -rf $OUT_PATH
