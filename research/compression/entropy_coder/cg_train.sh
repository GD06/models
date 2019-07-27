mkdir -p ${LOG_OUTPUT_DIR}/tmp

python3 ./core/entropy_coder_train.py --task=0 \
    --train_dir=${LOG_OUTPUT_DIR}/tmp \
    --model=progressive \
    --model_config=./configs/synthetic/model_config.json \
    --train_config=./configs/synthetic/train_config.json \
    --input_config=./configs/synthetic/input_config.json

rm -rf ${LOG_OUTPUT_DIR}/tmp
