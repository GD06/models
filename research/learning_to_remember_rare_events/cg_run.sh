
mkdir -p ${LOG_OUTPUT_DIR}/tmp

python3 train.py --memory_size=8192 \
    --batch_size=16 --validation_length=50 \
    --episode_width=5 --episode_length=30 \
    --save_dir=${LOG_OUTPUT_DIR}/tmp

rm -rf ${LOG_OUTPUT_DIR}/tmp
