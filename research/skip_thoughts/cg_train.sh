mkdir -p ${LOG_OUTPUT_DIR}/tmp

bazel-bin/skip_thoughts/train \
    --input_file_pattern="${DATA_INPUT_DIR}/nltk_data/train/train-?????-of-00100" \
    --train_dir=${LOG_OUTPUT_DIR}/tmp

rm -rf ${LOG_OUTPUT_DIR}/tmp
