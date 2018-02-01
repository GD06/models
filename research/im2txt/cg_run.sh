
CHECKPOINT_PATH=${MODEL_INPUT_DIR}/im2txt
VOCAB_FILE=${DATA_INPUT_DIR}/im2txt/word_counts.txt
IMAGE_FILE=${DATA_INPUT_DIR}/im2txt/COCO_val2014_000000224477.jpg

bazel-bin/im2txt/run_inference \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --vocab_file=${VOCAB_FILE} \
    --input_files=${IMAGE_FILE}
