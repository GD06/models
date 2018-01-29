
MODEL_FILE=${MODEL_INPUT_DIR}/compression/compression_residual_gru/residual_gru.pb
IMAGE_FILE=./example.png
OUTPUT_RESULT_DIR=${LOG_OUTPUT_DIR}/tmp

mkdir -p ${OUTPUT_RESULT_DIR}

OUTPUT_RESULT_FILE=${OUTPUT_RESULT_DIR}/output_codes.npz

python3 encoder.py --input_image=$IMAGE_FILE \
    --output_codes=${OUTPUT_RESULT_FILE} \
    --iteration=15 \
    --model=${MODEL_FILE}

python3 decoder.py --input_codes=${OUTPUT_RESULT_FILE} \
    --output_directory=${OUTPUT_RESULT_DIR} \
    --model=${MODEL_FILE}

rm -rf ${OUTPUT_RESULT_DIR}
