
mkdir ${LOG_OUTPUT_DIR}/tmp

python3 swivel.py --input_base_path ${DATA_INPUT_DIR}/swivel_data \
    --output_base_path ${LOG_OUTPUT_DIR}/tmp \
    --submatrix_rows 16384 \
    --submatrix_cols 16384

rm -rf ${LOG_OUTPUT_DIR}/tmp
