
python3 names.py --mode eval \
    --checkpoint_path ${MODEL_INPUT_DIR}/namignizer/small/model-12 \
    --config small

python3 names.py --mode eval \
    --checkpoint_path ${MODEL_INPUT_DIR}/namignizer/large/model-38 \
    --config large
