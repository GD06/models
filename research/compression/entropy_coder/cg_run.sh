python3 ./core/entropy_coder_single.py --model=progressive \
    --model_config=./configs/synthetic/model_config.json \
    --input_codes=${DATA_INPUT_DIR}/compression/sample_0000.npz \
    --checkpoint=${MODEL_INPUT_DIR}/compression/entropy_coder_train/model.ckpt-14394
