python3 demo_inference.py \
    --checkpoint=$MODEL_INPUT_DIR/attention_ocr/model.ckpt-399731 \
    --batch_size=1 \
    --image_path_pattern=./OCRExample_%02d.png
