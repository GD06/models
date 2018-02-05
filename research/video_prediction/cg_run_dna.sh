mkdir -p ${LOG_OUTPUT_DIR}/tmp

python3 prediction_train.py \
    --data_dir=${DATA_INPUT_DIR}/video_prediction \
    --model=DNA \
    --output_dir=${LOG_OUTPUT_DIR}/tmp \
    --event_log_dir=${LOG_OUTPUT_DIR}/tmp \
    --num_iterations=10 \
    --sequence_length=10 \
    --context_frames=2 \
    --use_state=1 \
    --num_tasks=10 \
    --schedsamp_k=900.0 \
    --train_val_split=0.95 \
    --batch_size=32 \
    --num_masks=1 \
    --learning_rate=0.001

rm -rf ${LOG_OUTPUT_DIR}/tmp


