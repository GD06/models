
export EVAL_DIR=$LOG_OUTPUT_DIR/tmp
mkdir -p ${EVAL_DIR}

export IMDB_DATA_DIR=$DATA_INPUT_DIR/adversarial_text
export TRAIN_DIR=$MODEL_INPUT_DIR/adversarial_text

bazel run :evaluate -- \
    --eval_dir=$EVAL_DIR \
    --checkpoint_dir=$TRAIN_DIR \
    --eval_data=test \
    --run_once \
    --num_examples=25000 \
    --data_dir=$IMDB_DATA_DIR \
    --vocab_size=86934 \
    --embedding_dims=256 \
    --rnn_cell_size=1024 \
    --batch_size=256 \
    --num_timesteps=400 \
    --normalize_embeddings

rm -rf ${EVAL_DIR}

unset EVAL_DIR
unset TRAIN_DIR
unset IMDB_DATA_DIR
