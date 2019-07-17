mkdir -p ${LOG_OUTPUT_DIR}/tmp

bazel-bin/seq2seq_attention \
    --mode=train \
    --article_key=article \
    --abstract_key=abstract \
    --data_path=data/data \
    --vocab_path=data/vocab \
    --log_root=${MODEL_INPUT_DIR}/textsum \
    --train_dir=${LOG_OUTPUT_DIR}/tmp \
    --num_gpus=1

rm -rf ${LOG_OUTPUT_DIR}/tmp
