mkdir -p ${LOG_OUTPUT_DIR}/tmp

bazel-bin/seq2seq_attention \
    --mode=eval \
    --article_key=article \
    --abstract_key=abstract \
    --data_path=data/data \
    --vocab_path=data/vocab \
    --log_root=${MODEL_INPUT_DIR}/textsum \
    --eval_root=${LOG_OUTPUT_DIR}/tmp

rm -rf ${LOG_OUTPUT_DIR}/tmp
