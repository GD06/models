python3 lm_1b_eval.py --mode=eval \
    --pbtxt=${MODEL_INPUT_DIR}/lm_1b/graph-2016-09-10.pbtxt \
    --vocab_file=${DATA_INPUT_DIR}/lm_1b/vocab-2016-09-10.txt \
    --input_data=${DATA_INPUT_DIR}/lm_1b/news.en.heldout-00000-of-00050 \
    --ckpt=${MODEL_INPUT_DIR}/lm_1b/ckpt-* \
    --max_eval_steps=0
