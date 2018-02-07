
MODEL_PATH=${MODEL_INPUT_DIR}/skip_thoughts

./run_infer.py ${MODEL_PATH}/skip_thoughts_uni_2017_02_02/vocab.txt \
    ${MODEL_PATH}/skip_thoughts_uni_2017_02_02/embeddings.npy \
    ${MODEL_PATH}/skip_thoughts_uni_2017_02_02/model.ckpt-501424 \
    ${MODEL_PATH}/rt-polaritydata \
    --model_name=skipt_thoughts_uni

./run_infer.py ${MODEL_PATH}/skip_thoughts_bi_2017_02_16/vocab.txt \
    ${MODEL_PATH}/skip_thoughts_bi_2017_02_16/embeddings.npy \
    ${MODEL_PATH}/skip_thoughts_bi_2017_02_16/model.ckpt-500008 \
    ${MODEL_PATH}/rt-polaritydata \
    --model_name=skipt_thoughts_bi \
    --bidirect=True
