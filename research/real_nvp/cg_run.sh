
mkdir -p ${LOG_OUTPUT_DIR}/tmp

python3 real_nvp_multiscale_dataset.py \
    --image_size 64 \
    --hpconfig=n_scale=5,base_dim=32,clip_gradient=100,residual_blocks=4 \
    --dataset celeba \
    --traindir ${MODEL_INPUT_DIR}/real_nvp \
    --logdir ${LOG_OUTPUT_DIR}/tmp \
    --data_path ${DATA_INPUT_DIR}/real_nvp/celeba_valid.tfrecords \
    --eval_set_size 19867 \
    --mode eval

rm -rf ${LOG_OUTPUT_DIR}/tmp
