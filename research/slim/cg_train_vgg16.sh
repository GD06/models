mkdir -p ${LOG_OUTPUT_DIR}/tmp

DATASET_DIR=/home/xinfeng/docker-input/datasets/ImageNet2012/output
TRAIN_DIR=${LOG_OUTPUT_DIR}/tmp

python3 train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=vgg_16

rm -rf ${LOG_OUTPUT_DIR}/tmp
