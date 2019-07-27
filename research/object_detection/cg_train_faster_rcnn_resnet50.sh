mkdir -p ${LOG_OUTPUT_DIR}/tmp

python3 train.py \
    --logtostderr \
    --pipeline_config_path=/home/xinfeng/docker-input/models/object_detection/faster_rcnn_resnet50_coco_2017_11_08/pipeline.config \
    --train_dir=${LOG_OUTPUT_DIR}/tmp

rm -rf ${LOG_OUTPUT_DIR}/tmp
