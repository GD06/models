mkdir -p ${LOG_OUTPUT_DIR}/tmp

./run_infer.py ssd_mobilenet_v1_coco ${LOG_OUTPUT_DIR}/tmp
./run_infer.py ssd_inception_v2_coco ${LOG_OUTPUT_DIR}/tmp
./run_infer.py faster_rcnn_inception_v2_coco ${LOG_OUTPUT_DIR}/tmp
./run_infer.py faster_rcnn_resnet50_coco ${LOG_OUTPUT_DIR}/tmp
./run_infer.py rfcn_resnet101_coco ${LOG_OUTPUT_DIR}/tmp

./run_infer.py faster_rcnn_resnet101_coco ${LOG_OUTPUT_DIR}/tmp
./run_infer.py faster_rcnn_inception_resnet_v2_atrous_coco ${LOG_OUTPUT_DIR}/tmp
./run_infer.py faster_rcnn_nas_coco ${LOG_OUTPUT_DIR}/tmp

rm -rf ${LOG_OUTPUT_DIR}/tmp
