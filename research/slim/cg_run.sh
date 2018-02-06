mkdir -p ${LOG_OUTPUT_DIR}/tmp

python3 eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${MODEL_INPUT_DIR}/slim/inception_v4.ckpt \
    --dataset_dir=${DATA_INPUT_DIR}/imagenet \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=inception_v4 \
    --batch_size=1 \
    --max_num_batches=10 \
    --eval_dir=${LOG_OUTPUT_DIR}/tmp

python3 eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${MODEL_INPUT_DIR}/slim/inception_resnet_v2.ckpt \
    --dataset_dir=${DATA_INPUT_DIR}/imagenet \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=inception_resnet_v2 \
    --batch_size=1 \
    --max_num_batches=10 \
    --eval_dir=${LOG_OUTPUT_DIR}/tmp

python3 eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${MODEL_INPUT_DIR}/slim/vgg_16.ckpt \
    --dataset_dir=${DATA_INPUT_DIR}/imagenet \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=vgg_16 \
    --batch_size=1 \
    --max_num_batches=10 \
    --labels_offset=1 \
    --eval_dir=${LOG_OUTPUT_DIR}/tmp

python3 eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${MODEL_INPUT_DIR}/slim/vgg_19.ckpt \
    --dataset_dir=${DATA_INPUT_DIR}/imagenet \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=vgg_19 \
    --batch_size=1 \
    --max_num_batches=10 \
    --labels_offset=1 \
    --eval_dir=${LOG_OUTPUT_DIR}/tmp

python3 eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${MODEL_INPUT_DIR}/slim/resnet_v2_50/resnet_v2_50.ckpt \
    --dataset_dir=${DATA_INPUT_DIR}/imagenet \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=resnet_v2_50 \
    --batch_size=1 \
    --max_num_batches=10 \
    --eval_dir=${LOG_OUTPUT_DIR}/tmp

python3 eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${MODEL_INPUT_DIR}/slim/resnet_v2_101/resnet_v2_101.ckpt \
    --dataset_dir=${DATA_INPUT_DIR}/imagenet \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=resnet_v2_101 \
    --batch_size=1 \
    --max_num_batches=10 \
    --eval_dir=${LOG_OUTPUT_DIR}/tmp

python3 eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${MODEL_INPUT_DIR}/slim/resnet_v2_152/resnet_v2_152.ckpt \
    --dataset_dir=${DATA_INPUT_DIR}/imagenet \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=resnet_v2_152 \
    --batch_size=1 \
    --max_num_batches=10 \
    --eval_dir=${LOG_OUTPUT_DIR}/tmp

python3 eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${MODEL_INPUT_DIR}/slim/mobilenet/mobilenet_v1.ckpt \
    --dataset_dir=${DATA_INPUT_DIR}/imagenet \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=mobilenet_v1 \
    --batch_size=1 \
    --max_num_batches=10 \
    --eval_dir=${LOG_OUTPUT_DIR}/tmp

python3 eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${MODEL_INPUT_DIR}/slim/nasnet-a_large/model.ckpt \
    --dataset_dir=${DATA_INPUT_DIR}/imagenet \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=nasnet_large \
    --batch_size=1 \
    --max_num_batches=10 \
    --eval_dir=${LOG_OUTPUT_DIR}/tmp

python3 eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${MODEL_INPUT_DIR}/slim/nasnet-a_mobile/model.ckpt \
    --dataset_dir=${DATA_INPUT_DIR}/imagenet \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=nasnet_mobile \
    --batch_size=1 \
    --max_num_batches=10 \
    --eval_dir=${LOG_OUTPUT_DIR}/tmp


rm -rf ${LOG_OUTPUT_DIR}/tmp
