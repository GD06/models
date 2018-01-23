export CHECKPOINT_DIR=/home/xinfeng/docker-input/models/adv_imagenet_models
export DATASET_DIR=/home/xinfeng/docker-input/datasets/imagenet

python3 eval_on_adversarial.py \
    --model_name=inception_resnet_v2 \
    --checkpoint_path=${CHECKPOINT_DIR}/ens_adv_inception_resnet_v2.ckpt \
    --dataset_dir=${DATASET_DIR} \
    --batch_size=50 \
    --adversarial_method=stepllnoise \
    --adversarial_eps=16

unset CHECKPOINT_DIR
unset DATASET_DIR
