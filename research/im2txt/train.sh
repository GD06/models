MSCOCO_DIR=/home/xinfeng/mscoco-data
INCEPTION_CHECKPOINT=/home/xinfeng/tmp/im2txt/data/inception_v3.ckpt

MODEL_DIR=/home/xinfeng/tmp/im2txt/model

bazel-bin/im2txt/train \
    --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
    --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
    --train_dir="${MODEL_DIR}/train" \
    --number_of_steps=1000
