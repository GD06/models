
logdir=${LOG_OUTPUT_DIR}/tmp
c=configs

configs=$c/tcn_default_inception.yml,$c/pouring_inception.yml
cp -r ${MODEL_INPUT_DIR}/tcn/inception $logdir

bazel-bin/eval \
    --config_paths $configs --logdir $logdir \
    --tfrecords ${DATA_INPUT_DIR}/tcn/box_to_clear0_real.tfrecord \
    --model_name tcn_inception_v3

rm -rf $logdir

configs=$c/tcn_default_resnet.yml,$c/pouring_resnet.yml
cp -r ${MODEL_INPUT_DIR}/tcn/resnet $logdir

bazel-bin/eval \
    --config_paths $configs --logdir $logdir \
    --tfrecords ${DATA_INPUT_DIR}/tcn/box_to_clear0_real.tfrecord \
    --model_name tcn_resnet_v2

rm -rf $logdir
