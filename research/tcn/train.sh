
logdir=/home/xinfeng/tmp/tcn_model
c=configs
configs=$c/tcn_default_resnet.yml,$c/pouring_resnet.yml

bazel-bin/train \
    --config_paths $configs --logdir $logdir
