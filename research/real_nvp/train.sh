
python3 real_nvp_multiscale_dataset.py \
    --image_size 64 \
    --hpconfig=n_scale=5,base_dim=32,clip_gradient=100,residual_blocks=4 \
    --dataset lsun \
    --traindir /home/xinfeng/tmp/real_nvp/train \
    --logdir /home/xinfeng/tmp/real_nvp/train \
    --data_path /home/xinfeng/tmp/real_nvp/celeba_train.tfrecords

