
bazel run -c opt :eval_rotator -- --checkpoint_dir=${MODEL_INPUT_DIR}/ptn/rotator \
    --inp_dir=${DATA_INPUT_DIR}/ptn
