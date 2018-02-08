
bazel run -c opt :eval_rotator -- --checkpoint_dir=${MODEL_INPUT_DIR}/ptn/rotator \
    --inp_dir=${DATA_INPUT_DIR}/ptn

bazel run -c opt :eval_ptn -- --checkpoint_dir=${MODEL_INPUT_DIR}/ptn/ptn \
    --inp_dir=${DATA_INPUT_DIR}/ptn
