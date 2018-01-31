
mkdir -p ${LOG_OUTPUT_DIR}/tmp

bazel-bin/dp_sgd/dp_mnist/dp_mnist \
    --training_data_path=${DATA_INPUT_DIR}/mnist/mnist_train.tfrecord \
    --eval_data_path=${DATA_INPUT_DIR}/mnist/mnist_test.tfrecord \
    --save_path=${LOG_OUTPUT_DIR}/tmp

rm -rf ${LOG_OUTPUT_DIR}/tmp
