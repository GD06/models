
mkdir -p ${LOG_OUTPUT_DIR}/tmp

python3 eval.py --checkpoint_dir=${MODEL_INPUT_DIR}/gan/cifar/ \
    --eval_dir=${LOG_OUTPUT_DIR}/tmp/ \
    --dataset_dir=${DATA_INPUT_DIR}/cifar \
    --max_number_of_evaluations=1

python3 eval.py --checkpoint_dir=${MODEL_INPUT_DIR}/gan/cifar/ \
    --eval_dir=${LOG_OUTPUT_DIR}/tmp/ \
    --dataset_dir=${DATA_INPUT_DIR}/cifar \
    --conditional_eval=True \
    --max_number_of_evaluations=1

rm -rf ${LOG_OUTPUT_DIR}/tmp
