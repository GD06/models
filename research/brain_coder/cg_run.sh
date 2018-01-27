DESC="v0"
EXPS=( pg-5M topk-5M ga-5M rand-5M pg-20M topk-20M ga-20M rand-20M )
for exp in "${EXPS[@]}"
do
    ./single_task/run_eval_tasks.py \
        --exp "$exp" --iclr_tasks --desc "$DESC"
done
