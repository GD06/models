
#mkdir -p ${LOG_OUTPUT_DIR}/tmp

#python3 run_lfads.py --kind=train \
#    --data_dir=${DATA_INPUT_DIR}/rnn_synth_data_v1.0/ \
#    --data_filename_stem=chaotic_rnn_inputs_g2p5 \
#    --lfads_save_dir=${LOG_OUTPUT_DIR}/tmp \
#    --co_dim=1 \
#    --factors_dim=20 \
#    --output_dist=poisson \
#    --model_name=lfads_chaotic_rnn_inputs_g2p5

#rm -rf ${LOG_OUTPUT_DIR}/tmp

#mkdir -p ${LOG_OUTPUT_DIR}/tmp

#python3 run_lfads.py --kind=train \
#    --data_dir=${DATA_INPUT_DIR}/rnn_synth_data_v1.0/ \
#    --data_filename_stem=chaotic_rnn_multisession \
#    --lfads_save_dir=${LOG_OUTPUT_DIR}/tmp \
#    --factors_dim=10 \
#    --output_dist=poisson \
#    --model_name=lfads_chaotic_rnn_multisession

#rm -rf ${LOG_OUTPUT_DIR}/tmp

#mkdir -p ${LOG_OUTPUT_DIR}/tmp

#python3 run_lfads.py --kind=train \
#    --data_dir=${DATA_INPUT_DIR}/rnn_synth_data_v1.0/ \
#    --data_filename_stem=itb_rnn \
#    --lfads_save_dir=${LOG_OUTPUT_DIR}/tmp \
#    --co_dim=1 \
#    --factors_dim=20 \
#    --controller_input_lag=0 \
#    --output_dist=poisson \
#    --model_name=lfads_itb_rnn

#rm -rf ${LOG_OUTPUT_DIR}/tmp

mkdir -p ${LOG_OUTPUT_DIR}/tmp

python3 run_lfads.py --kind=train \
    --data_dir=${DATA_INPUT_DIR}/rnn_synth_data_v1.0/ \
    --data_filename_stem=chaotic_rnns_labeled \
    --lfads_save_dir=${LOG_OUTPUT_DIR}/tmp \
    --co_dim=0 \
    --factors_dim=20 \
    --controller_input_lag=0 \
    --ext_input_dim=1 \
    --output_dist=poisson \
    --model_name=lfads_chaotic_rnns_labeled

rm -rf ${LOG_OUTPUT_DIR}/tmp

