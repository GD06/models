
cd delf/python/examples/data

ln -s ${DATA_INPUT_DIR}/delf oxford5k_images
mkdir -p ${LOG_OUTPUT_DIR}/tmp
ln -s ${LOG_OUTPUT_DIR}/tmp oxford5k_features

cd ..
ln -s ${MODEL_INPUT_DIR}/delf parameters

python3 extract_features.py \
    --config_path delf_config_example.pbtxt \
    --list_images_path list_images.txt \
    --output_dir data/oxford5k_features

# Delete the soft link
rm parameters
rm data/oxford5k_images
rm data/oxford5k_features
# Delete the temporal directory for outputs
rm -rf ${LOG_OUTPUT_DIR}/tmp

cd ${TF_MODEL_DIR}/research/delf
