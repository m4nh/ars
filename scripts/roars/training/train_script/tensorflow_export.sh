#!/bin/bash

#check if all the variables exists and are not null
BASE_CODE_DIR=${TF_MODEL_DIR:?}
PIPELINE_CONFIG_PATH=${CONFIG_PATH:?}
CHECKPOINT_FILE_PATH=${CHECKPOINT_FILE:?}
OUT_FOLDER=${OUTPUT_FOLDER:?}

PYTHON_SCRIPT="object_detection/export_inference_graph.py"

#change current working dir to tensorflow/model/research
cd ${BASE_CODE_DIR}

#start training
python ${PYTHON_SCRIPT} --input_type image_tensor --pipeline_config_path ${PIPELINE_CONFIG_PATH} --trained_checkpoint_prefix ${CHECKPOINT_FILE_PATH} --output_directory ${OUT_FOLDER}