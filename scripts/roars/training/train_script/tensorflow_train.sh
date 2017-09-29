#!/bin/bash

echo "Train starting"
#check if all the variables exists and are not null
BASE_CODE_DIR=${TF_MODEL_DIR:?}
PIPELINE_CONFIG_PATH=${CONFIG_PATH:?}
OUT_FOLDER=${OUTPUT_FOLDER:?}

PYTHON_SCRIPT="object_detection/train.py"

#change current working dir to tensorflow/model/research
cd ${BASE_CODE_DIR}

#start training
python ${PYTHON_SCRIPT} --logtostderr --pipeline_config_path=$PIPELINE_CONFIG_PATH --train_dir=$OUT_FOLDER
