#!/bin/bash

CKPT="ckpts/model.safetensors"
BASE_PATH="/home/disk3_SSD/ylx/dataset_pi3"

for i in {0..29}; do
    DATA_PATH="${BASE_PATH}/visymscenes_test_${i}/input"
    SAVE_PATH="${BASE_PATH}/visymscenes_test_${i}/result.ply"

    echo "Running visymscenes_test_${i} ..."
    python example_colmap.py \
        --ckpt ${CKPT} \
        --data_path ${DATA_PATH} \
        --save_path ${SAVE_PATH}
done
