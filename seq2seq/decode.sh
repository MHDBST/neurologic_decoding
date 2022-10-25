#!/usr/bin/env bash

DATA_DIR='../dataset/clean'
DEVICES=$1
SPLIT=$2
MODEL_DIR=$3
OUTPUT_DIR=$4
OUTPUT_FILE=$5


INPUT="${DATA_DIR}/commongen.${SPLIT}.src_alpha.txt"
TARGET="${DATA_DIR}/commongen.${SPLIT}.tgt.txt"

# MODEL_RECOVER_PATH="${MODEL_DIR}/best_tfmr"
MODEL_RECOVER_PATH="${MODEL_DIR}"

# bart
CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py \
  --model_name ${MODEL_RECOVER_PATH} \
  --input_path ${INPUT} --reference_path ${TARGET} \
  --constraint_file ${DATA_DIR}/constraint/${SPLIT}.constraint.json \
  --min_tgt_length 20 --max_tgt_length 256 \
  --bs 4 --beam_size 50 --length_penalty 0.1 --ngram_size 10 \
  --prune_factor 50 --sat_tolerance 2 --beta 2 --early_stop 1.5 --pos_constraint \
  --save_path "${OUTPUT_DIR}/${OUTPUT_FILE}.${SPLIT}" --score_path "${OUTPUT_DIR}/${OUTPUT_FILE}.json"

#### use this one to satisfy positive constraint --pos_constraint \
#### use this one to run from a specific index --index



# t5-large
#CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py \
#  --model_name ${MODEL_RECOVER_PATH} \
#  --input_path ${INPUT} --reference_path ${TARGET} \
#  --constraint_file ${DATA_DIR}/constraint/${SPLIT}.constraint.json \
#  --min_tgt_length 5 --max_tgt_length 32 \
#  --bs 32 --beam_size 20 --length_penalty 0.2 --ngram_size 3 \
#  --prune_factor 50 --sat_tolerance 2 --beta 0 --early_stop 1.5 \
#  --save_path "${OUTPUT_DIR}/${OUTPUT_FILE}.${SPLIT}" --score_path "${OUTPUT_DIR}/${OUTPUT_FILE}.json"