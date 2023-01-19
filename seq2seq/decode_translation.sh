#!/usr/bin/env bash


DEVICES=$1
INPUT=$2
MODEL_DIR='Helsinki-NLP/opus-mt-de-en'
TARGET=$3
CONSTRAINT_FILE=$4
OUTPUT_FILE=$5


# BEAM=100
# PRUNE=100
# LENGTHP=0.2
# SAT=2


BEAM=20
PRUNE=200
LENGTHP=0.6
SAT=2
CUDA_VISIBLE_DEVICES=${DEVICES} /home/mbastan/anaconda3/envs/hug3/bin/python3 decode_translation.py \
  --model_name ${MODEL_DIR} --task translation \
  --input_path ${INPUT} --reference_path ${TARGET} \
  --constraint_file ${CONSTRAINT_FILE} \
  --min_tgt_length 3 --max_tgt_length 256 \
  --bs 64 --beam_size $BEAM --length_penalty $LENGTHP --ngram_size 3 \
  --prune_factor $PRUNE --sat_tolerance $SAT  --beta 0 \
  --save_path "${OUTPUT_FILE}" --score_path "${OUTPUT_FILE}.json" --index 980
# --early_stop 1.5  \


# bash decode_translation.sh 2  "../translation/newstest2019-deen-in.txt" "facebook/wmt19-de-en"  "../translation/newstest2019-deen-ref.txt" "../translation/newstest2019_deen.constraint.json" ../translation/SD_out_deen.out

# bash decode_translation.sh 2  "../translation/newstest2019-deen-in.txt"  "../translation/newstest2019-deen-ref.txt" "../translation/newstest2019_deen.constraint.json" ../translation/SD_out_deen_v2.out