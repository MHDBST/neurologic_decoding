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
# MODEL_RECOVER_PATH="${MODEL_DIR}/model_file"
MODEL_RECOVER_PATH="${MODEL_DIR}"
# length_penalty: < 1.0  encourage the model to generate shorter sequences, > 1.0 encourage the model to produce longer sequences.
# bart
# CUDA_VISIBLE_DEVICES=${DEVICES} python3 decode.py \
#   --model_name ${MODEL_RECOVER_PATH} \
#   --input_path ${INPUT} --reference_path ${TARGET} \
#   --constraint_file ${DATA_DIR}/constraint/${SPLIT}.constraint.json \
#   --min_tgt_length 5 --max_tgt_length 40 \
#   --bs 1 --beam_size 50 --length_penalty 0.8 --ngram_size 3 \
#   --prune_factor 100 --sat_tolerance 1 --beta 0 --early_stop 1.5 \
#   --save_path "${OUTPUT_DIR}/${OUTPUT_FILE}.${SPLIT}" --score_path "${OUTPUT_DIR}/${OUTPUT_FILE}.json"
#70 80 90 100
# 10 20 30 40 50 60 70 80 90 100
for BEAM in 50
do
  for PRUNE in 50 60 70 80 90 100
  do
    for LENGTHP in 0.5 0.6 0.7 0.8 0.9
    do
      for SAT in 1 2 3
      do
        OUT_FILE="${OUTPUT_DIR}/${OUTPUT_FILE}_${BEAM}_${PRUNE}_${LENGTHP}_${SAT}.${SPLIT}"
        if [ ! -f $OUT_FILE ]; then
          CUDA_VISIBLE_DEVICES=${DEVICES} python3 decode.py \
            --model_name ${MODEL_RECOVER_PATH} \
            --input_path ${INPUT} --reference_path ${TARGET} \
            --constraint_file ${DATA_DIR}/constraint/${SPLIT}.constraint.json \
            --min_tgt_length 8 --max_tgt_length 64 \
            --bs 8 --beam_size $BEAM --length_penalty $LENGTHP --ngram_size 3 \
            --prune_factor $PRUNE --sat_tolerance $SAT --beta 0 --early_stop 1.5 \
            --save_path "${OUTPUT_DIR}/${OUTPUT_FILE}_${BEAM}_${PRUNE}_${LENGTHP}_${SAT}.${SPLIT}" --score_path "${OUTPUT_DIR}/${OUTPUT_FILE}_${BEAM}_${PRUNE}_${LENGTHP}_${SAT}.json"
        fi
      done
    done
  done
done
  # --min_tgt_length 20 --max_tgt_length 32 \
  # --bs 32 --beam_size 20 --length_penalty 0.1 --ngram_size 3 \
  # --prune_factor 50 --sat_tolerance 2  --beta 0 --early_stop 1.5

# t5-large
#CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py \
#  --model_name ${MODEL_RECOVER_PATH} \
#  --input_path ${INPUT} --reference_path ${TARGET} \
#  --constraint_file ${DATA_DIR}/constraint/${SPLIT}.constraint.json \
#  --min_tgt_length 5 --max_tgt_length 32 \
#  --bs 32 --beam_size 20 --length_penalty 0.2 --ngram_size 3 \
#  --prune_factor 50 --sat_tolerance 2 --beta 0 --early_stop 1.5 \
#  --save_path "${OUTPUT_DIR}/${OUTPUT_FILE}.${SPLIT}" --score_path "${OUTPUT_DIR}/${OUTPUT_FILE}.json"


### tags: 
# ['punct', 'case', 'det', 'nsubj', 'root', 'amod', 'advmod', 'obj', 'obl', 'nmod', 'conj', 'mark','compound', \\
#   'cc', 'aux', 'cop', 'nmod:poss', 'advcl', 'xcomp', 'nummod', 'acl:relcl', 'ccomp', 'flat', 'acl', 'aux:pass',\\
#   'appos', 'parataxis', 'nsubj:pass', 'compound:prt', 'discourse', 'fixed', 'expl', 'obl:tmod', 'dep', \\
#   'obl:npmod', 'nmod:tmod', 'iobj', 'csubj', 'list', 'det:predet', 'obl:agent', 'nmod:npmod', 'reparandum', \\
#   'vocative', 'cc:preconj', 'goeswith', 'orphan', 'dislocated', 'csubj:pass', 'flat:foreign']