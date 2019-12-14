#!/usr/bin/env bash
model_name='themis_word_char'
for seed in {12,22,32,42,52}
do
  echo "Running model ${model_name}"
  echo "Seed: ${seed}"
  CUDA_VISIBLE_DEVICES=0,1 python train_themis_word_char.py --seed ${seed}
done