#!/usr/bin/env bash
model_name='themis_word'
for seed in {12,22,32,42,52}
do
  echo "Running model ${model_name}"
  echo "Seed: ${seed}"
  CUDA_VISIBLE_DEVICES=0 python train_sentiment_analysis.py --seed ${seed}
done