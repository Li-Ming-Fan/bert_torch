#!/usr/bin/env bash

python3.6 trainer.py \
  --task_name=sim \
  --data_dir=../data_sim \
  --output_dir=../model_sim \
  --gpu_id=0 \
  --adaptive_decay=0 \
  --with_multibatch=0 \
  --do_train=0 \
  --do_eval=1 \
  --checkpoint_steps=1000 \
  --max_seq_len=196 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_epochs=10
