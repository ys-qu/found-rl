#!/bin/bash

NUM_GPUS=1
TRAIN_DATA="/home/user/Downloads/roach_data_shards_semantic_v2/carla-{000000..000274}.tar"
TRAIN_NUM_SAMPLES=1400000

MODEL="ViT-B-16"
PRETRAINED="laion2b_s34b_b88k"

LOG_PATH="./logs/roach_clip_finetune_laion2b_s34b_b88k_semantic_v2"


BATCH_SIZE_PER_GPU=1024
WORKERS=16

export PYTHONPATH="$PYTHONPATH:$(pwd)/open_clip/src"

echo "Starting training..."
echo "Using Precision: BF16 | Batch: $BATCH_SIZE_PER_GPU | Image Size: 192"

torchrun --nproc_per_node=$NUM_GPUS --master_port=12345 -m open_clip_train.main -- \
    --train-data="$TRAIN_DATA" \
    --train-num-samples=$TRAIN_NUM_SAMPLES \
    --dataset-type="webdataset" \
    --batch-size=$BATCH_SIZE_PER_GPU \
    --workers=$WORKERS \
    --model="$MODEL" \
    --pretrained="$PRETRAINED" \
    --epochs=10 \
    --lr=1e-5 \
    --beta1=0.9 \
    --beta2=0.98 \
    --wd=0.2 \
    --eps=1e-6 \
    --precision="amp_bfloat16" \
    --grad-clip-norm=1.0 \
    --warmup=2000 \
    --save-frequency=1 \
    --logs="$LOG_PATH" \
    --report-to="tensorboard" \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --force-image-size 192 \
    --grad-checkpointing