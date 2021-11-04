#!/bin/bash

CHECKPOINT_PATH=$1
MPSIZE=1
NLAYERS=32
NHIDDEN=2560
NATT=32
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/ds_config.json"

python3 -m torch.distributed.launch --nproc_per_node=$MPSIZE --master_port $MASTER_PORT generate_samples.py \
       --model-parallel-size $MPSIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --load $CHECKPOINT_PATH \
       --num-attention-heads $NATT \
       --max-position-embeddings 1024 \
       --tokenizer-type ChineseSPTokenizer \
       --fp16 \
       --cache-dir cache \
       --seq-length 1024 \
       --mem-length 512 \
       --transformer-xl \
       --seed 16 \

# nohup bash scripts/generate_text.sh ./checkpoints/3200/ > ./services.log &