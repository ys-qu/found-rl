export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=offline

# Switch to the parent directory of the script's parent directory
#cd "$(dirname "$(dirname "$0")")/.."

# Print current working directory
echo "Current working directory: $(pwd)"


python bc_stage/train.py --model_path out/rwkv0b1-v0700_pretrain/VisualRWKV-v0700-0B1-v1.0-20250109.pth\
    --wandb "rwkv0b1-v0700_mix665k_0B1" --proj_dir out/rwkv0b1-v0700_mix665k_0B1 \
    --data_file /home/ai/qys/datasets/roach_bc_processed/carla_roach.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 2048 --epoch_steps 1000 --epoch_count 200 --epoch_begin 0 --epoch_save 20 \
    --micro_bsz 4 --accumulate_grad_batches 1 --n_layer 12 --n_embd 768 --pre_ffn 0 \
    --lr_init 6e-5 --lr_final 1.5e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 2 --precision fp16 --strategy deepspeed_stage_1 --grad_cp 1 \
    --image_folder /home/ai/qys/datasets/roach_bc_processed/ \
    --vision_tower_dir /home/ai/qys/projects/RWKV_RL/huggingface_models/ \
    --freeze_rwkv 0 --freeze_proj 0  --num_token_per_image 1024 --proj_type mlp


#python bc_stage/train.py --model_path out/rwkv0b1-v0700_pretrain/VisualRWKV-v0700-1B5-v1.0-20250204.pth\
#    --wandb "rwkv0b1-v0700_mix665k_1B5" --proj_dir out/rwkv0b1-v0700_mix665k_1B5 \
#    --data_file /home/ai/qys/datasets/roach_bc_processed/carla_roach.json \
#    --data_type "json" --vocab_size 65536 \
#    --ctx_len 2048 --epoch_steps 1000 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
#    --micro_bsz 2 --accumulate_grad_batches 1 --n_layer 12 --n_embd 2048 --pre_ffn 0 \
#    --lr_init 6e-5 --lr_final 1.5e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
#    --accelerator gpu --devices 2 --precision fp16 --strategy deepspeed_stage_1 --grad_cp 1 \
#    --image_folder /home/ai/qys/datasets/roach_bc_processed/ \
#    --vision_tower_dir /home/ai/qys/projects/RWKV_RL/huggingface_models/ \
#    --freeze_rwkv 0 --freeze_proj 0  --num_token_per_image 1024 --proj_type mlp