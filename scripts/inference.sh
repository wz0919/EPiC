#!/bin/bash
trap 'kill 0' SIGINT

export MODEL_PATH="./pretrained/CogVideoX-5b-I2V"

processed_data_name=$1
ckpt_steps=500
ckpt_dir=./out/EPiC_pretrained
ckpt_file=checkpoint-${ckpt_steps}.pt
ckpt_path=${ckpt_dir}/${ckpt_file}
video_root_dir="./data/${processed_data_name}"
out_dir=${ckpt_dir}/test/${ckpt_steps}_${processed_data_name}

CUDA_VISIBLE_DEVICES=0 python inference/cli_demo_camera_i2v_pcd.py \
    --video_root_dir $video_root_dir \
    --base_model_path $MODEL_PATH \
    --controlnet_model_path $ckpt_path \
    --output_path "${out_dir}" \
    --start_camera_idx 0 \
    --end_camera_idx 8 \
    --controlnet_weights 1.0 \
    --controlnet_guidance_start 0.0 \
    --controlnet_guidance_end 0.4 \
    --controlnet_input_channels 3 \
    --controlnet_transformer_num_attn_heads 4 \
    --controlnet_transformer_attention_head_dim 64 \
    --controlnet_transformer_out_proj_dim_factor 64 \
    --controlnet_transformer_out_proj_dim_zero_init \
    --vae_channels 16 \
    --num_frames 49 \
    --controlnet_transformer_num_layers 8 \
    --infer_with_mask \
    --pool_style 'max' \
    --seed 1
