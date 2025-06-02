MODEL_PATH="pretrained/CogVideoX-5b-I2V"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

video_root_dir="data/train" # subfolders: annotations/ pose_files/ video_clips/

dir=`pwd`
output_dir=${dir}/out/EPiC
MODEL_PATH=${dir}/${MODEL_PATH}
video_root_dir=${dir}/${video_root_dir}
cd training

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --config_file accelerate_config_machine.yaml --multi_gpu --main_process_port 29502 \
  train_controlnet_i2v_pcd_render_mask_aware_add_dash_use_latent.py \
  --tracker_name "cogvideox-controlnet" \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --enable_tiling \
  --enable_slicing \
  --num_inference_steps 28 \
  --seed 42 \
  --mixed_precision bf16 \
  --output_dir $output_dir \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --video_root_dir $video_root_dir \
  --hflip_p 0.0 \
  --controlnet_transformer_num_layers 8 \
  --controlnet_input_channels 3 \
  --downscale_coef 8 \
  --controlnet_weights 1.0 \
  --train_batch_size 2 \
  --dataloader_num_workers 0 \
  --num_train_epochs 200 \
  --checkpointing_steps 500 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 250 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --enable_time_sampling \
  --time_sampling_type truncated_normal \
  --time_sampling_mean 0.95 \
  --time_sampling_std 0.1 \
  --controlnet_guidance_start 0.0 \
  --controlnet_guidance_end 1.0 \
  --controlnet_transformer_num_attn_heads 4 \
  --controlnet_transformer_attention_head_dim 64 \
  --controlnet_transformer_out_proj_dim_factor 64 \
  --controlnet_transformer_out_proj_dim_zero_init \
  --text_embedding_path "${video_root_dir}/caption_embs" 