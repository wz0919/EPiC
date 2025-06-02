#!/bin/bash
trap 'kill 0' SIGINT

mode=$1  # Options: caption, masking, latent

video_root="../data/train"
pretrained_model_path="../pretrained/CogVideoX-5b-I2V"
raft_ckpt="../pretrained/RAFT/raft-things.pth"
# gpu_list="0,1,2,3,4,5,6,7"
gpu_list="2,3,6,7"

gpus=(${gpu_list//,/ })
num_gpus=${#gpus[@]}

if [[ "$mode" == "caption" ]]; then
    echo "==== Running CAPTION EMBEDDING ===="

    caption_path="$video_root/captions"
    caption_emb_path="$video_root/caption_embs"
    all_files=($caption_path/*.txt)
    total=${#all_files[@]}
    chunk_size=$(( (total + num_gpus - 1) / num_gpus ))

    echo "Total caption files: $total"
    echo "Using $num_gpus GPUs, chunk size: $chunk_size"

    for ((i=0; i<num_gpus; i++)); do
        start_idx=$((i * chunk_size))
        end_idx=$(( (i + 1) * chunk_size ))
        (( end_idx > total )) && end_idx=$total

        gpu_id=${gpus[$i]}
        echo "Launching GPU $gpu_id: captions $start_idx to $end_idx"

        CUDA_VISIBLE_DEVICES=$gpu_id python get_prompt_emb.py \
            --pretrained_model_name_or_path $pretrained_model_path \
            --caption_path $caption_path \
            --output_path $caption_emb_path \
            --gpu_id $gpu_id \
            --start_idx $start_idx \
            --end_idx $end_idx &
    done

elif [[ "$mode" == "masking" ]]; then
    echo "==== Running VIDEO MASKING ===="

    source_video_dir="$video_root/videos"
    mask_dir="$video_root/masks"
    masked_video_dir="$video_root/masked_videos"
    all_videos=($source_video_dir/*.mp4)
    total=${#all_videos[@]}
    chunk_size=$(( (total + num_gpus - 1) / num_gpus ))

    echo "Total videos: $total"
    echo "Using $num_gpus GPUs, chunk size: $chunk_size"

    for ((i=0; i<num_gpus; i++)); do
        start_idx=$((i * chunk_size))
        end_idx=$(( (i + 1) * chunk_size ))
        (( end_idx > total )) && end_idx=$total

        gpu_id=${gpus[$i]}
        echo "Launching GPU $gpu_id: videos $start_idx to $end_idx"

        CUDA_VISIBLE_DEVICES=$gpu_id python get_masked_videos.py \
            --video_path $source_video_dir \
            --output_path $masked_video_dir \
            --mask_path $mask_dir \
            --raft_ckpt $raft_ckpt \
            --start_idx $start_idx \
            --end_idx $end_idx \
            --gpu_id $gpu_id &
    done

elif [[ "$mode" == "latent" ]]; then
    echo "==== Running LATENT ENCODING ===="

    all_videos=($video_root/videos/*.mp4)
    total=${#all_videos[@]}
    chunk_size=$(( (total + num_gpus - 1) / num_gpus ))

    echo "Total videos: $total"
    echo "Using $num_gpus GPUs, chunk size: $chunk_size"

    for ((i=0; i<num_gpus; i++)); do
        start_idx=$((i * chunk_size))
        end_idx=$(( (i + 1) * chunk_size ))
        (( end_idx > total )) && end_idx=$total

        gpu_id=${gpus[$i]}
        echo "Launching GPU $gpu_id: videos $start_idx to $end_idx"

        CUDA_VISIBLE_DEVICES=$gpu_id python get_vae_latent.py \
            --video_root $video_root \
            --pretrained_model_path $pretrained_model_path \
            --start_idx $start_idx \
            --end_idx $end_idx \
            --gpu_id $gpu_id &
    done

else
    echo "Unknown mode: $mode"
    echo "Usage: bash preprocess.sh [caption|masking|latent]"
    exit 1
fi

wait
echo "All processes completed."