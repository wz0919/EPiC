trap 'kill 0' SIGINT  

target_pose="0 30 -0.6 0 0"
target_pose_str="0_30_-0.6_0_0" 

traj_name="loop1"
traj_txt="test/trajs/${traj_name}.txt"

video="../../data/test_v2v/videos/amalfi-coast_traj_loop2.mp4"

processed_data_name=$1
# filename=$(basename "$video" .mp4)
filename="amalfi-coast"
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --video_path "$video" \
    --stride 1 \
    --out_dir experiments \
    --radius_scale 1 \
    --camera 'traj' \
    --mask \
    --target_pose $target_pose \
    --traj_txt "$traj_txt" \
    --save_name "${filename}_traj_${traj_name}" \
    --mode "gradual" \
    --out_dir ../../data/${processed_data_name}

CUDA_VISIBLE_DEVICES=0 python inference.py \
    --video_path "$video" \
    --stride 1 \
    --out_dir experiments \
    --radius_scale 1 \
    --camera 'target' \
    --mask \
    --target_pose $target_pose \
    --traj_txt "$traj_txt" \
    --save_name "${filename}_target_${target_pose_str}" \
    --mode "gradual" \
    --out_dir ../../data/${processed_data_name}
