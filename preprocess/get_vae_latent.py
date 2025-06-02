import os
import torch
import numpy as np
import math
import argparse
from decord import VideoReader
from diffusers import AutoencoderKLCogVideoX
from safetensors.torch import save_file
import tqdm
import random

def encode_video(video, vae):
    video = video[None].permute(0, 2, 1, 3, 4).contiguous()
    video = video.to(vae.device, dtype=vae.dtype)
    latent_dist = vae.encode(video).latent_dist
    latent = latent_dist.sample() * vae.config.scaling_factor
    return latent

def add_dashed_rays_to_video(video_tensor, num_perp_samples=50, density_decay=0.075):
    T, C, H, W = video_tensor.shape
    max_length = int((H**2 + W**2) ** 0.5) + 10
    center = torch.tensor([W / 2, H / 2])
    theta = torch.rand(1).item() * 2 * math.pi
    direction = torch.tensor([math.cos(theta), math.sin(theta)])
    direction = direction / direction.norm()
    d_perp = torch.tensor([-direction[1], direction[0]])
    half_len = max(H, W) // 2
    positions = torch.linspace(-half_len, half_len, num_perp_samples)
    perp_coords = center[None, :] + positions[:, None] * d_perp[None, :]
    x0, y0 = perp_coords[:, 0], perp_coords[:, 1]
    steps = []
    dist = 0
    while dist < max_length:
        steps.append(dist)
        dist += 1.0 + density_decay * dist
    steps = torch.tensor(steps)
    S = len(steps)
    dxdy = direction[None, :] * steps[:, None]
    all_xy = perp_coords[:, None, :] + dxdy[None, :, :]
    all_xy = all_xy.reshape(-1, 2)
    all_x = all_xy[:, 0].round().long()
    all_y = all_xy[:, 1].round().long()
    valid = (0 <= all_x) & (all_x < W) & (0 <= all_y) & (all_y < H)
    all_x = all_x[valid]
    all_y = all_y[valid]
    x0r = x0.round().long().clamp(0, W - 1)
    y0r = y0.round().long().clamp(0, H - 1)
    frame0 = video_tensor[0]
    base_colors = frame0[:, y0r, x0r]
    base_colors = base_colors.repeat_interleave(S, dim=1)[:, valid]
    video_out = video_tensor.clone()
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for dxo, dyo in offsets:
        ox = all_x + dxo
        oy = all_y + dyo
        inside = (0 <= ox) & (ox < W) & (0 <= oy) & (oy < H)
        ox = ox[inside]
        oy = oy[inside]
        colors = base_colors[:, inside]
        for c in range(C):
            video_out[1:, c, oy, ox] = colors[c][None, :].expand(T - 1, -1)
    return video_out

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKLCogVideoX.from_pretrained(args.pretrained_model_path, subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device, dtype=torch.float16)

    masked_video_path = os.path.join(args.video_root, "masked_videos")
    source_video_path = os.path.join(args.video_root, "videos")
    joint_latent_path = os.path.join(args.video_root, "joint_latents")
    os.makedirs(joint_latent_path, exist_ok=True)

    all_video_names = sorted(os.listdir(source_video_path))
    video_names = all_video_names[args.start_idx : args.end_idx]

    for video_name in tqdm.tqdm(video_names, desc=f"GPU {args.gpu_id}"):
        masked_video_file = os.path.join(masked_video_path, video_name)
        source_video_file = os.path.join(source_video_path, video_name)
        output_file = os.path.join(joint_latent_path, video_name.replace('.mp4', '.safetensors'))

        if not os.path.exists(masked_video_file):
            print(f"Skipping {video_name}, masked video not found.")
            continue
        if os.path.exists(output_file):
            continue

        try:
            vr = VideoReader(source_video_file)
            video = torch.from_numpy(vr.get_batch(np.arange(49)).asnumpy()).permute(0, 3, 1, 2).contiguous()
            video = (video / 255.0) * 2 - 1
            source_latent = encode_video(video, vae)

            vr = VideoReader(masked_video_file)
            video = torch.from_numpy(vr.get_batch(np.arange(49)).asnumpy()).permute(0, 3, 1, 2).contiguous()
            video = (video / 255.0) * 2 - 1
            video = add_dashed_rays_to_video(video)
            masked_latent = encode_video(video, vae)

            source_latent = source_latent.to("cpu")
            masked_latent = masked_latent.to("cpu")
            cated_latent = torch.cat([source_latent, masked_latent], dim=2)
            save_file({'joint_latents': cated_latent}, output_file)

        except Exception as e:
            print(f"[GPU {args.gpu_id}] Error processing {video_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)
    parser.add_argument("--gpu_id", type=int, required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    main(args)
