import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import imageio
import argparse
import sys

sys.path.append("RAFT/core")
from raft import RAFT
from utils.utils import InputPadder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_raft_model(ckpt_path):
    args = argparse.Namespace(
        small=False,
        mixed_precision=False,
        alternate_corr=False,
        dropout=0.0,
        max_depth=8,
        depth_network=False,
        depth_residual=False,
        depth_scale=1.0
    )
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model.module.to(DEVICE).eval()

def run_masking(video_path, output_path, mask_path, raft):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, first = cap.read()
    if not ok:
        print(f"Failed to read first frame in {video_path}")
        return

    resize_to = (720, 480)
    first = cv2.resize(first, resize_to)
    H, W, _ = first.shape
    area_thresh = (H * W) // 6

    grid = np.stack(np.meshgrid(np.arange(W), np.arange(H)), -1).astype(np.float32)
    pos = grid.copy()
    vis = np.ones((H, W), dtype=bool)

    writer = imageio.get_writer(output_path, fps=int(fps))

    prev = first.copy()
    frames_since_corr = 0
    freeze_mask = False
    frozen_mask = None
    all_masks = []

    writer.append_data(first[:, :, ::-1])
    all_masks.append(np.ones((H, W), dtype=bool))

    def to_tensor(bgr):
        return transforms.ToTensor()(bgr).unsqueeze(0).to(DEVICE)

    def raft_flow(img1_bgr, img2_bgr):
        t1, t2 = to_tensor(img1_bgr), to_tensor(img2_bgr)
        padder = InputPadder(t1.shape)
        i1, i2 = padder.pad(t1, t2)
        with torch.no_grad():
            _, flow = raft(i1, i2, iters=20, test_mode=True)
        return padder.unpad(flow)[0].permute(1, 2, 0).cpu().numpy()

    for _ in range(1, n_frames):
        ok, cur = cap.read()
        if not ok:
            break
        cur = cv2.resize(cur, resize_to)

        if not freeze_mask:
            flow_fw = raft_flow(prev, cur)
            pos += flow_fw
            frames_since_corr += 1

            x_ok = (0 <= pos[..., 0]) & (pos[..., 0] < W)
            y_ok = (0 <= pos[..., 1]) & (pos[..., 1] < H)
            vis &= x_ok & y_ok

            m = np.zeros((H, W), np.uint8)

            ys, xs = np.where(vis)
            px = np.round(pos[ys, xs, 0]).astype(int)
            py = np.round(pos[ys, xs, 1]).astype(int)

            inb = (0 <= px) & (px < W) & (0 <= py) & (py < H)
            m[py[inb], px[inb]] = 1
            m = cv2.dilate(m, np.ones((2, 2), np.uint8))

            visible_ratio = m.sum() / (H * W)
            if visible_ratio < 0.3:
                flow_0t = raft_flow(first, cur)
                pos = grid + flow_0t

                vis = np.ones((H, W), dtype=bool)
                x_ok = (0 <= pos[..., 0]) & (pos[..., 0] < W)
                y_ok = (0 <= pos[..., 1]) & (pos[..., 1] < H)
                vis &= x_ok & y_ok

                m.fill(0)
                ys, xs = np.where(vis)
                px = np.round(pos[ys, xs, 0]).astype(int)
                py = np.round(pos[ys, xs, 1]).astype(int)
                inb = (0 <= px) & (px < W) & (0 <= py) & (py < H)
                m[py[inb], px[inb]] = 1
                m = cv2.dilate(m, np.ones((2, 2), np.uint8))

                if m.sum() < area_thresh:
                    freeze_mask = True
                    frozen_mask = m.copy()

                frames_since_corr = 0
        else:
            m = frozen_mask

        effective_mask = m.astype(bool)
        all_masks.append(effective_mask)

        out = cur.copy()
        out[~effective_mask] = 0
        writer.append_data(out[:, :, ::-1])

        prev = cur if not freeze_mask else prev

    writer.close()
    cap.release()

    all_masks_array = np.stack(all_masks, axis=0)
    np.savez_compressed(mask_path, mask=all_masks_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--raft_ckpt", type=str, required=True)
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)
    parser.add_argument("--gpu_id", type=int, required=True)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.mask_path, exist_ok=True)

    video_list = sorted([
        f for f in os.listdir(args.video_path)
        if f.endswith(".mp4")
    ])
    selected_videos = video_list[args.start_idx : args.end_idx]

    print(f"[GPU {args.gpu_id}] Processing {len(selected_videos)} videos: {args.start_idx} to {args.end_idx}")
    model = load_raft_model(args.raft_ckpt)

    for fname in tqdm(selected_videos, desc="Batch Processing"):
        input_path = os.path.join(args.video_path, fname)
        mask_path = os.path.join(args.mask_path, fname.replace(".mp4", ".npz"))
        output_path = os.path.join(args.output_path, fname)

        if os.path.exists(mask_path):
            try:
                np.load(mask_path)["mask"]
                continue
            except:
                print(f"⚠️ Mask corrupt or unreadable: {mask_path} - Regenerating")

        if os.path.exists(output_path):
            continue

        run_masking(input_path, output_path, mask_path, model)