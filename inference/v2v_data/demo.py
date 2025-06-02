import gc
import os
import torch
from models.infer import DepthCrafterDemo
import numpy as np
import torch
from PIL import Image
from models.utils import *

import torch
import torch.nn.functional as F

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def get_center_crop_resolution(original_resoultion, target_aspect_ratio=(2, 3)):
    target_h, target_w = target_aspect_ratio
    aspect_ratio = target_w / target_h

    original_h, original_w = original_resoultion
    crop_h = original_h
    crop_w = int(crop_h * aspect_ratio)
    if crop_w > original_w:
        crop_w = original_w
        crop_h = int(crop_w / aspect_ratio)

    resized_h = 576
    resized_w = 1024
    
    h_ratio = resized_h / original_h
    w_ratio = resized_w / original_w
    
    crop_h = int(crop_h * h_ratio)
    crop_w = int(crop_w * w_ratio)
    return crop_h, crop_w

def process_video_tensor(video, resolution=(480, 720)):
    video_resized = F.interpolate(video, size=(resolution[0], resolution[1]), mode='bilinear', align_corners=False)

    video_uint8 = (video_resized.clamp(0, 1) * 255).byte()

    return video_uint8

def process_mask_tensor(video, resolution=(480, 720)):
    video_resized = F.interpolate(video, size=(resolution[0], resolution[1]), mode='bilinear', align_corners=False)

    return (video_resized==1).bool()

def center_crop_to_ratio(tensor: torch.Tensor, resolution=(480, 720)):
    """
    Args:
        tensor: [T, C, H, W], float32 or uint8
    Returns:
        cropped: [T, C, H_crop, W_crop], where H_crop:W_crop = 2:3 (480:720 ratio)
    """
    T, C, H, W = tensor.shape
    h, w = resolution
    target_ratio = w / h

    crop_h = H
    crop_w = int(H * target_ratio)
    if crop_w > W:
        crop_w = W
        crop_h = int(W / target_ratio)

    top = (H - crop_h) // 2
    left = (W - crop_w) // 2

    return tensor[:, :, top:top + crop_h, left:left + crop_w]

import imageio
import numpy as np

def save_video_as_mp4(video_tensor, save_path, fps=24):
    """
    video_tensor: [T, 3, H, W], dtype=uint8, values in [0, 255]
    save_path: e.g., "output_video.mp4"
    """
    assert video_tensor.dtype == torch.uint8 and video_tensor.ndim == 4
    T, C, H, W = video_tensor.shape

    video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()
    print(video_np.shape)

    imageio.mimwrite(
        save_path,
        video_np,
        fps=fps,
    )


class GetAnchorVideos:
    def __init__(self, opts, gradio=False):
        self.funwarp = Warper(device=opts.device)
        self.depth_estimater = DepthCrafterDemo(
            unet_path=opts.unet_path,
            pre_train_path=opts.pre_train_path,
            cpu_offload=opts.cpu_offload,
            device=opts.device,
        )

        # default: Load the model on the available device(s)
        self.caption_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            opts.qwen_path, torch_dtype="auto", device_map="auto"
        )
        # default processer
        self.caption_processor = AutoProcessor.from_pretrained(opts.qwen_path)

        if gradio:
            self.opts = opts

    def infer_gradual(self, opts):
        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        vr = VideoReader(opts.video_path, ctx=cpu(0))
        frame_shape = vr[0].shape  # (H, W, 3)
        ori_resolution = frame_shape[:2]
        print(f"==> original video shape: {frame_shape}")
        target_resolution = get_center_crop_resolution(ori_resolution)
        print(f"==> target video shape resized: {target_resolution}")

        prompt = self.get_caption(opts, opts.video_path)
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == opts.video_length
        
        depths = center_crop_to_ratio(depths, resolution=target_resolution)
        frames = center_crop_to_ratio(frames, resolution=target_resolution)
        pose_s, pose_t, K = self.get_poses(opts, depths, num_frames=opts.video_length)
        warped_images = []
        masks = []
        for i in tqdm(range(opts.video_length)):
            warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                frames[i : i + 1],
                None,
                depths[i : i + 1],
                pose_s[i : i + 1],
                pose_t[i : i + 1],
                K[i : i + 1],
                None,
                opts.mask,
                twice=False,
            )
            warped_images.append(warped_frame2)
            masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        
        cond_video_save = process_video_tensor(cond_video).cpu()
        ori_video_save = process_video_tensor((frames+1.0) / 2.0).cpu()
        save_cated = torch.cat([ori_video_save, cond_video_save], dim=3)
        # post_t  captions  depth  intrinsics  joint_videos
        save_name = os.path.basename(opts.video_path).split('.')[0]
        save_name = opts.save_name

        os.makedirs(f'{opts.out_dir}', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/videos', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/masked_videos', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/depth', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/masks', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/post_t', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/pose_s', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/intrinsics', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/captions', exist_ok=True)
        
        mask_save = process_mask_tensor(torch.cat(masks)).squeeze().cpu().numpy()
        np.save(f"{opts.out_dir}/depth/{save_name}.npy",depths.cpu().numpy())
        np.savez_compressed(f"{opts.out_dir}/masks/{save_name}.npz",mask=mask_save)
        save_video_as_mp4(ori_video_save,f"{opts.out_dir}/videos/{save_name}.mp4", fps=8)
        save_video_as_mp4(cond_video_save,f"{opts.out_dir}/masked_videos/{save_name}.mp4", fps=8)
        np.save(f'{opts.out_dir}/post_t/' + save_name + '.npy',pose_t.cpu().numpy())
        np.save(f'{opts.out_dir}/pose_s/' + save_name + '.npy',pose_s.cpu().numpy())
        np.save(f'{opts.out_dir}/intrinsics/' + save_name + '.npy',K[0].cpu().numpy())
        # save prompt to txt
        with open(f'{opts.out_dir}/captions/' + save_name + '.txt', 'w') as f:
            f.write(prompt)

    def infer_image(self, opts):
        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        frames = frames[:1].repeat(opts.video_length, 0)
        if opts.video_path.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
            vr = VideoReader(opts.video_path, ctx=cpu(0))
            frame_shape = vr[0].shape  # (H, W, 3)
            ori_resolution = frame_shape[:2]
        elif opts.video_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img = Image.open(opts.video_path)
            ori_resolution = img.size[::-1]  # PIL gives (W, H), convert to (H, W)
        print(f"==> original video shape: {ori_resolution}")
        target_resolution = get_center_crop_resolution(ori_resolution)
        print(f"==> target video shape resized: {target_resolution}")
        # prompt = self.get_caption(opts, frames[opts.video_length // 2])
        prompt = self.get_caption(opts, opts.video_path)
        # depths= self.depth_estimater.infer(frames, opts.near, opts.far).to(opts.device)
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == opts.video_length
        
        # depths = mask_out_cropped_edges(depths)
        depths = center_crop_to_ratio(depths, resolution=target_resolution)
        frames = center_crop_to_ratio(frames, resolution=target_resolution)
        assert frames.shape[0] == opts.video_length
        pose_s, pose_t, K = self.get_poses(opts, depths, num_frames=opts.video_length)
        warped_images = []
        masks = []
        for i in tqdm(range(opts.video_length)):
            warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                frames[i : i + 1],
                None,
                depths[i : i + 1],
                pose_s[i : i + 1],
                pose_t[i : i + 1],
                K[i : i + 1],
                None,
                opts.mask,
                twice=False,
            )
            warped_images.append(warped_frame2)
            masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        
        cond_video_save = process_video_tensor(cond_video).cpu()
        ori_video_save = process_video_tensor((frames+1.0) / 2.0).cpu()
        save_cated = torch.cat([ori_video_save, cond_video_save], dim=3)
        # post_t  captions  depth  intrinsics  joint_videos
        save_name = os.path.basename(opts.video_path).split('.')[0]
        # save_name = f"{save_name}_"
        save_name = opts.save_name

        os.makedirs(f'{opts.out_dir}', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/videos', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/masked_videos', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/depth', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/masks', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/post_t', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/pose_s', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/intrinsics', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/captions', exist_ok=True)
        
        mask_save = process_mask_tensor(torch.cat(masks)).squeeze().cpu().numpy()
        np.save(f"{opts.out_dir}/depth/{save_name}.npy",depths.cpu().numpy())
        np.savez_compressed(f"{opts.out_dir}/masks/{save_name}.npz",mask=mask_save)
        save_video_as_mp4(ori_video_save,f"{opts.out_dir}/videos/{save_name}.mp4", fps=8)
        save_video_as_mp4(cond_video_save,f"{opts.out_dir}/masked_videos/{save_name}.mp4", fps=8)
        np.save(f'{opts.out_dir}/post_t/' + save_name + '.npy',pose_t.cpu().numpy())
        np.save(f'{opts.out_dir}/pose_s/' + save_name + '.npy',pose_s.cpu().numpy())
        np.save(f'{opts.out_dir}/intrinsics/' + save_name + '.npy',K[0].cpu().numpy())
        # save prompt to txt
        with open(f'{opts.out_dir}/captions/' + save_name + '.txt', 'w') as f:
            f.write(prompt)

        
    def infer_direct(self, opts):
        opts.cut = 20
        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        vr = VideoReader(opts.video_path, ctx=cpu(0))
        frame_shape = vr[0].shape  # (H, W, 3)
        ori_resolution = frame_shape[:2]
        print(f"==> original video shape: {frame_shape}")
        target_resolution = get_center_crop_resolution(ori_resolution)
        print(f"==> target video shape resized: {target_resolution}")

        prompt = self.get_caption(opts, opts.video_path)
        
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == opts.video_length
        depths = center_crop_to_ratio(depths, resolution=target_resolution)
        frames = center_crop_to_ratio(frames, resolution=target_resolution)
        
        pose_s, pose_t, K = self.get_poses(opts, depths, num_frames=opts.cut)

        warped_images = []
        masks = []
        for i in tqdm(range(opts.video_length)):
            if i < opts.cut:
                warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                    frames[0:1],
                    None,
                    depths[0:1],
                    pose_s[0:1],
                    pose_t[i : i + 1],
                    K[0:1],
                    None,
                    opts.mask,
                    twice=False,
                )
                warped_images.append(warped_frame2)
                masks.append(mask2)
            else:
                warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                    frames[i - opts.cut : i - opts.cut + 1],
                    None,
                    depths[i - opts.cut : i - opts.cut + 1],
                    pose_s[0:1],
                    pose_t[-1:],
                    K[0:1],
                    None,
                    opts.mask,
                    twice=False,
                )
                warped_images.append(warped_frame2)
                masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_video_save = process_video_tensor(cond_video).cpu()
        ori_video_save = process_video_tensor((frames+1.0) / 2.0).cpu()
        save_cated = torch.cat([ori_video_save, cond_video_save], dim=3)
        # post_t  captions  depth  intrinsics  joint_videos
        save_name = os.path.basename(opts.video_path).split('.')[0]
        save_name = opts.save_name

        os.makedirs(f'{opts.out_dir}', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/videos', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/masked_videos', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/depth', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/masks', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/post_t', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/pose_s', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/intrinsics', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/captions', exist_ok=True)
        
        mask_save = process_mask_tensor(torch.cat(masks)).squeeze().cpu().numpy()
        np.save(f"{opts.out_dir}/depth/{save_name}.npy",depths.cpu().numpy())
        np.savez_compressed(f"{opts.out_dir}/masks/{save_name}.npz",mask=mask_save)
        save_video_as_mp4(ori_video_save,f"{opts.out_dir}/videos/{save_name}.mp4", fps=8)
        save_video_as_mp4(cond_video_save,f"{opts.out_dir}/masked_videos/{save_name}.mp4", fps=8)
        np.save(f'{opts.out_dir}/post_t/' + save_name + '.npy',pose_t.cpu().numpy())
        np.save(f'{opts.out_dir}/pose_s/' + save_name + '.npy',pose_s.cpu().numpy())
        np.save(f'{opts.out_dir}/intrinsics/' + save_name + '.npy',K[0].cpu().numpy())
        # save prompt to txt
        with open(f'{opts.out_dir}/captions/' + save_name + '.txt', 'w') as f:
            f.write(prompt)


    def infer_bullet(self, opts):
        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        vr = VideoReader(opts.video_path, ctx=cpu(0))
        frame_shape = vr[0].shape  # (H, W, 3)
        ori_resolution = frame_shape[:2]
        print(f"==> original video shape: {frame_shape}")
        target_resolution = get_center_crop_resolution(ori_resolution)
        print(f"==> target video shape resized: {target_resolution}")

        prompt = self.get_caption(opts, opts.video_path)

        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)

        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == opts.video_length
        
        depths = center_crop_to_ratio(depths, resolution=target_resolution)
        frames = center_crop_to_ratio(frames, resolution=target_resolution)
        
        pose_s, pose_t, K = self.get_poses(opts, depths, num_frames=opts.video_length)

        warped_images = []
        masks = []
        for i in tqdm(range(opts.video_length)):
            warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                frames[-1:],
                None,
                depths[-1:],
                pose_s[0:1],
                pose_t[i : i + 1],
                K[0:1],
                None,
                opts.mask,
                twice=False,
            )
            warped_images.append(warped_frame2)
            masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        
        cond_video_save = process_video_tensor(cond_video).cpu()
        ori_video_save = process_video_tensor((frames+1.0) / 2.0).cpu()
        save_cated = torch.cat([ori_video_save, cond_video_save], dim=3)
        # post_t  captions  depth  intrinsics  joint_videos
        save_name = os.path.basename(opts.video_path).split('.')[0]
        save_name = opts.save_name

        os.makedirs(f'{opts.out_dir}', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/videos', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/masked_videos', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/depth', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/masks', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/post_t', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/pose_s', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/intrinsics', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/captions', exist_ok=True)
        
        mask_save = process_mask_tensor(torch.cat(masks)).squeeze().cpu().numpy()
        np.save(f"{opts.out_dir}/depth/{save_name}.npy",depths.cpu().numpy())
        np.savez_compressed(f"{opts.out_dir}/masks/{save_name}.npz",mask=mask_save)
        save_video_as_mp4(ori_video_save,f"{opts.out_dir}/videos/{save_name}.mp4", fps=8)
        save_video_as_mp4(cond_video_save,f"{opts.out_dir}/masked_videos/{save_name}.mp4", fps=8)
        np.save(f'{opts.out_dir}/post_t/' + save_name + '.npy',pose_t.cpu().numpy())
        np.save(f'{opts.out_dir}/pose_s/' + save_name + '.npy',pose_s.cpu().numpy())
        np.save(f'{opts.out_dir}/intrinsics/' + save_name + '.npy',K[0].cpu().numpy())
        # save prompt to txt
        with open(f'{opts.out_dir}/captions/' + save_name + '.txt', 'w') as f:
            f.write(prompt)

    def infer_zoom(self, opts):
        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        vr = VideoReader(opts.video_path, ctx=cpu(0))
        frame_shape = vr[0].shape  # (H, W, 3)
        ori_resolution = frame_shape[:2]
        print(f"==> original video shape: {frame_shape}")
        target_resolution = get_center_crop_resolution(ori_resolution)
        print(f"==> target video shape resized: {target_resolution}")

        prompt = self.get_caption(opts, opts.video_path)
        
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == opts.video_length

        depths = center_crop_to_ratio(depths, resolution=target_resolution)
        frames = center_crop_to_ratio(frames, resolution=target_resolution)
        
        pose_s, pose_t, K = self.get_poses_f(opts, depths, num_frames=opts.video_length, f_new=250)

        warped_images = []
        masks = []
        for i in tqdm(range(opts.video_length)):
            warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                frames[i : i + 1],
                None,
                depths[i : i + 1],
                pose_s[i : i + 1],
                pose_t[i : i + 1],
                K[0 : 1],
                K[i : i + 1],
                opts.mask,
                twice=False,
            )
            warped_images.append(warped_frame2)
            masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        
        cond_video_save = process_video_tensor(cond_video).cpu()
        ori_video_save = process_video_tensor((frames+1.0) / 2.0).cpu()
        save_cated = torch.cat([ori_video_save, cond_video_save], dim=3)
        # post_t  captions  depth  intrinsics  joint_videos
        save_name = os.path.basename(opts.video_path).split('.')[0]
        save_name = opts.save_name

        os.makedirs(f'{opts.out_dir}', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/videos', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/masked_videos', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/depth', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/masks', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/post_t', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/pose_s', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/intrinsics', exist_ok=True)
        os.makedirs(f'{opts.out_dir}/captions', exist_ok=True)
        
        mask_save = process_mask_tensor(torch.cat(masks)).squeeze().cpu().numpy()
        np.save(f"{opts.out_dir}/depth/{save_name}.npy",depths.cpu().numpy())
        np.savez_compressed(f"{opts.out_dir}/masks/{save_name}.npz",mask=mask_save)
        save_video_as_mp4(ori_video_save,f"{opts.out_dir}/videos/{save_name}.mp4", fps=8)
        save_video_as_mp4(cond_video_save,f"{opts.out_dir}/masked_videos/{save_name}.mp4", fps=8)
        np.save(f'{opts.out_dir}/post_t/' + save_name + '.npy',pose_t.cpu().numpy())
        np.save(f'{opts.out_dir}/pose_s/' + save_name + '.npy',pose_s.cpu().numpy())
        np.save(f'{opts.out_dir}/intrinsics/' + save_name + '.npy',K[0].cpu().numpy())
        # save prompt to txt
        with open(f'{opts.out_dir}/captions/' + save_name + '.txt', 'w') as f:
            f.write(prompt)

    def get_caption(self, opts, video_path):
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": "Give me a detailed caption of this video. Directly discribe the content of the video. Don't start with \"in the video\" stuff."},
                ],
            }
        ]
        
        text = self.caption_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = self.caption_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            # fps=fps,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.caption_model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_text = self.caption_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return generated_text[0] + opts.refine_prompt

    def get_poses(self, opts, depths, num_frames):
        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * opts.radius_scale
        )
        radius = min(radius, 5)
        # cx = 512.0  # depths.shape[-1]//2
        # cy = 288.0  # depths.shape[-2]//2
        cx = depths.shape[-1]//2
        cy = depths.shape[-2]//2
        f = 500  # 500.
        K = (
            torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            .repeat(num_frames, 1, 1)
            .to(opts.device)
        )
        c2w_init = (
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            .to(opts.device)
            .unsqueeze(0)
        )
        if opts.camera == 'target':
            dtheta, dphi, dr, dx, dy = opts.target_pose
            poses = generate_traj_specified(
                c2w_init, dtheta, dphi, dr * radius, dx, dy, num_frames, opts.device
            )
        elif opts.camera == 'target_fast':
            dtheta, dphi, dr, dx, dy = opts.target_pose
            poses = generate_traj_specified_fast(
                c2w_init, dtheta, dphi, dr * radius, dx, dy, num_frames, opts.device
            )
        elif opts.camera == 'traj':
            with open(opts.traj_txt, 'r') as file:
                lines = file.readlines()
                theta = [float(i) for i in lines[0].split()]
                phi = [float(i) for i in lines[1].split()]
                r = [float(i) * radius for i in lines[2].split()]
            poses = generate_traj_txt(c2w_init, phi, theta, r, num_frames, opts.device)
        poses[:, 2, 3] = poses[:, 2, 3] + radius
        pose_s = poses[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)
        pose_t = poses
        return pose_s, pose_t, K

    def get_poses_f(self, opts, depths, num_frames, f_new):
        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * opts.radius_scale
        )
        radius = min(radius, 5)
        cx = depths.shape[-1]//2
        cy = depths.shape[-2]//2
        # cx = 512.0  
        # cy = 288.0  
        f = 500
        # f_new,d_r: 250,0.5; 1000,-0.9
        f_values = torch.linspace(f, f_new, num_frames, device=opts.device)
        K = torch.zeros((num_frames, 3, 3), device=opts.device)
        K[:, 0, 0] = f_values
        K[:, 1, 1] = f_values
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0
        c2w_init = (
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            .to(opts.device)
            .unsqueeze(0)
        )
        if opts.camera == 'target':
            dtheta, dphi, dr, dx, dy = opts.target_pose
            poses = generate_traj_specified(
                c2w_init, dtheta, dphi, dr * radius, dx, dy, num_frames, opts.device
            )
        elif opts.camera == 'target_fast':
            dtheta, dphi, dr, dx, dy = opts.target_pose
            poses = generate_traj_specified_fast(
                c2w_init, dtheta, dphi, dr * radius, dx, dy, num_frames, opts.device
            )
        elif opts.camera == 'traj':
            with open(opts.traj_txt, 'r') as file:
                lines = file.readlines()
                theta = [float(i) for i in lines[0].split()]
                phi = [float(i) for i in lines[1].split()]
                r = [float(i) * radius for i in lines[2].split()]
            poses = generate_traj_txt(c2w_init, phi, theta, r, num_frames, opts.device)
        poses[:, 2, 3] = poses[:, 2, 3] + radius
        pose_s = poses[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)
        pose_t = poses
        return pose_s, pose_t, K