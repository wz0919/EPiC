import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.append('.')
sys.path.append('..')
import argparse
import os

import torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import (
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    AutoencoderKLCogVideoX
)
from diffusers.utils import export_to_video, load_video

from controlnet_pipeline import ControlnetCogVideoXImageToVideoPCDPipeline
from cogvideo_transformer import CustomCogVideoXTransformer3DModel
from cogvideo_controlnet_pcd import CogVideoXControlnetPCD
from training.controlnet_datasets_camera_pcd_mask import RealEstate10KPCDRenderDataset
from torchvision.transforms.functional import to_pil_image

from inference.utils import stack_images_horizontally
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import cv2

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import cv2
import numpy as np
import torch

def get_black_region_mask_tensor(video_tensor, threshold=2, kernel_size=15):
    """
    Generate cleaned binary masks for black regions in a video tensor.
    
    Args:
        video_tensor (torch.Tensor): shape (T, H, W, 3), RGB, uint8
        threshold (int): pixel intensity threshold to consider a pixel as black (default: 20)
        kernel_size (int): morphological kernel size to smooth masks (default: 7)
    
    Returns:
        torch.Tensor: binary mask tensor of shape (T, H, W), where 1 indicates black region
    """
    video_uint8 = ((video_tensor + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)  # shape (T, H, W, C)
    video_np = video_uint8.numpy()

    T, H, W, _ = video_np.shape
    masks = np.empty((T, H, W), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for t in range(T):
        img = video_np[t]  # (H, W, 3), uint8
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        masks[t] = (mask_cleaned > 0).astype(np.uint8)
    return torch.from_numpy(masks)

def maxpool_mask_tensor(mask_tensor):
    """
    Apply spatial and temporal max pooling to a binary mask tensor.
    
    Args:
        mask_tensor (torch.Tensor): shape (T, H, W), binary mask (0 or 1)
    
    Returns:
        torch.Tensor: shape (12, 30, 45), pooled binary mask
    """
    T, H, W = mask_tensor.shape
    assert T % 12 == 0, "T must be divisible by 12 (e.g., 48)"
    assert H % 30 == 0 and W % 45 == 0, "H and W must be divisible by 30 and 45"

    # Reshape to (B=T, C=1, H, W) for 2D spatial pooling
    x = mask_tensor.unsqueeze(1).float()  # (T, 1, H, W)
    x_pooled = F.max_pool2d(x, kernel_size=(H // 30, W // 45))  # → (T, 1, 30, 45)

    # Temporal pooling: reshape to (12, T//12, 30, 45) and max along dim=1
    t_groups = T // 12
    x_pooled = x_pooled.view(12, t_groups, 30, 45)
    pooled_mask = torch.amax(x_pooled, dim=1)  # → (12, 30, 45)

    # Add a zero frame at the beginning: shape (1, 30, 45)
    zero_frame = torch.zeros_like(pooled_mask[0:1])  # (1, 30, 45)
    pooled_mask = torch.cat([zero_frame, pooled_mask], dim=0)  # → (13, 30, 45)
    
    return 1 - pooled_mask.int()

def avgpool_mask_tensor(mask_tensor):
    """
    Apply spatial and temporal average pooling to a binary mask tensor,
    and threshold at 0.5 to retain only majority-active regions.
    
    Args:
        mask_tensor (torch.Tensor): shape (T, H, W), binary mask (0 or 1)
    
    Returns:
        torch.Tensor: shape (13, 30, 45), pooled binary mask with first frame zeroed
    """
    T, H, W = mask_tensor.shape
    assert T % 12 == 0, "T must be divisible by 12 (e.g., 48)"
    assert H % 30 == 0 and W % 45 == 0, "H and W must be divisible by 30 and 45"

    # Spatial average pooling
    x = mask_tensor.unsqueeze(1).float()  # (T, 1, H, W)
    x_pooled = F.avg_pool2d(x, kernel_size=(H // 30, W // 45))  # → (T, 1, 30, 45)

    # Temporal pooling
    t_groups = T // 12
    x_pooled = x_pooled.view(12, t_groups, 30, 45)
    pooled_avg = torch.mean(x_pooled, dim=1)  # → (12, 30, 45)

    # Threshold: keep only when > 0.5
    pooled_mask = (pooled_avg > 0.5).int()

    # Add zero frame
    zero_frame = torch.zeros_like(pooled_mask[0:1])
    pooled_mask = torch.cat([zero_frame, pooled_mask], dim=0)  # → (13, 30, 45)

    return 1 - pooled_mask  # inverting as before

@torch.no_grad()
def generate_video(
    prompt,
    image,
    video_root_dir: str,
    base_model_path: str,
    use_zero_conv: bool,
    controlnet_model_path: str,
    controlnet_weights: float = 1.0,
    controlnet_guidance_start: float = 0.0,
    controlnet_guidance_end: float = 1.0,
    use_dynamic_cfg: bool = True,
    lora_path: str = None,
    lora_rank: int = 128,
    output_path: str = "./output/",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    num_frames: int = 49,
    height: int = 480,
    width: int = 720,
    start_camera_idx: int = 0,
    end_camera_idx: int = 1,
    controlnet_transformer_num_attn_heads: int = None,
    controlnet_transformer_attention_head_dim: int = None,
    controlnet_transformer_out_proj_dim_factor: int = None,
    controlnet_transformer_out_proj_dim_zero_init: bool = False,
    controlnet_transformer_num_layers: int = 8,
    downscale_coef: int = 8,
    controlnet_input_channels: int = 6,
    infer_with_mask: bool = False,
    pool_style: str = 'avg',
    pipe_cpu_offload: bool = False,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - video_root_dir (str): The path to the camera dataset
    - annotation_json (str): Name of subset (train.json or test.json)
    - base_model_path (str): The path of the pre-trained model to be used.
    - controlnet_model_path (str): The path of the pre-trained conrolnet model to be used.
    - controlnet_weights (float): Strenght of controlnet
    - controlnet_guidance_start (float): The stage when the controlnet starts to be applied
    - controlnet_guidance_end (float): The stage when the controlnet end to be applied
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - seed (int): The seed for reproducibility.
    """
    os.makedirs(output_path, exist_ok=True)
    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    tokenizer = T5Tokenizer.from_pretrained(
        base_model_path, subfolder="tokenizer"
    )
    text_encoder = T5EncoderModel.from_pretrained(
        base_model_path, subfolder="text_encoder"
    )
    transformer = CustomCogVideoXTransformer3DModel.from_pretrained(
        base_model_path, subfolder="transformer"
    )
    vae = AutoencoderKLCogVideoX.from_pretrained(
        base_model_path, subfolder="vae"
    )
    scheduler = CogVideoXDDIMScheduler.from_pretrained(
        base_model_path, subfolder="scheduler"
    )
    # ControlNet
    num_attention_heads_orig = 48 if "5b" in base_model_path.lower() else 30
    controlnet_kwargs = {}
    if controlnet_transformer_num_attn_heads is not None:
        controlnet_kwargs["num_attention_heads"] = args.controlnet_transformer_num_attn_heads
    else:
        controlnet_kwargs["num_attention_heads"] = num_attention_heads_orig
    if controlnet_transformer_attention_head_dim is not None:
        controlnet_kwargs["attention_head_dim"] = controlnet_transformer_attention_head_dim
    if controlnet_transformer_out_proj_dim_factor is not None:
        controlnet_kwargs["out_proj_dim"] = num_attention_heads_orig * controlnet_transformer_out_proj_dim_factor
    controlnet_kwargs["out_proj_dim_zero_init"] = controlnet_transformer_out_proj_dim_zero_init
    controlnet = CogVideoXControlnetPCD(
        num_layers=controlnet_transformer_num_layers,
        downscale_coef=downscale_coef,
        in_channels=controlnet_input_channels,
        use_zero_conv=use_zero_conv,
        **controlnet_kwargs,   
    )
    if controlnet_model_path:
        ckpt = torch.load(controlnet_model_path, map_location='cpu', weights_only=False)
        controlnet_state_dict = {}
        for name, params in ckpt['state_dict'].items():
            controlnet_state_dict[name] = params
        m, u = controlnet.load_state_dict(controlnet_state_dict, strict=False)
        print(f'[ Weights from pretrained controlnet was loaded into controlnet ] [M: {len(m)} | U: {len(u)}]')
    
    # Full pipeline
    pipe = ControlnetCogVideoXImageToVideoPCDPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        controlnet=controlnet,
        scheduler=scheduler,
    ).to('cuda')
    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(lora_scale=1 / lora_rank)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")

    # pipe.to("cuda")
    pipe = pipe.to(dtype=dtype)
    # pipe.enable_sequential_cpu_offload()
    if pipe_cpu_offload:
        pipe.enable_model_cpu_offload()

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    # 4. Load dataset
    eval_dataset = RealEstate10KPCDRenderDataset(
        video_root_dir=video_root_dir,
        image_size=(height, width), 
        sample_n_frames=num_frames,
    )
    
    None_prompt = True
    if prompt:
        None_prompt = False
    print(eval_dataset.dataset)
    
    for camera_idx in range(start_camera_idx, end_camera_idx):
        # Get data
        data_dict = eval_dataset[camera_idx]
        reference_video = data_dict['video']
        anchor_video = data_dict['anchor_video']
        print(eval_dataset.dataset[camera_idx],seed)
        
        if None_prompt:
            # Set output directory
            output_path_file = os.path.join(output_path, f"{camera_idx:05d}_{seed}_out.mp4")
            prompt = data_dict['caption']
        else:
            # Set output directory
            output_path_file = os.path.join(output_path, f"{prompt[:10]}_{camera_idx:05d}_{seed}_out.mp4")

        if image is None:
            input_images = reference_video[0].unsqueeze(0)
        else:
            input_images = torch.tensor(np.array(Image.open(image))).permute(2,0,1).unsqueeze(0)/255
            pixel_transforms = [transforms.Resize((480, 720)),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
            for transform in pixel_transforms:
                input_images = transform(input_images)

        # if image is None:
        #     input_images = reference_video[:24]
        # else:
        #     input_images = torch.tensor(np.array(Image.open(image))).permute(2,0,1)/255
        #     pixel_transforms = [transforms.Resize((480, 720)),
        #                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        #     for transform in pixel_transforms:
        #         input_images = transform(input_images)
            
        reference_frames = [to_pil_image(frame) for frame in ((reference_video)/2+0.5)]
        
        output_path_file_reference = output_path_file.replace("_out.mp4", "_reference.mp4")
        output_path_file_out_reference = output_path_file.replace(".mp4", "_reference.mp4")
        
        if infer_with_mask:
            try:
                video_mask = 1 - torch.from_numpy(np.load(os.path.join(eval_dataset.root_path,'masks',eval_dataset.dataset[camera_idx]+'.npz'))['mask']*1)
            except:
                print('using derived mask')
                video_mask = get_black_region_mask_tensor(anchor_video)
            
            if pool_style == 'max':
                controlnet_output_mask = maxpool_mask_tensor(video_mask[1:]).flatten().unsqueeze(0).unsqueeze(-1).to('cuda')
            elif pool_style == 'avg':
               controlnet_output_mask = avgpool_mask_tensor(video_mask[1:]).flatten().unsqueeze(0).unsqueeze(-1).to('cuda')
        else:
            controlnet_output_mask = None
        # if os.path.isfile(output_path_file):
        #     continue
        
        # 5. Generate the video frames based on the prompt.
        # `num_frames` is the Number of frames to generate.
        # This is the default value for 6 seconds video and 8 fps and will plus 1 frame for the first frame and 49 frames.
        video_generate_all = pipe(
            image=input_images,
            anchor_video=anchor_video,
            controlnet_output_mask=controlnet_output_mask,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate，changed to 49 for diffusers version `0.30.3` and after.
            use_dynamic_cfg=use_dynamic_cfg,  # This id used for DPM Sechduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
            controlnet_weights=controlnet_weights,
            controlnet_guidance_start=controlnet_guidance_start,
            controlnet_guidance_end=controlnet_guidance_end,
        ).frames
        video_generate = video_generate_all[0]

        # 6. Export the generated frames to a video file. fps must be 8 for original video.
        export_to_video(video_generate, output_path_file, fps=8)
        export_to_video(reference_frames, output_path_file_reference, fps=8)
        out_reference_frames = [
            stack_images_horizontally(frame_reference, frame_out)
            for frame_out, frame_reference in zip(video_generate, reference_frames)
            ]
        
        anchor_video = [to_pil_image(frame) for frame in ((anchor_video)/2+0.5)]
        out_reference_frames = [
            stack_images_horizontally(frame_out, frame_reference)
            for frame_out, frame_reference in zip(out_reference_frames, anchor_video)
            ]
        export_to_video(out_reference_frames, output_path_file_out_reference, fps=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, default=None, help="The description of the video to be generated")
    parser.add_argument("--image", type=str, default=None, help="The reference image of the video to be generated")
    parser.add_argument(
        "--video_root_dir",
        type=str,
        required=True,
        help="The path of the video for controlnet processing.",
    )
    parser.add_argument(
        "--base_model_path", type=str, default="THUDM/CogVideoX-5b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--controlnet_model_path", type=str, default="TheDenk/cogvideox-5b-controlnet-hed-v1", help="The path of the controlnet pre-trained model to be used"
    )
    parser.add_argument("--controlnet_weights", type=float, default=0.5, help="Strenght of controlnet")
    parser.add_argument("--use_zero_conv", action="store_true", default=False, help="Use zero conv")
    parser.add_argument("--infer_with_mask", action="store_true", default=False, help="add mask to controlnet")
    parser.add_argument("--pool_style", default='max', help="max pool or avg pool")
    parser.add_argument("--controlnet_guidance_start", type=float, default=0.0, help="The stage when the controlnet starts to be applied")
    parser.add_argument("--controlnet_guidance_end", type=float, default=0.5, help="The stage when the controlnet end to be applied")
    parser.add_argument("--use_dynamic_cfg", type=bool, default=True, help="Use dynamic cfg")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument(
        "--output_path", type=str, default="./output", help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--start_camera_idx", type=int, default=0)
    parser.add_argument("--end_camera_idx", type=int, default=1)
    parser.add_argument("--controlnet_transformer_num_attn_heads", type=int, default=None)
    parser.add_argument("--controlnet_transformer_attention_head_dim", type=int, default=None)
    parser.add_argument("--controlnet_transformer_out_proj_dim_factor", type=int, default=None)
    parser.add_argument("--controlnet_transformer_out_proj_dim_zero_init", action="store_true", default=False, help=("Init project zero."),
    )
    parser.add_argument("--downscale_coef", type=int, default=8)
    parser.add_argument("--vae_channels", type=int, default=16)
    parser.add_argument("--controlnet_input_channels", type=int, default=6)
    parser.add_argument("--controlnet_transformer_num_layers", type=int, default=8)
    parser.add_argument("--enable_model_cpu_offload", action="store_true", default=False, help="Enable model CPU offload")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        image=args.image,
        video_root_dir=args.video_root_dir,
        base_model_path=args.base_model_path,
        use_zero_conv=args.use_zero_conv,
        controlnet_model_path=args.controlnet_model_path,
        controlnet_weights=args.controlnet_weights,
        controlnet_guidance_start=args.controlnet_guidance_start,
        controlnet_guidance_end=args.controlnet_guidance_end,
        use_dynamic_cfg=args.use_dynamic_cfg,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        start_camera_idx=args.start_camera_idx,
        end_camera_idx=args.end_camera_idx,
        controlnet_transformer_num_attn_heads=args.controlnet_transformer_num_attn_heads,
        controlnet_transformer_attention_head_dim=args.controlnet_transformer_attention_head_dim,
        controlnet_transformer_out_proj_dim_factor=args.controlnet_transformer_out_proj_dim_factor,
        controlnet_transformer_num_layers=args.controlnet_transformer_num_layers,
        downscale_coef=args.downscale_coef,
        controlnet_input_channels=args.controlnet_input_channels,
        infer_with_mask=args.infer_with_mask,
        pool_style=args.pool_style,
        pipe_cpu_offload=args.enable_model_cpu_offload,
    )
