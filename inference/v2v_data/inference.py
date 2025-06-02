from demo import GetAnchorVideos
import os
from datetime import datetime
import argparse
import torch


def get_parser():
    parser = argparse.ArgumentParser()

    ## general
    parser.add_argument('--video_path', type=str, help='Input path')
    parser.add_argument(
        '--out_dir', type=str, required=True, help='Output dir'
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='The device to use'
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help='Experiment name, use video file name by default',
    )
    parser.add_argument(
        '--save_name',
        type=str,
        default=None,
        help='Experiment name, use video file name by default',
    )
    parser.add_argument(
        '--seed', type=int, default=43, help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--video_length', type=int, default=49, help='Length of the video frames'
    )
    parser.add_argument('--fps', type=int, default=10, help='Fps for saved video')
    parser.add_argument(
        '--stride', type=int, default=1, help='Sampling stride for input video'
    )
    parser.add_argument('--server_name', type=str, help='Server IP address')

    ## render
    parser.add_argument(
        '--radius_scale',
        type=float,
        default=1.0,
        help='Scale factor for the spherical radius',
    )
    parser.add_argument('--camera', type=str, default='traj', help='traj or target')
    parser.add_argument(
        '--mode', type=str, default='gradual', help='gradual, bullet or direct'
    )
    parser.add_argument(
        '--mask', action='store_true', default=False, help='Clean the pcd if true'
    )
    parser.add_argument(
        '--traj_txt',
        type=str,
        help="Required for 'traj' camera, a txt file that specify camera trajectory",
    )
    parser.add_argument(
        '--target_pose',
        nargs=5,
        type=float,
        help="Required for 'target' mode, specify target camera pose, <theta phi r x y>",
    )
    parser.add_argument(
        '--near', type=float, default=0.0001, help='Near clipping plane distance'
    )
    parser.add_argument(
        '--far', type=float, default=10000.0, help='Far clipping plane distance'
    )
    parser.add_argument('--anchor_idx', type=int, default=0, help='One GT frame')

    ## diffusion
    parser.add_argument(
        '--low_gpu_memory_mode',
        type=bool,
        default=False,
        help='Enable low GPU memory mode',
    )
    # parser.add_argument('--model_name', type=str, default='checkpoints/CogVideoX-Fun-V1.1-5b-InP', help='Path to the model')
    parser.add_argument(
        '--model_name',
        type=str,
        default='../../pretrained/CogVideoX-Fun-V1.1-5b-InP',
        help='Path to the model',
    )
    parser.add_argument(
        '--sampler_name',
        type=str,
        choices=["Euler", "Euler A", "DPM++", "PNDM", "DDIM_Cog", "DDIM_Origin"],
        default='DDIM_Origin',
        help='Choose the sampler',
    )
    # parser.add_argument('--transformer_path', type=str, default='checkpoints/TrajectoryCrafter/crosstransformer', help='Path to the pretrained transformer model')
    parser.add_argument(
        '--transformer_path',
        type=str,
        default="../../pretrained/TrajectoryCrafter",
        help='Path to the pretrained transformer model',
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        nargs=2,
        default=[384, 672],
        help='Sample size as [height, width]',
    )
    parser.add_argument(
        '--diffusion_guidance_scale',
        type=float,
        default=6.0,
        help='Guidance scale for inference',
    )
    parser.add_argument(
        '--diffusion_inference_steps',
        type=int,
        default=50,
        help='Number of inference steps',
    )
    parser.add_argument(
        '--prompt', type=str, default=None, help='Prompt for video generation'
    )
    parser.add_argument(
        '--negative_prompt',
        type=str,
        default="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
        help='Negative prompt for video generation',
    )
    parser.add_argument(
        '--refine_prompt',
        type=str,
        default=". The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.",
        help='Prompt for video generation',
    )
    parser.add_argument('--qwen_path', type=str, default="../../pretrained/Qwen2.5-VL-7B-Instruct")

    ## depth
    # parser.add_argument('--unet_path', type=str, default='checkpoints/DepthCrafter', help='Path to the UNet model')
    parser.add_argument(
        '--unet_path',
        type=str,
        default="../../pretrained/DepthCrafter",
        help='Path to the UNet model',
    )

    # parser.add_argument('--pre_train_path', type=str, default='checkpoints/stable-video-diffusion-img2vid-xt', help='Path to the pre-trained model')
    parser.add_argument(
        '--pre_train_path',
        type=str,
        default="../../pretrained/stable-video-diffusion-img2vid",
        help='Path to the pre-trained model',
    )
    parser.add_argument(
        '--cpu_offload', type=str, default='model', help='CPU offload strategy'
    )
    parser.add_argument(
        '--depth_inference_steps', type=int, default=5, help='Number of inference steps'
    )
    parser.add_argument(
        '--depth_guidance_scale',
        type=float,
        default=1.0,
        help='Guidance scale for inference',
    )
    parser.add_argument(
        '--window_size', type=int, default=110, help='Window size for processing'
    )
    parser.add_argument(
        '--overlap', type=int, default=25, help='Overlap size for processing'
    )
    parser.add_argument(
        '--max_res', type=int, default=1024, help='Maximum resolution for processing'
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()  # infer config.py
    opts = parser.parse_args()
    opts.weight_dtype = torch.bfloat16
    pvd = GetAnchorVideos(opts)
    if opts.mode == 'gradual':
        pvd.infer_gradual(opts)
    elif opts.mode == 'direct':
        pvd.infer_direct(opts)
    elif opts.mode == 'bullet':
        pvd.infer_bullet(opts)
    elif opts.mode == 'image':
        pvd.infer_image(opts)
    elif opts.mode == 'start_end':
        pvd.infer_start_end(opts)
    elif opts.mode == 'zoom':
        pvd.infer_zoom(opts)