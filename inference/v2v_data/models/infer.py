import gc
import os
import numpy as np
import torch

from diffusers.training_utils import set_seed
# from models.depth_crafter_ppl import DepthCrafterPipeline
# from models.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from DepthCrafter.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from DepthCrafter.depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter

class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_train_path: str,
        cpu_offload: str = "model",
        device: str = "cuda:0",
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        # load weights of other components from the provided checkpoint
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_train_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # for saving memory, we can offload the model to CPU, or even run the model sequentially to save more memory
        if cpu_offload is not None:
            if cpu_offload == "sequential":
                # This will slow, but save more memory
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to(device)
        # enable attention slicing and xformers memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
            print("Xformers is not enabled")
        self.pipe.enable_attention_slicing()

    def infer(
        self,
        frames,
        near,
        far,
        num_denoising_steps: int,
        guidance_scale: float,
        window_size: int = 110,
        overlap: int = 25,
        seed: int = 42,
        track_time: bool = True,
    ):
        set_seed(seed)

        # inference the depth map using the DepthCrafter pipeline
        with torch.inference_mode():
            res = self.pipe(
                frames,
                height=frames.shape[1],
                width=frames.shape[2],
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=window_size,
                overlap=overlap,
                track_time=track_time,
            ).frames[0]
        # convert the three-channel output to a single channel depth map
        res = res.sum(-1) / res.shape[-1]
        # normalize the depth map to [0, 1] across the whole video
        depths = (res - res.min()) / (res.max() - res.min())
        # visualize the depth map and save the results
        # vis = vis_sequence_depth(res)
        # save the depth map and visualization with the target FPS
        depths = torch.from_numpy(depths).unsqueeze(1)  # 49 576 1024 ->
        depths *= 3900  # compatible with da output
        depths[depths < 1e-5] = 1e-5
        depths = 10000.0 / depths
        depths = depths.clip(near, far)

        return depths
