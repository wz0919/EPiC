import os
import random
import json
import torch

import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np

from torch.utils.data.dataset import Dataset
from packaging import version as pver
from decord import VideoReader

from safetensors.torch import load_file
  
class RealEstate10KPCDRenderLatentCapEmbDataset(Dataset):
    def __init__(
            self,
            video_root_dir,
            text_embedding_path
    ):
        root_path = video_root_dir
        self.root_path = root_path
        self.latent_root = os.path.join(self.root_path, 'joint_latents')
        self.source_video_root = os.path.join(self.root_path, 'videos')
        self.captions_root = os.path.join(self.root_path, 'captions')
        self.dataset = sorted([n.replace('.safetensors','') for n in os.listdir(self.latent_root)])
        self.length = len(self.dataset)
        self.text_embedding_path = text_embedding_path
        self.mask_root = os.path.join(self.root_path, 'masks')

    def get_batch(self, idx):
        clip_name = self.dataset[idx]
        cap_emb_path = os.path.join(self.text_embedding_path, clip_name + '.pt')
        video_caption_emb = torch.load(cap_emb_path, weights_only=True)
        joint_latent_path = os.path.join(self.latent_root, clip_name + '.safetensors')
        joint_latent = load_file(joint_latent_path, device='cpu')['joint_latent']
        video_reader = VideoReader(os.path.join(self.source_video_root, clip_name + '.mp4'))
        indices = [0]
        first_frame = torch.from_numpy(video_reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).contiguous()
        first_frame = (first_frame / 255.)*2-1
        
        T = joint_latent.shape[2] // 2
        source_latent = joint_latent[:, :, :T]
        anchor_latent = joint_latent[:, :, T:]
        masks = np.load(os.path.join(self.mask_root, clip_name + '.npz'))['mask']*1.0
        masks = torch.from_numpy(masks).unsqueeze(1)
        return source_latent, anchor_latent, first_frame, masks, video_caption_emb, clip_name
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        while True:
            try:
                source_latent, anchor_latent, image, mask, video_caption_emb, clip_name = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length - 1)
        data = {
            'source_latent': source_latent,
            'anchor_latent': anchor_latent,
            'image': image,
            'caption_emb': video_caption_emb, 
            'mask': mask
        }
        return data