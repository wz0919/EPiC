import os
import random
import json
import torch

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np

from decord import VideoReader
from torch.utils.data.dataset import Dataset
from packaging import version as pver

class RandomHorizontalFlipWithPose(nn.Module):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlipWithPose, self).__init__()
        self.p = p

    def get_flip_flag(self, n_image):
        return torch.rand(n_image) < self.p

    def forward(self, image, flip_flag=None):
        n_image = image.shape[0]
        if flip_flag is not None:
            assert n_image == flip_flag.shape[0]
        else:
            flip_flag = self.get_flip_flag(n_image)

        ret_images = []
        for fflag, img in zip(flip_flag, image):
            if fflag:
                ret_images.append(F.hflip(img))
            else:
                ret_images.append(img)
        return torch.stack(ret_images, dim=0)

class RealEstate10KPCDRenderDataset(Dataset):
    def __init__(
            self,
            video_root_dir,
            sample_n_frames=49,
            image_size=[480, 720],
            shuffle_frames=False,
            hflip_p=0.0,
    ):
        if hflip_p != 0.0:
            use_flip = True
        else:
            use_flip = False
        root_path = video_root_dir
        self.root_path = root_path
        self.sample_n_frames = sample_n_frames
        self.source_video_root = os.path.join(self.root_path, 'videos')
        self.mask_video_root = os.path.join(self.root_path, 'masked_videos')
        self.captions_root = os.path.join(self.root_path, 'captions')
        self.dataset = sorted([n.replace('.mp4','') for n in os.listdir(self.source_video_root)])
        self.length = len(self.dataset)
        sample_size = image_size
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size
        if use_flip:
            pixel_transforms = [transforms.Resize(sample_size),
                                RandomHorizontalFlipWithPose(hflip_p),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        else:
            pixel_transforms = [transforms.Resize(sample_size),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        self.sample_wh_ratio = sample_size[1] / sample_size[0]

        self.pixel_transforms = pixel_transforms
        self.shuffle_frames = shuffle_frames
        self.use_flip = use_flip

    def load_video_reader(self, idx):
        clip_name = self.dataset[idx]
        video_path = os.path.join(self.source_video_root, clip_name + '.mp4')
        video_reader = VideoReader(video_path)
        mask_video_path = os.path.join(self.mask_video_root, clip_name + '.mp4')
        mask_video_reader = VideoReader(mask_video_path)
        caption_path = os.path.join(self.captions_root, clip_name + '.txt')
        if os.path.exists(caption_path):
            caption = open(caption_path, 'r').read().strip()
        else:
            caption = ''
        return clip_name, video_reader, mask_video_reader, caption

    def get_batch(self, idx):
        clip_name, video_reader, mask_video_reader, video_caption = self.load_video_reader(idx)
        if self.use_flip:
            flip_flag = self.pixel_transforms[1].get_flip_flag(self.sample_n_frames)
        else:
            flip_flag = torch.zeros(self.sample_n_frames, dtype=torch.bool)

        indices = np.arange(self.sample_n_frames)
        pixel_values = torch.from_numpy(video_reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        
        anchor_pixels = torch.from_numpy(mask_video_reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).contiguous()
        anchor_pixels = anchor_pixels / 255.
        
        return pixel_values, anchor_pixels, video_caption, flip_flag, clip_name

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        while True:
            try:
                video, anchor_video, video_caption, flip_flag, clip_name = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length - 1)
        if self.use_flip:
            video = self.pixel_transforms[0](video)
            video = self.pixel_transforms[1](video, flip_flag)
            video = self.pixel_transforms[2](video)
            anchor_video = self.pixel_transforms[0](anchor_video)
            anchor_video = self.pixel_transforms[1](anchor_video, flip_flag)
            anchor_video = self.pixel_transforms[2](anchor_video)
        else:
            for transform in self.pixel_transforms:
                video = transform(video)
                anchor_video = transform(anchor_video)
        data = {
            'video': video, 
            'anchor_video': anchor_video,
            'caption': video_caption, 
        }
        return data
    
class RealEstate10KPCDRenderCapEmbDataset(RealEstate10KPCDRenderDataset):
    def __init__(
            self,
            video_root_dir,
            text_embedding_path,
            sample_n_frames=49,
            image_size=[480, 720],
            shuffle_frames=False,
            hflip_p=0.0,
    ):
        super().__init__(
            video_root_dir,
            sample_n_frames=sample_n_frames,
            image_size=image_size,
            shuffle_frames=shuffle_frames,
            hflip_p=hflip_p,
        )
        self.text_embedding_path = text_embedding_path
        self.mask_root = os.path.join(self.root_path, 'masks')

    def get_batch(self, idx):
        clip_name, video_reader, mask_video_reader, video_caption = self.load_video_reader(idx)
        cap_emb_path = os.path.join(self.text_embedding_path, clip_name + '.pt')
        video_caption_emb = torch.load(cap_emb_path, weights_only=True)
        if self.use_flip:
            flip_flag = self.pixel_transforms[1].get_flip_flag(self.sample_n_frames)
        else:
            flip_flag = torch.zeros(self.sample_n_frames, dtype=torch.bool)
        indices = np.arange(self.sample_n_frames)
        pixel_values = torch.from_numpy(video_reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        
        anchor_pixels = torch.from_numpy(mask_video_reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).contiguous()
        anchor_pixels = anchor_pixels / 255.
        try:
            masks = np.load(os.path.join(self.mask_root, clip_name + '.npz'))['mask']*1.0
            masks = torch.from_numpy(masks).unsqueeze(1)
        except:
            threshold = 0.1  # you can adjust this value
            masks = (anchor_pixels.sum(dim=1, keepdim=True) < threshold).float()
        return pixel_values, anchor_pixels, masks, video_caption_emb, flip_flag, clip_name
    
    def __getitem__(self, idx):
        while True:
            try:
                video, anchor_video, mask, video_caption_emb, flip_flag, clip_name = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length - 1)
        if self.use_flip:
            video = self.pixel_transforms[0](video)
            video = self.pixel_transforms[1](video, flip_flag)
            video = self.pixel_transforms[2](video)
            anchor_video = self.pixel_transforms[0](anchor_video)
            anchor_video = self.pixel_transforms[1](anchor_video, flip_flag)
            anchor_video = self.pixel_transforms[2](anchor_video)
        else:
            for transform in self.pixel_transforms:
                video = transform(video)
                anchor_video = transform(anchor_video)
        data = {
            'video': video, 
            'anchor_video': anchor_video,
            'caption_emb': video_caption_emb, 
            'mask': mask
        }
        return data