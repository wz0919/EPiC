# EPiC: Efficient Video Camera Control Learning with Precise Anchor-Video Guidance

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://zunwang1.github.io/Epic)  [![arXiv](https://img.shields.io/badge/arXiv-2505.21876-b31b1b.svg)](http://arxiv.org/abs/2505.21876)

#### [Zun Wang](https://zunwang1.github.io/), [Jaemin Cho](https://j-min.io/),  [Jialu Li](https://jialuli-luka.github.io/), [Han Lin](https://hl-hanlin.github.io/), [Jaehong Yoon](https://jaehong31.github.io), [Yue Zhang](https://zhangyuejoslin.github.io/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

<div align="center">
  <video src="https://zunwang1.github.io/images/epic.mp4" controls autoplay loop muted width="600">
    Your browser does not support the video tag.
  </video>
</div>

## ✅ To-Do Checklist

- ✅ Release source training & inference code
- ✅ Provide training data and processing scripts
- ✅ Provide example inference for I2V and V2V
- ✅ V2V customized inference pipeline
- ⏳ I2V customized inference pipeline (Coming Soon!)


## 🚀  Setup

### 1. Clone EPiC
```
git clone --recursive https://github.com/wz0919/EPiC.git
cd EPiC
```

### 2. Setup environments
```
conda create -n epic python=3.10
conda activate epic
pip install -r requirements.txt
```

### 3. Downloading Pretrained Models
Download [CogVideoX-5B-I2V](https://github.com/THUDM/CogVideo) (Base Model), [RAFT](https://huggingface.co/THUDM/CogVideoX-5b-I2V) (To Extract dense optical flow for masking source videos), [Depth-Crafter](https://github.com/Tencent/DepthCrafter) (For video depth estimation), [Qwen2.5-VL-7B-Instruct](https://github.com/QwenLM/Qwen2.5-VL) (For getting detailed captions over videos) with the script
```
bash download/download_models.sh
```

### 🎬 Demo Inference
We provide processed sample test sets in `data/test_i2v` and `data/test_i2v`. You can have a try with our pretrained model (in `out/EPiC_pretrained`) by
```
bash scripts/inference.sh test_v2v
```
and
```
bash scripts/inference.sh test_i2v
```


## 🧠 Training

### 1. Downloading Training Data
Download the ~5K training videos from [EPiC_Data](https://huggingface.co/datasets/ZunWang/EPiC_Data/tree/main) by
```
cd data/train
wget https://huggingface.co/datasets/ZunWang/EPiC_Data/resolve/main/train.zip
unzip train.zip
```
(Optional) Download the extracted vae latents by (You can also extract the latents yourself, which may take several hours)
```
wget https://huggingface.co/datasets/ZunWang/EPiC_Data/resolve/main/train_joint_latents.zip
unzip train_joint_latents.zip
```

### 2.  Preprocessing
Extract caption embeddings (please specify the GPU list in `preprocess.sh`)
```
cd preprocess
bash preprocess.sh caption
```
(Optional) Extract vae latents 
```
bash preprocess.sh latent
```

After preprocessing, your data folder should look like:
```
data/
├── test_i2v/
├── test_v2v/
└── train/
    ├── caption_embs/
    ├── captions/
    ├── joint_latents/
    ├── masked_videos/
    ├── masks/
    └── videos/
```

### Custom Training Data (Optional)
You can prepare you own videos + captions.
To do so, you need to first prepare the them like `train/videos`, `train/captions`. Then
```
bash preprocess.sh masking
```
To get the corresponding masked anchor videos from estimated dense optical flow, and
```
bash preprocess.sh caption
bash preprocess.sh latent
```
To get the extracted textual embeddings and visual latents.

### 3. Training
Editing GPU configs in `scripts/train_with_latent.sh` and `training/accelerate_config_machine.yaml`→ `num_processes`, then
```
bash scripts/train_with_latent.sh
```
You can stop training after 500 iteration, which will take less than 2 hours on 8xH100 GPUs.

(Alternatively: bash scripts/train.sh for online latent encoding, but much slower)

## 🧪 Inference

### 1. V2V Inference
example inference data processing script 
```
cd inference/v2v_data
bash get_anchor_videos.sh v2v_try
```
The processed data will be saved to `data/v2v_try`.
You can modify camera pos type, operation mode, and other parameters to get anchor videos following your own trajectory, please refer to [configuration document](inference/v2v_data/config_help.md) for setup.
Then inference with 
``` 
bash scripts/inference.sh v2v_try
```

### 2. I2V Inference
Coming soon!

## 📚 Acknowledgements
- This code mainly builds upon [CogVideoX-ControlNet](https://github.com/TheDenk/cogvideox-controlnet) and [AC3D](https://github.com/snap-research/ac3d)
- This code uses the original CogVideoX model [CogVideoX](https://github.com/THUDM/CogVideo/tree/main)
- The v2v data processing pipeline largely builds upon [TrajectoryCrafter](https://github.com/TrajectoryCrafter)

## 🔗 Related Works
A non-exhaustive list of related works includes: [CogVideoX](https://github.com/THUDM/CogVideo/tree/main), [ViewCrafter](https://github.com/Drexubery/ViewCrafter), [GCD](https://gcd.cs.columbia.edu/), [NVS-Solver](https://github.com/ZHU-Zhiyu/NVS_Solver), [DimensionX](https://github.com/wenqsun/DimensionX), [ReCapture](https://generative-video-camera-controls.github.io/), [TrajAttention](https://xizaoqu.github.io/trajattn/), [GS-DiT](https://wkbian.github.io/Projects/GS-DiT/), [DaS](https://igl-hkust.github.io/das/), [RecamMaster](https://github.com/KwaiVGI/ReCamMaster), [TrajectoryCrafter](https://github.com/TrajectoryCrafter/TrajectoryCrafter), [GEN3C](https://research.nvidia.com/labs/toronto-ai/GEN3C/), [CAT4D](https://cat-4d.github.io/), [Uni3C](https://github.com/ewrfcas/Uni3C), [AC3D](https://github.com/snap-research/ac3d), [RealCam-I2V](https://github.com/ZGCTroy/RealCam-I2V), [CamCtrl3D](https://camctrl3d.github.io/)...