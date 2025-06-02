from huggingface_hub import snapshot_download

def download_model():
    snapshot_download(
        repo_id="tencent/DepthCrafter",
        local_dir="../pretrained/DepthCrafter",
        local_dir_use_symlinks=False,
    )
    snapshot_download(
        repo_id="stabilityai/stable-video-diffusion-img2vid",
        local_dir="../pretrained/stable-video-diffusion-img2vid",
        local_dir_use_symlinks=False,
    )
    snapshot_download(
        repo_id= "Qwen/Qwen2.5-VL-7B-Instruct",
        local_dir="../pretrained/Qwen2.5-VL-7B-Instruct",
        local_dir_use_symlinks=False,
    )
    snapshot_download(
        repo_id="THUDM/CogVideoX-5b-I2V",
        local_dir="../pretrained/CogVideoX-5b-I2V",
        local_dir_use_symlinks=False,
    )

download_model()