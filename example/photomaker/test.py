import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import jtorch.nn as nn
import numpy as np
import random
import os
from PIL import Image

from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, DDIMScheduler

from photomaker import PhotoMakerStableDiffusionXLPipeline
# gloal variable and function
def image_grid(imgs, rows, cols, size_after_resize):
    assert len(imgs) == rows*cols

    w, h = size_after_resize, size_after_resize

    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        img = img.resize((w,h))
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

base_model_path = 'SG161222/RealVisXL_V3.0'
device = "cuda"
save_path = "./outputs"

from huggingface_hub import hf_hub_download

photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")

pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to(device)

pipe.load_photomaker_adapter(
    os.path.dirname(photomaker_ckpt),
    subfolder="",
    weight_name=os.path.basename(photomaker_ckpt),
    trigger_word="img"
)
pipe.id_encoder.to(device)



#pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
#pipe.fuse_lora()

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe.set_adapters(["photomaker"], adapter_weights=[1.0])
pipe.fuse_lora()

# import os
# import torch
# from huggingface_hub import hf_hub_download
# from photomaker import PhotoMakerStableDiffusionXLPipeline

# # Define model paths
# civitai_model_path = "./civitai_models"
# os.makedirs(civitai_model_path, exist_ok=True)

# # Define and download base model
# base_model_name = "sdxlUnstableDiffusers_v11.safetensors"
# base_model_path = os.path.join(civitai_model_path, base_model_name)
# if not os.path.exists(base_model_path):
    # base_model_path = hf_hub_download(repo_id="Paper99/sdxlUnstableDiffusers_v11", filename=base_model_name, repo_type="model")

# # Define and download LoRA model
# lora_model_name = "xl_more_art-full.safetensors"
# lora_path = os.path.join(civitai_model_path, lora_model_name)
# if not os.path.exists(lora_path):
    # lora_path = hf_hub_download(repo_id="Paper99/sdxlUnstableDiffusers_v11", filename=lora_model_name, repo_type="model")

# # Verify symbolic links
# base_model_resolved_path = os.path.realpath(base_model_path)
# lora_resolved_path = os.path.realpath(lora_path)
# print(f"Resolved base model path: {base_model_resolved_path}")
# print(f"Resolved LoRA model path: {lora_resolved_path}")

# # Download PhotoMaker checkpoint to cache
# photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")

# # Initialize pipeline with resolved paths
# pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    # r"/root/.cache/huggingface/hub/models--Paper99--sdxlUnstableDiffusers_v11",
    # torch_dtype=torch.float32,
    # original_config_file=None,
    # local_files_only=True,
# ).to('cuda')
