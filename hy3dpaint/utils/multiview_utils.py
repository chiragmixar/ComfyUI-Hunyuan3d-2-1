# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import torch
import random
import numpy as np
from PIL import Image
from typing import List
import huggingface_hub
from omegaconf import OmegaConf
from diffusers import DiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, DDIMScheduler, UniPCMultistepScheduler
from ..hunyuanpaintpbr.pipeline import HunyuanPaintPipeline

# Global cache for multiviewDiffusionNet instances
# Key: (device, multiview_cfg_path, custom_pipeline) tuple
# Value: multiviewDiffusionNet instance
_MULTIVIEW_MODEL_CACHE = {}
_MULTIVIEW_CACHE_LOCK = None  # For thread safety if needed

def get_cached_multiview_model(config):
    """
    Get a cached multiviewDiffusionNet instance or create a new one.
    This avoids reloading the HuggingFace model on every generation.
    """
    global _MULTIVIEW_MODEL_CACHE
    
    # Create a cache key from config parameters that affect model loading
    cache_key = (
        config.device,
        config.multiview_cfg_path,
        config.custom_pipeline,
        config.multiview_pretrained_path,
    )
    
    if cache_key not in _MULTIVIEW_MODEL_CACHE:
        print(f"[Pipeline Cache] Creating new multiviewDiffusionNet instance (cache miss)")
        _MULTIVIEW_MODEL_CACHE[cache_key] = multiviewDiffusionNet(config)
        print(f"[Pipeline Cache] Model cached successfully")
    else:
        print(f"[Pipeline Cache] Reusing cached multiviewDiffusionNet instance (cache hit)")
        # Ensure the pipeline is on the correct device before reuse
        cached_model = _MULTIVIEW_MODEL_CACHE[cache_key]
        try:
            # Move pipeline components back to CUDA if they were offloaded
            if hasattr(cached_model.pipeline, 'to'):
                cached_model.pipeline.to(config.device)
            print(f"[Pipeline Cache] Pipeline moved to {config.device}")
        except Exception as e:
            print(f"[Pipeline Cache] Warning: Could not move pipeline to device: {e}")
    
    return _MULTIVIEW_MODEL_CACHE[cache_key]

def clear_multiview_cache():
    """Clear the global model cache to free memory."""
    global _MULTIVIEW_MODEL_CACHE
    _MULTIVIEW_MODEL_CACHE.clear()
    torch.cuda.empty_cache()
    print("[Pipeline Cache] Cleared multiview model cache")


class multiviewDiffusionNet:
    def __init__(self, config) -> None:
        self.device = config.device

        cfg_path = config.multiview_cfg_path
        custom_pipeline = config.custom_pipeline
        cfg = OmegaConf.load(cfg_path)
        self.cfg = cfg
        self.mode = self.cfg.model.params.stable_diffusion_config.custom_pipeline[2:]

        # Use local model path instead of HuggingFace download (pre-downloaded during build)
        model_path = "ComfyUI/models/diffusers/hunyuan3d-paintpbr-v2-1"
        print(f"[Pipeline Cache] Loading model from local path: {model_path}")
        
        print(f"[Pipeline Cache] Loading HunyuanPaintPipeline from {model_path}...")
        pipeline = HunyuanPaintPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        )

        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
        pipeline.set_progress_bar_config(disable=False)
        pipeline.eval()
        setattr(pipeline, "view_size", cfg.model.params.get("view_size", 320))
        # NOTE: Disabled CPU offload to fix device mismatch when reusing cached pipeline
        # The pipeline will stay in VRAM which is fine since we're caching it
        # pipeline.enable_model_cpu_offload()
        self.pipeline = pipeline.to(self.device)
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()
        print(f"[Pipeline Cache] Pipeline loaded and configured successfully (no CPU offload for caching)")

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            from ..hunyuanpaintpbr.unet.modules import Dino_v2
            self.dino_v2 = Dino_v2(config.dino_ckpt_path).to(torch.float16)
            self.dino_v2 = self.dino_v2.to(self.device)

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)

    @torch.no_grad()
    def __call__(self, images, conditions, prompt=None, custom_view_size=None, resize_input=False, num_steps=10, guidance_scale=3.0, seed=0):
        pils = self.forward_one(
            images, conditions, prompt=prompt, custom_view_size=custom_view_size, resize_input=resize_input, num_steps=num_steps, guidance_scale=guidance_scale, seed=seed
        )
        return pils

    def forward_one(self, input_images, control_images, prompt=None, custom_view_size=None, resize_input=False, num_steps=10, guidance_scale=3.0, seed=0):
        self.seed_everything(seed)
        custom_view_size = custom_view_size if custom_view_size is not None else self.pipeline.view_size
        
        if not isinstance(input_images, List):
            input_images = [input_images]
            
        if not resize_input:
            input_images = [
                input_image.resize((self.pipeline.view_size, self.pipeline.view_size)) for input_image in input_images
            ]
        else:
            input_images = [input_image.resize((custom_view_size, custom_view_size)) for input_image in input_images]
            
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((custom_view_size, custom_view_size))
            if control_images[i].mode == "L":
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode="1")
        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        kwargs["width"] = custom_view_size
        kwargs["height"] = custom_view_size
        kwargs["num_in_batch"] = num_view
        kwargs["images_normal"] = normal_image
        kwargs["images_position"] = position_image

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            dino_hidden_states = self.dino_v2(input_images[0])
            kwargs["dino_hidden_states"] = dino_hidden_states

        sync_condition = None

        infer_steps_dict = {
            "EulerAncestralDiscreteScheduler": 10,
            "UniPCMultistepScheduler": 10,
            "DDIMScheduler": 10,
            "ShiftSNRScheduler": 10,
        }

        mvd_image = self.pipeline(
            input_images[0:1],
            num_inference_steps=num_steps,
            prompt=prompt,
            sync_condition=sync_condition,
            guidance_scale=guidance_scale,
            **kwargs,
        ).images

        if "pbr" in self.mode:
            mvd_image = {"albedo": mvd_image[:num_view], "mr": mvd_image[num_view:]}
            # mvd_image = {'albedo':mvd_image[:num_view]}
        else:
            mvd_image = {"hdr": mvd_image}

        return mvd_image
