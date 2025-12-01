# app/inference.py
import os, time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
)

from .models import MODEL_REGISTRY, DEFAULT_SAFETY

# 输出目录
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

@dataclass
class GenConfig:
    model_key: str = "sd15"          # "sd15" | "sdxl" | "leosam"
    prompt: str = "a scenic mountain landscape at sunrise"
    negative_prompt: Optional[str] = ""
    width: int = 512
    height: int = 512
    steps: int = 28
    guidance: float = 7.0
    seed: Optional[int] = None
    device: Optional[str] = None     # "cuda" | "cpu" | "mps"
    safety: bool = DEFAULT_SAFETY

_PIPELINES = {}
_DEVICE = None

def _pick_device(pref: Optional[str] = None) -> torch.device:
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    if pref in {"cuda","cpu","mps"}:
        d = pref
    else:
        if torch.cuda.is_available():
            d = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            d = "mps"
        else:
            d = "cpu"
    _DEVICE = torch.device(d)
    return _DEVICE

def _load_pipeline(model_key: str, device: torch.device, safety: bool = True):
    if model_key in _PIPELINES:
        return _PIPELINES[model_key]
    cfg = MODEL_REGISTRY[model_key]
    hf_id = cfg["hf_id"]
    torch_dtype = torch.float16 if device.type in ("cuda","mps") else torch.float32
    if cfg["type"] == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(hf_id, torch_dtype=torch_dtype, use_safetensors=True)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(hf_id, torch_dtype=torch_dtype, use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if device.type == "cuda":
        try: pipe.enable_xformers_memory_efficient_attention()
        except Exception: pass
    try: pipe.enable_attention_slicing()
    except Exception: pass
    if not safety and hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    pipe.to(device)
    _PIPELINES[model_key] = pipe
    return pipe

def generate(config: GenConfig) -> Tuple[str, dict]:
    device = _pick_device(config.device or os.getenv("TORCH_DEVICE"))
    pipe = _load_pipeline(config.model_key, device, config.safety)
    generator = torch.Generator(device.type).manual_seed(int(config.seed)) if config.seed is not None else None
    t0 = time.time()
    out = pipe(
        prompt=config.prompt,
        negative_prompt=(config.negative_prompt or None),
        num_inference_steps=int(config.steps),
        guidance_scale=float(config.guidance),
        width=int(config.width),
        height=int(config.height),
        generator=generator,
    )
    dt = time.time() - t0
    image = out.images[0]
    nsfw = bool(getattr(out, "nsfw_content_detected", [False])[0]) if hasattr(out, "nsfw_content_detected") else False
    ts = time.strftime("%Y%m%d-%H%M%S")
    fname = f"{ts}_{config.model_key}_{config.width}x{config.height}.png"
    fpath = os.path.join(OUTPUT_DIR, fname)
    if nsfw:
        Image.new("RGB", image.size, (32,32,32)).save(fpath)
    else:
        image.save(fpath)
    meta = {
        "latency_sec": round(dt, 3),
        "device": device.type,
        "model_key": config.model_key,
        "nsfw_blocked": nsfw,
        "path": fpath,
        "filename": fname,
    }
    return fpath, meta
