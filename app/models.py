from os import getenv
from typing import Literal

ModelKey = Literal["sd15", "sdxl", "leosam"]

MODEL_REGISTRY = {
    "sd15": {
        "hf_id": getenv("HF_ID_SD15", "runwayml/stable-diffusion-v1-5"),
        "type": "sd15",
    },
    "sdxl": {
        "hf_id": getenv("HF_ID_SDXL", "stabilityai/stable-diffusion-xl-base-1.0"),
        "type": "sdxl",
    },
    "leosam": {
        # Replace with your exact HF model ID or local path for the LEOSAM SDXL variant
        "hf_id": getenv("HF_ID_LEOSAM", "YOUR_LEOSAM_SDXL_HF_ID_OR_PATH"),
        "type": "sdxl",
    },
}

DEFAULT_SAFETY = True
