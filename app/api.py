import os
import fastapi
from fastapi.responses import FileResponse, HTMLResponse
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .inference import GenConfig, generate, OUTPUT_DIR
from .safety.moderation import moderate_prompt
from .telemetry.metrics import GEN_REQUESTS, GEN_FAILURES, GEN_LAT

app = FastAPI(title="SD Image Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

class GenRequest(BaseModel):
    prompt: str = Field(..., example="a cozy cabin in the woods, golden hour")
    negative_prompt: str | None = Field(None, example="low quality, deformed")
    model_key: str = Field("sd15", description="sd15 | sdxl | leosam")
    width: int = 512
    height: int = 512
    steps: int = 28
    guidance: float = 7.0
    seed: int | None = None
    device: str | None = None
    safety: bool = True

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return fastapi.Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.post("/generate")
def generate_image(req: GenRequest):
    # Input moderation
    allowed, msg = moderate_prompt(req.prompt, req.negative_prompt)
    if not allowed:
        raise HTTPException(status_code=400, detail=msg)

    model_key = req.model_key if req.model_key in ("sd15", "sdxl", "leosam") else "sd15"
    GEN_REQUESTS.labels(model=model_key).inc()

    cfg = GenConfig(
        model_key=model_key,
        prompt=req.prompt,
        negative_prompt=req.negative_prompt or "",
        width=req.width,
        height=req.height,
        steps=req.steps,
        guidance=req.guidance,
        seed=req.seed,
        device=req.device,
        safety=req.safety,
    )

    with GEN_LAT.labels(model=model_key).time():
        try:
            fpath, meta = generate(cfg)
        except Exception:
            GEN_FAILURES.labels(model=model_key).inc()
            raise

    url = f"/outputs/{meta['filename']}"
    return {"url": url, **meta}

@app.post("/generate_bytes")
def generate_image_bytes(req: GenRequest):
    # 输入审查
    allowed, msg = moderate_prompt(req.prompt, req.negative_prompt)
    if not allowed:
        raise HTTPException(status_code=400, detail=msg)

    model_key = req.model_key if req.model_key in ("sd15", "sdxl", "leosam") else "sd15"
    GEN_REQUESTS.labels(model=model_key).inc()

    cfg = GenConfig(
        model_key=model_key,
        prompt=req.prompt,
        negative_prompt=req.negative_prompt or "",
        width=req.width,
        height=req.height,
        steps=req.steps,
        guidance=req.guidance,
        seed=req.seed,
        device=req.device,
        safety=req.safety,
    )

    try:
        with GEN_LAT.labels(model=model_key).time():
            fpath, meta = generate(cfg)
    except Exception as e:
        GEN_FAILURES.labels(model=model_key).inc()
        # 让 500 错误在 /docs 里可读
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Generation error: {e}\n{tb}")

    # 直接把图片文件返回（若保存为 jpg 改 media_type）
    return FileResponse(
        path=fpath,
        media_type="image/png",
        filename=meta["filename"],
    )

@app.get("/gallery", response_class=HTMLResponse)
def gallery():
    imgs = []
    for name in os.listdir(OUTPUT_DIR):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            imgs.append(name)
    imgs.sort(key=lambda n: os.path.getmtime(os.path.join(OUTPUT_DIR, n)), reverse=True)

    tiles = "\n".join(
        f'<div style="display:inline-block;margin:8px;text-align:center">'
        f'<a href="/outputs/{n}" target="_blank"><img src="/outputs/{n}" style="width:256px;height:256px;object-fit:cover;border-radius:8px"/></a>'
        f'<div style="font-family:monospace;font-size:12px">{n}</div>'
        f'</div>'
        for n in imgs
    )
    return f"""
    <html>
      <head><title>Outputs</title></head>
      <body style="padding:16px;font-family:system-ui">
        <h2>Generated Images</h2>
        {'<p>No images yet.</p>' if not imgs else tiles}
      </body>
    </html>
    """
