import json
from pathlib import Path
import gradio as gr

STATE = Path("state/current.json")

def _load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            return None
    return None

def refresh():
    data = _load_state()
    if not data:
        return None, "No image yet. Click Generate in controller or wait for next cycle."
    url = data.get("result", {}).get("url")
    meta = data.get("result", {})
    subtitle = (
        f"Model: {meta.get('model_key')} | Latency: {meta.get('latency_sec')}s | "
        f"Device: {meta.get('device')} | NSFW blocked: {meta.get('nsfw_blocked')}\n"
        f"Prompt: {data.get('prompt')}\n"
        f"Condition: {data.get('context',{}).get('condition')} | "
        f"Temp: {data.get('context',{}).get('temp_c')}°C"
    )
    if url and not url.startswith("http"):  # served by API at /outputs/...
        url = f"http://127.0.0.1:8000{url}"
    return url, subtitle

def build_ui():
    with gr.Blocks(title="Generative Signage") as demo:
        gr.Markdown("# Ambient-Aware Generative Signage\nAuto-refresh latest image.")
        img = gr.Image(type="filepath", label="Current Image", interactive=False)
        info = gr.Markdown()
        btn = gr.Button("Refresh")
        btn.click(refresh, outputs=[img, info])
        demo.load(refresh, None, [img, info], every=15)
    return demo


if __name__ == "__main__":
    import os
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    build_ui().launch(
        server_name="127.0.0.1",
        server_port=7861,
        show_api=False,  
        share=True,                 
    )
