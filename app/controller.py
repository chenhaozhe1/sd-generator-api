import os, time, json, argparse, requests
from pathlib import Path
from .sensors.collector import collect_context_dict
from .planner.promptor import compose_prompt

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
STATE_DIR = Path(os.getenv("STATE_DIR", "state"))
STATE_DIR.mkdir(parents=True, exist_ok=True)

def run_once():
    ctx = collect_context_dict()
    prompt, negative, model_key = compose_prompt(ctx)
    payload = {
        "prompt": prompt,
        "negative_prompt": negative,
        "model_key": model_key,
        "width": int(os.getenv("GEN_WIDTH", 512)),
        "height": int(os.getenv("GEN_HEIGHT", 512)),
        "steps": int(os.getenv("GEN_STEPS", 28)),
        "guidance": float(os.getenv("GEN_GUIDANCE", 7.0)),
        "safety": os.getenv("GEN_SAFETY", "true").lower() != "false",
    }
    r = requests.post(f"{API_BASE}/generate", json=payload, timeout=900)
    r.raise_for_status()
    data = r.json()
    current = {
        "context": ctx,
        "prompt": prompt,
        "negative": negative,
        "result": data,
        "timestamp": int(time.time()),
    }
    (STATE_DIR / "current.json").write_text(json.dumps(current, indent=2))
    return current

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=900, help="Seconds between generations")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    args = parser.parse_args()
    if args.once:
        run_once(); return
    while True:
        try:
            run_once()
        except Exception as e:
            print("controller error:", e, flush=True)
        time.sleep(max(60, args.interval))

if __name__ == "__main__":
    main()
