from typing import Tuple, Dict

def compose_prompt(context: Dict) -> Tuple[str, str, str]:
    """Return (prompt, negative_prompt, model_key)."""
    cond = (context.get("condition") or "").lower()
    hour = int(context.get("hour", 12))
    temp = float(context.get("temp_c", 15.0))

    # Time of day styling
    if 6 <= hour < 10:
        tod = "soft morning light, warm tone"
    elif 10 <= hour < 17:
        tod = "bright daylight, vibrant colors"
    elif 17 <= hour < 20:
        tod = "golden hour, cinematic lighting"
    else:
        tod = "night scene, moody lighting"

    weather_scene = {
        "clear": "clear sky",
        "partly cloudy": "scattered clouds",
        "cloudy": "overcast sky",
        "rain": "rainy day, wet streets",
        "snow": "fresh snowfall",
        "storm": "dramatic thunderstorm",
    }.get(cond, "dynamic weather")

    prompt = f"city street, {weather_scene}, {tod}, ultra-detailed"
    # Colder/night/storm → prefer SDXL quality; otherwise SD15 speed
    model_key = "sdxl" if (cond in ("storm","snow") or hour < 7 or hour >= 20 or temp < 5) else "sd15"
    negative = "low quality, deformed, extra fingers, watermark, text"
    return prompt, negative, model_key
