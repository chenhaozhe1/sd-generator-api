import os, time, requests
from dataclasses import dataclass, asdict
from typing import Dict

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

@dataclass
class Context:
    city: str
    lat: float
    lon: float
    temp_c: float
    condition: str
    hour: int
    is_day: bool

def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

def _wcode_to_text(code: int) -> str:
    if code in (0, 1): return "clear"
    if code in (2,):   return "partly cloudy"
    if code in (3, 45, 48): return "cloudy"
    if code in (51, 53, 55, 61, 63, 65): return "rain"
    if code in (71, 73, 75, 77, 85, 86): return "snow"
    if code in (95, 96, 99): return "storm"
    return "mixed"

def collect_weather() -> Context:
    city = os.getenv("CITY", "Logan, UT")
    lat = _get_env_float("LAT", 41.7355)
    lon = _get_env_float("LON", -111.8344)
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "hourly": "weathercode,temperature_2m",
        "timezone": os.getenv("TZ", "America/Denver"),
    }
    try:
        r = requests.get(OPEN_METEO_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("current_weather", {})
        temp_c = float(data.get("temperature", 10.0))
        hour = int(time.localtime().tm_hour)
        code = int(data.get("weathercode", 0))
        cond = _wcode_to_text(code)
        is_day = bool(data.get("is_day", 1))
        return Context(city, lat, lon, temp_c, cond, hour, is_day)
    except Exception:
        hour = int(time.localtime().tm_hour)
        return Context(city, lat, lon, 12.0, "clear", hour, hour in range(7,20))

def collect_context_dict() -> Dict:
    return asdict(collect_weather())
