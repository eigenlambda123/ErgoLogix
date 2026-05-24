from __future__ import annotations

import math
from typing import Dict, Optional

try:
    import requests
except Exception:
    requests = None


WORKSPACE_MET = {
    'sitting': 1.3,
    'standing': 2.0,
    'walking_pad': 3.5,
}


def normalize_workspace_mode(mode: str) -> str:
    normalized = (mode or 'sitting').strip().lower().replace(' ', '_')
    aliases = {
        'sit': 'sitting',
        'chair': 'sitting',
        'stand': 'standing',
        'walk': 'walking_pad',
        'walkingpad': 'walking_pad',
        'walking_pad': 'walking_pad',
    }
    return aliases.get(normalized, normalized if normalized in WORKSPACE_MET else 'sitting')


def met_for_workspace_mode(mode: str) -> float:
    return WORKSPACE_MET.get(normalize_workspace_mode(mode), WORKSPACE_MET['sitting'])


def compute_calorie_burn(weight_kg: float, duration_minutes: float, workspace_mode: str) -> float:
    if weight_kg <= 0 or duration_minutes <= 0:
        return 0.0
    met = met_for_workspace_mode(workspace_mode)
    calories = (met * 3.5 * weight_kg * duration_minutes) / 200.0
    return round(float(calories), 2)


def compute_thermal_fatigue_multiplier(temperature_c: float, humidity_percent: float) -> float:
    multiplier = 1.0
    if temperature_c > 30.0:
        multiplier += (temperature_c - 30.0) * 0.03
    if humidity_percent > 70.0:
        multiplier += 0.05
    return math.floor(multiplier * 100.0) / 100.0


def _safe_number(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _hourly_value(hourly: Dict, key: str, time_value: Optional[str]) -> Optional[float]:
    values = hourly.get(key)
    times = hourly.get('time')
    if not isinstance(values, list) or not values:
        return None
    if isinstance(times, list) and time_value in times:
        index = times.index(time_value)
        if 0 <= index < len(values):
            return _safe_number(values[index])
    return _safe_number(values[0])


def fetch_open_meteo(latitude: float, longitude: float, timeout: int = 10) -> Optional[Dict[str, float]]:
    if requests is None:
        return None

    url = 'https://api.open-meteo.com/v1/forecast'
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'current_weather': 'true',
        'hourly': 'temperature_2m,relativehumidity_2m,apparent_temperature,cloudcover,precipitation_probability',
        'timezone': 'auto',
    }

    try:
        response = requests.get(url, params=params, timeout=timeout)
    except Exception:
        return None

    if response.status_code != 200:
        return None

    try:
        payload = response.json()
    except Exception:
        return None

    current = payload.get('current_weather') or {}
    hourly = payload.get('hourly') or {}
    current_time = current.get('time')

    temperature_c = _safe_number(current.get('temperature', _hourly_value(hourly, 'temperature_2m', current_time)))
    wind_speed_kph = _safe_number(current.get('windspeed', 0.0))
    humidity_percent = _hourly_value(hourly, 'relativehumidity_2m', current_time)
    cloud_cover_percent = _hourly_value(hourly, 'cloudcover', current_time)
    rain_probability_percent = _hourly_value(hourly, 'precipitation_probability', current_time)
    apparent_temperature_c = _hourly_value(hourly, 'apparent_temperature', current_time)

    return {
        'latitude': _safe_number(latitude),
        'longitude': _safe_number(longitude),
        'temperature_c': round(temperature_c, 2),
        'humidity_percent': round(_safe_number(humidity_percent), 2),
        'cloud_cover_percent': round(_safe_number(cloud_cover_percent), 2),
        'rain_probability_percent': round(_safe_number(rain_probability_percent), 2),
        'wind_speed_kph': round(wind_speed_kph, 2),
        'apparent_temperature_c': round(_safe_number(apparent_temperature_c, temperature_c), 2),
    }


def analyze_environment(
    latitude: float,
    longitude: float,
    workspace_mode: str,
    weight_kg: float,
    duration_minutes: float,
    timeout: int = 10,
) -> Dict[str, float]:
    weather = fetch_open_meteo(latitude, longitude, timeout=timeout)
    if weather is None:
        weather = {
            'latitude': _safe_number(latitude),
            'longitude': _safe_number(longitude),
            'temperature_c': 30.0,
            'humidity_percent': 50.0,
            'cloud_cover_percent': 0.0,
            'rain_probability_percent': 0.0,
            'wind_speed_kph': 0.0,
            'apparent_temperature_c': 30.0,
        }

    multiplier = compute_thermal_fatigue_multiplier(
        weather['temperature_c'],
        weather['humidity_percent'],
    )

    result = dict(weather)
    result['workspace_mode'] = normalize_workspace_mode(workspace_mode)
    result['met'] = round(met_for_workspace_mode(workspace_mode), 2)
    result['calories_burned'] = compute_calorie_burn(weight_kg, duration_minutes, workspace_mode)
    result['thermal_fatigue_multiplier'] = multiplier
    return result