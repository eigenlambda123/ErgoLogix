from unittest.mock import MagicMock

import pytest

import environmental


def test_compute_thermal_fatigue_multiplier_thresholds():
    assert environmental.compute_thermal_fatigue_multiplier(29.0, 65.0) == 1.0
    assert environmental.compute_thermal_fatigue_multiplier(31.0, 72.0) == 1.08


def test_compute_calorie_burn_uses_workspace_mode():
    calories = environmental.compute_calorie_burn(70.0, 60.0, 'sitting')
    assert calories == pytest.approx(95.55)


def test_fetch_user_location_uses_ipapi_payload(monkeypatch):
    payload = {
        'latitude': 37.7749,
        'longitude': -122.4194,
        'city': 'San Francisco',
        'region': 'California',
        'country_name': 'United States',
    }

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload

    monkeypatch.setattr(environmental.requests, 'get', MagicMock(return_value=response))

    result = environmental.fetch_user_location()

    assert result['latitude'] == 37.7749
    assert result['longitude'] == -122.4194
    assert result['city'] == 'San Francisco'
    assert result['region'] == 'California'
    assert result['country'] == 'United States'
    assert result['source'] == 'ipapi'


def test_analyze_environment_uses_open_meteo_payload(monkeypatch):
    payload = {
        'current_weather': {
            'temperature': 32.2,
            'windspeed': 7.4,
            'time': '2026-05-24T12:00',
        },
        'hourly': {
            'time': ['2026-05-24T11:00', '2026-05-24T12:00'],
            'relativehumidity_2m': [55, 72],
            'cloudcover': [10, 40],
            'apparent_temperature': [31.0, 35.5],
            'precipitation_probability': [5, 20],
        },
    }

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload

    monkeypatch.setattr(environmental.requests, 'get', MagicMock(return_value=response))

    result = environmental.analyze_environment(
        latitude=37.7,
        longitude=-122.4,
        workspace_mode='standing',
        weight_kg=80.0,
        duration_minutes=90.0,
    )

    assert result['temperature_c'] == 32.2
    assert result['humidity_percent'] == 72.0
    assert result['cloud_cover_percent'] == 40.0
    assert result['rain_probability_percent'] == 20.0
    assert result['wind_speed_kph'] == 7.4
    assert result['apparent_temperature_c'] == 35.5
    assert result['thermal_fatigue_multiplier'] == 1.11
    assert result['met'] == 2.0
    assert result['calories_burned'] == pytest.approx(252.0)


def test_analyze_environment_falls_back_on_error(monkeypatch):
    monkeypatch.setattr(environmental.requests, 'get', MagicMock(side_effect=Exception('timeout')))

    result = environmental.analyze_environment(
        latitude=0.0,
        longitude=0.0,
        workspace_mode='walking_pad',
        weight_kg=75.0,
        duration_minutes=60.0,
    )

    assert result['temperature_c'] == 30.0
    assert result['humidity_percent'] == 50.0
    assert result['thermal_fatigue_multiplier'] == 1.0
    assert result['met'] == 3.5
    assert result['calories_burned'] == pytest.approx(275.62)