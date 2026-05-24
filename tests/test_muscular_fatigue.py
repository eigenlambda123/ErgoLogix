import environmental
from environmental import analyze_environment


def test_muscular_fatigue_index_computation(monkeypatch):
    # Mock fetch_open_meteo to return controlled values: 30C, 50% humidity
    def fake_fetch_open_meteo(lat, lon, timeout=10):
        return {
            'latitude': lat,
            'longitude': lon,
            'temperature_c': 30.0,
            'humidity_percent': 50.0,
            'cloud_cover_percent': 0.0,
            'rain_probability_percent': 0.0,
            'wind_speed_kph': 0.0,
            'apparent_temperature_c': 30.0,
        }

    monkeypatch.setattr(environmental, 'fetch_open_meteo', fake_fetch_open_meteo)

    env = analyze_environment(37.0, -122.0, 'sitting', weight_kg=70.0, duration_minutes=60.0)
    assert 'muscular_fatigue_index' in env
    # For these inputs MFI should equal base MET-hours * multiplier => 1.3 * 1.0
    assert abs(env['muscular_fatigue_index'] - 1.3) < 1e-6


def test_muscular_fatigue_multiplier_effect(monkeypatch):
    # Cooler baseline
    def fake_cool(lat, lon, timeout=10):
        return {
            'latitude': lat,
            'longitude': lon,
            'temperature_c': 25.0,
            'humidity_percent': 40.0,
            'cloud_cover_percent': 0.0,
            'rain_probability_percent': 0.0,
            'wind_speed_kph': 0.0,
            'apparent_temperature_c': 25.0,
        }

    # Hotter conditions
    def fake_hot(lat, lon, timeout=10):
        return {
            'latitude': lat,
            'longitude': lon,
            'temperature_c': 35.0,
            'humidity_percent': 80.0,
            'cloud_cover_percent': 0.0,
            'rain_probability_percent': 0.0,
            'wind_speed_kph': 0.0,
            'apparent_temperature_c': 35.0,
        }

    monkeypatch.setattr(environmental, 'fetch_open_meteo', fake_cool)
    env_cool = analyze_environment(37.0, -122.0, 'standing', weight_kg=70.0, duration_minutes=60.0)

    monkeypatch.setattr(environmental, 'fetch_open_meteo', fake_hot)
    env_hot = analyze_environment(37.0, -122.0, 'standing', weight_kg=70.0, duration_minutes=60.0)

    assert env_hot['muscular_fatigue_index'] > env_cool['muscular_fatigue_index']
