from core.risk import compute_crowd_stress_risk, predict_crowd_stress_risk


def test_risk_score_in_range() -> None:
    result = compute_crowd_stress_risk(
        people_density=0.0015,
        density_growth=0.04,
        zone_congestion=0.7,
        avg_speed=2.5,
        directional_consistency=0.4,
        bottleneck_pressure=0.5,
        anomaly_count=3,
    )
    assert 0.0 <= result["current_risk_score"] <= 1.0
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert isinstance(result["reasons"], list)


def test_predicted_risk_trend() -> None:
    rising = [0.2, 0.28, 0.35, 0.4]
    predicted = predict_crowd_stress_risk(rising, horizon_seconds=15)
    assert predicted >= rising[-1]
