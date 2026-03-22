from core.zones import build_default_zones, compute_zone_metrics


def test_build_default_zones() -> None:
    zones = build_default_zones(1000, 500)
    assert len(zones) >= 3
    assert all("rect" in z for z in zones)


def test_zone_metrics_with_people() -> None:
    zones = [{"name": "Exit A", "type": "exit", "rect": (0, 0, 200, 200)}]
    tracked_people = [{"centroid": (100, 100)}, {"centroid": (150, 150)}, {"centroid": (300, 300)}]
    metrics = compute_zone_metrics(zones, tracked_people, frame_area=1000 * 1000)
    assert metrics[0]["count"] == 2
    assert metrics[0]["local_density"] > 0
