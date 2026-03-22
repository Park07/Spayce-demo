from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_uploaded_file(uploaded_file: Any, target_dir: Path) -> Path:
    ensure_dir(target_dir)
    suffix = Path(uploaded_file.name).suffix
    unique_name = f"{Path(uploaded_file.name).stem}_{uuid.uuid4().hex[:8]}{suffix}"
    file_path = target_dir / unique_name
    with open(file_path, "wb") as out_file:
        out_file.write(uploaded_file.read())
    return file_path


def clamp_01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def risk_level_from_score(score: float, levels: list[tuple[float, str]]) -> str:
    for threshold, label in levels:
        if score <= threshold:
            return label
    return "CRITICAL"
