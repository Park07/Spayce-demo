from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from core.llm_agent import CrowdSafetyLLM


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY")
        return 1

    llm = CrowdSafetyLLM()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    result = llm.analyze_scene(
        frame,
        {
            "latest_current_risk": 0.62,
            "active_tracks": 14,
            "dominant_direction": "EAST",
            "predicted_bottleneck_risk": 0.55,
        },
    )
    print(f"provider={llm.provider}")
    print(f"model={llm.model_name}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
