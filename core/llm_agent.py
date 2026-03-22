from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict

import cv2
import numpy as np

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class CrowdSafetyLLM:
    """
    Core Agentic Execution Layer - Step 6 & 7.
    Combines an anonymized frame with structured telemetry to produce
    an operator-readable risk assessment and action plan.
    """

    def __init__(
        self,
        api_key: str | None = None,
        use_mock: bool = False,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.use_mock = use_mock
        self.provider = provider or os.environ.get("LLM_PROVIDER", "").strip().lower()
        self.model_name = model
        self.base_url = base_url
        self.client = None
        self.gemini_model = None
        self.last_error = ""

        if use_mock:
            print("[LLM] Mock mode enabled - will return simulated responses.")
            return

        if not self.provider:
            if api_key or os.environ.get("OPENAI_API_KEY"):
                self.provider = "openai_compat"
            elif os.environ.get("GEMINI_API_KEY"):
                self.provider = "gemini"
            else:
                raise ValueError(
                    "No LLM API key configured. Set OPENAI_API_KEY for OpenAI/OpenRouter-style providers "
                    "or GEMINI_API_KEY for Gemini, or pass use_mock=True."
                )

        if self.provider == "openai_compat":
            self._init_openai_compat(api_key=api_key, model=model, base_url=base_url)
        elif self.provider == "gemini":
            self._init_gemini(api_key=api_key, model=model)
        else:
            raise ValueError(
                f"Unsupported LLM provider '{self.provider}'. "
                "Use 'openai_compat' or 'gemini'."
            )

    def _init_openai_compat(
        self,
        api_key: str | None,
        model: str | None,
        base_url: str | None,
    ) -> None:
        if OpenAI is None:
            raise ImportError(
                "openai package not found. Install it with: pip install openai"
            )

        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        resolved_model = model or os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"

        if not resolved_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set."
            )

        self.provider = "openai_compat"
        self.model_name = resolved_model
        self.base_url = resolved_base_url
        self.client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
        print(f"[LLM] Connected to OpenAI-compatible API using model '{self.model_name}'.")

    def _init_gemini(self, api_key: str | None, model: str | None) -> None:
        resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY")
        resolved_model = model or os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash"

        if not resolved_api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set."
            )
        if genai is None:
            raise ImportError(
                "google.generativeai package not found. Install it with: pip install google-generativeai"
            )

        try:
            genai.configure(api_key=resolved_api_key)
            self.gemini_model = genai.GenerativeModel(resolved_model)
            self.provider = "gemini"
            self.model_name = resolved_model
            print(f"[LLM] Connected to Gemini API using model '{self.model_name}'.")
        except Exception as exc:
            raise RuntimeError(f"Failed to configure Gemini API: {exc}") from exc

    def analyze_scene(self, anonymized_frame: np.ndarray, telemetry_data: Dict[str, Any]) -> Dict[str, str]:
        if self.use_mock:
            return self._mock_response()

        try:
            if self.provider == "openai_compat":
                return self._analyze_with_openai_compat(anonymized_frame, telemetry_data)
            if self.provider == "gemini":
                return self._analyze_with_gemini(anonymized_frame, telemetry_data)
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[LLM] API Error: {exc}")

        return {
            "reasoning": f"Failed to connect to {self.provider or 'LLM'} API.",
            "description": "API Connection Error.",
            "risk_level": "Unknown",
            "action": f"Check provider configuration and credentials. Last error: {self.last_error}",
        }

    def _analyze_with_openai_compat(
        self,
        anonymized_frame: np.ndarray,
        telemetry_data: Dict[str, Any],
    ) -> Dict[str, str]:
        if self.client is None or not self.model_name:
            raise RuntimeError("OpenAI-compatible client is not configured.")

        image_url = self._encode_frame_as_data_url(anonymized_frame)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Live telemetry data from core sensors:\n"
                                    f"{json.dumps(telemetry_data, indent=2)}\n\n"
                                    "Analyze the image and metrics to provide the security action plan."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    },
                ],
            )
        except Exception as exc:
            if "support image input" not in str(exc).lower():
                raise
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {
                        "role": "user",
                        "content": (
                            "The selected model does not support image input, so analyze telemetry only.\n\n"
                            "Live telemetry data from core sensors:\n"
                            f"{json.dumps(telemetry_data, indent=2)}\n\n"
                            "Infer the crowd state and provide the security action plan."
                        ),
                    },
                ],
            )
        content = response.choices[0].message.content or ""
        return self._parse_llm_json(content)

    def _analyze_with_gemini(
        self,
        anonymized_frame: np.ndarray,
        telemetry_data: Dict[str, Any],
    ) -> Dict[str, str]:
        if self.gemini_model is None:
            raise RuntimeError("Gemini client is not configured.")

        success, buffer = cv2.imencode(".jpg", anonymized_frame)
        if not success:
            raise ValueError("Could not encode image")

        image_part = {
            "mime_type": "image/jpeg",
            "data": buffer.tobytes(),
        }
        response = self.gemini_model.generate_content(
            [
                self._system_prompt(),
                image_part,
                (
                    "Live telemetry data from core sensors:\n"
                    f"{json.dumps(telemetry_data, indent=2)}\n\n"
                    "Analyze the image and metrics to provide the security action plan."
                ),
            ],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        return self._parse_llm_json(response.text)

    @staticmethod
    def _encode_frame_as_data_url(frame: np.ndarray) -> str:
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            raise ValueError("Could not encode image")
        image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{image_b64}"

    @staticmethod
    def _system_prompt() -> str:
        return """
You are an expert Security and Crowd Flow Analyst.
You will receive a privacy-blurred CCTV image and structured telemetry.

TASK:
Synthesize the visual crowd state with telemetry to identify crowd pressure,
bottleneck risk, and the best operator action.

OUTPUT FORMAT:
Return EXACTLY valid JSON matching this schema:
{
  "reasoning": "A concise 1-2 sentence explanation grounded in the telemetry and image.",
  "description": "A 1-sentence concise description of the crowd state suitable for an operator.",
  "risk_level": "Low" | "Medium" | "High" | "Critical",
  "action": "A highly specific operational command."
}
""".strip()

    @staticmethod
    def _parse_llm_json(content: str) -> Dict[str, str]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("LLM response was not valid JSON.")
            parsed = json.loads(content[start : end + 1])

        required_fields = {"reasoning", "description", "risk_level", "action"}
        missing = required_fields - set(parsed.keys())
        if missing:
            raise ValueError(f"LLM response missing fields: {missing}")
        return {
            "reasoning": str(parsed["reasoning"]),
            "description": str(parsed["description"]),
            "risk_level": str(parsed["risk_level"]),
            "action": str(parsed["action"]),
        }

    @staticmethod
    def _mock_response() -> Dict[str, str]:
        return {
            "reasoning": "[MOCK] Simulated crowd-flow inference indicates rapid density growth and severe localized slowdown.",
            "description": "[MOCK] High density observed near Gateway A causing critical slowdowns.",
            "risk_level": "High",
            "action": "[MOCK] Reroute inbound foot traffic to Gate B and broadcast delay warnings.",
        }
