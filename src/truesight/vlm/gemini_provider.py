"""
Google Gemini vision provider using the current `google-genai` SDK.

NOTE: the older `google-generativeai` package (import google.generativeai)
is fully deprecated by Google — this uses its replacement, `google-genai`
(import google.genai), which is a different package with a different API.
See: https://github.com/google-gemini/deprecated-generative-ai-python

Requires GEMINI_API_KEY in the environment.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .base_provider import BaseVLMProvider
from .prompts import RESPONSE_SCHEMA


class GeminiProvider(BaseVLMProvider):
    name = "gemini"

    def __init__(self, model: str = "gemini-3.6-flash", api_key: str | None = None):
        self.model_name = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set — check .env.example")
        # Imported lazily so this module can be imported without the
        # `google-genai` package installed (e.g. for unit tests).
        from google import genai
        self._genai = genai
        self._client = genai.Client(api_key=self.api_key)

    async def _call(self, image_path: str, prompt: str) -> dict[str, Any]:
        from google.genai import types

        image_bytes = open(image_path, "rb").read()
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

        response = await self._client.aio.models.generate_content(
            model=self.model_name,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=RESPONSE_SCHEMA,
            ),
        )
        return json.loads(response.text)