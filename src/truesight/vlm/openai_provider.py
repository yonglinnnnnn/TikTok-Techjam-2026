from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from .base_provider import BaseVLMProvider
from .prompts import RESPONSE_SCHEMA


class OpenAIProvider(BaseVLMProvider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set — check .env.example")
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=self.api_key)

    @staticmethod
    def _encode_image(image_path: str) -> str:
        data = Path(image_path).read_bytes()
        return base64.b64encode(data).decode("utf-8")

    async def _call(self, image_path: str, prompt: str) -> dict[str, Any]:
        b64 = self._encode_image(image_path)
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "vlm_consistency_check", "schema": RESPONSE_SCHEMA, "strict": True},
            },
        )
        return json.loads(response.choices[0].message.content)