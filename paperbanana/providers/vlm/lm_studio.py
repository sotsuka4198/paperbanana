"""LM Studio VLM provider — local OpenAI-compatible API."""

from __future__ import annotations

from typing import Optional

import structlog
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from paperbanana.core.utils import image_to_base64
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class LMStudioVLM(VLMProvider):
    """VLM provider for LM Studio's local OpenAI-compatible API.

    LM Studio exposes an OpenAI-compatible endpoint at http://localhost:1234/v1.
    No API key is required by default.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "",
        api_key: Optional[str] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._client = None

    @property
    def name(self) -> str:
        return "lm_studio"

    @property
    def model_name(self) -> str:
        return self._model or "(default)"

    def _get_client(self):
        """Lazy-init an async httpx client pointed at the LM Studio API."""
        if self._client is None:
            import httpx

            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=300.0,
            )
        return self._client

    def is_available(self) -> bool:
        return True

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        response_format: Optional[str] = None,
    ) -> str:
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build multimodal content array (vision images + text)
        content = []
        if images:
            for img in images:
                b64 = image_to_base64(img)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()
        text = data["choices"][0]["message"]["content"]

        logger.debug(
            "LM Studio response",
            model=self._model,
            usage=data.get("usage"),
        )
        return text
