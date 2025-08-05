import os
from typing import Any, Dict, Optional, Sequence, Type

import httpx
from pydantic import BaseModel

from ..logger import get_logger
from .base import BaseLLMResponse, ChatMessage, LLMClient, RetryConfig

try:
    import ollama
except ImportError:
    raise ImportError(
        "ollama is not installed. Please install it with `uv sync --extra ollama`."
    )

LOGGER = get_logger(__name__)


class OllamaClient(LLMClient):
    """Client for interacting with Ollama LLM models.

    This client implements the LLMClient interface for Ollama models.
    It supports both completion and chat-based interactions.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        retry_config: Optional[RetryConfig | dict[str, Any]] = None,
    ):
        """Initialize the Ollama client.

        Args:
            model (str): The model to use.
            temperature (float, optional): Sampling temperature. Defaults to 0.2.
            retry_config (Optional[RetryConfig | dict[str, Any]], optional): Retry configuration. Defaults to None.
        """
        if not os.getenv("OLLAMA_HOST"):
            raise ValueError(
                "OLLAMA_HOST environment variable is not set. Please set it with your Ollama host."
            )

        super().__init__(temperature=temperature, retry_config=retry_config)

        if self.retry_config:
            transport = httpx.AsyncHTTPTransport(
                retries=self.retry_config.attempts,
            )
            self.client = ollama.AsyncClient(
                host=os.getenv("OLLAMA_HOST"), transport=transport
            )

            if (
                self.retry_config.initial_delay
                or self.retry_config.max_delay
                or self.retry_config.exp_base
            ):
                LOGGER.warning(
                    "Initial delay, max delay, and exp base are not supported for Ollama."
                )
        else:
            self.client = ollama.AsyncClient(host=os.getenv("OLLAMA_HOST"))
        self.model = model

    async def complete(
        self,
        instructions: str,
        system_prompt: Optional[str] = None,
        response_schema: Type[BaseModel] = BaseLLMResponse,
    ) -> Dict[str, Any]:
        """Complete a prompt using Ollama.

        Args:
            instructions (str): The prompt/instructions to send to the model.
            system_prompt (Optional[str], optional): System prompt to prepend. Defaults to None.
            response_schema (Type[BaseModel], optional): Response schema to use. Defaults to BaseLLMResponse.

        Returns:
            Dict[str, Any]: The model's response parsed as a dictionary.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": instructions.strip()})

        response = await self.client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature},
            format=response_schema.model_json_schema(),
        )

        if response.message.content is None:
            raise ValueError("No response from model")

        return response_schema.model_validate_json(
            response.message.content
        ).model_dump()

    async def complete_chat(
        self,
        messages: Sequence[ChatMessage],
        response_schema: Type[BaseModel] = BaseLLMResponse,
    ) -> Dict[str, Any]:
        """Complete a batch of chat messages using Ollama.

        This method handles more complex chat completions with multiple messages and
        additional configuration options.

        Args:
            messages (Sequence[ChatMessage]): List of chat messages to send to the model.
            response_schema (Type[BaseModel], optional): Response schema to use. Defaults to BaseLLMResponse.

        Returns:
            Dict[str, Any]: The model's response parsed as a dictionary.
        """
        ollama_messages = []
        for message in messages:
            ollama_messages.append(
                {"role": message.role, "content": str(message.content).strip()}
            )

        response = await self.client.chat(
            model=self.model,
            messages=ollama_messages,
            options={"temperature": self.temperature},
            format=response_schema.model_json_schema(),
        )

        if response.message.content is None:
            raise ValueError("No response from model")

        return response_schema.model_validate_json(
            response.message.content
        ).model_dump()
