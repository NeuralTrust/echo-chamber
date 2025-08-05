import os
from typing import Any, Dict, Optional, Sequence, Type

from pydantic import BaseModel

from .base import BaseLLMResponse, ChatMessage, LLMClient, RetryConfig

try:
    from google.genai import Client
    from google.genai.types import (
        GenerateContentConfig,
        HttpOptions,
        HttpRetryOptions,
        ThinkingConfig,
    )
except ImportError:
    raise ImportError(
        "google-genai is not installed. Please install it with `uv sync --extra google`."
    )


class GoogleClient(LLMClient):
    """Client for interacting with Google's Generative AI API.

    This class implements the LLMClient interface and provides methods to interact
    with Google's Generative AI API. It supports single and batch completions with
    various configuration options.

    Attributes:
        model (str): The name of the model to use (e.g., "gemini-2.0-flash")
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.2,
        thinking_budget: int = 0,
        retry_config: Optional[RetryConfig | dict[str, Any]] = None,
    ):
        """Initialize the Google Generative AI client.

        Args:
            model (str, optional): The model to use. Defaults to "gemini-2.0-flash".
            temperature (float, optional): Sampling temperature. Defaults to 0.2.
            thinking_budget (int, optional): The budget for the model to think. Defaults to 0, which means no thinking.
            retry_config (Optional[RetryConfig | dict[str, Any]], optional): Retry configuration. Defaults to None.
        """
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. Please set it with your Google API key."
            )

        super().__init__(temperature=temperature, retry_config=retry_config)
        if self.retry_config:
            retry_options = HttpRetryOptions(
                attempts=self.retry_config.attempts,
                initial_delay=self.retry_config.initial_delay,
                max_delay=self.retry_config.max_delay,
                exp_base=self.retry_config.exp_base,
            )
        else:
            retry_options = None

        self.client = Client(
            api_key=os.getenv("GOOGLE_API_KEY"),
            http_options=HttpOptions(
                retry_options=retry_options,
            ),
        )
        self.model = model
        self.thinking_budget = thinking_budget

    async def complete(
        self,
        instructions: str,
        system_prompt: Optional[str] = None,
        response_schema: Type[BaseModel] = BaseLLMResponse,
    ) -> Dict[str, Any]:
        """Evaluate the response of the LLM using the ground truth and system prompt.

        Args:
            instructions (str): The prompt/instructions to send to the LLM.
            system_prompt (Optional[str], optional): System prompt to prepend. Defaults to None.
            response_schema (Type[BaseModel], optional): Response schema to use. Defaults to BaseLLMResponse.

        Returns:
            Dict[str, str]: The LLM's response parsed as a JSON dictionary.
        """
        if "gemini-2.5" in self.model:
            thinking_config = ThinkingConfig(thinking_budget=self.thinking_budget)
        else:
            thinking_config = None

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=instructions.strip(),
            config=GenerateContentConfig(
                system_instruction=system_prompt.strip() if system_prompt else None,
                response_modalities=["TEXT"],
                temperature=self.temperature,
                response_mime_type="application/json",
                thinking_config=thinking_config,
                response_schema=response_schema,
            ),
        )
        if response.parsed is None:
            raise ValueError("Received empty response from Google")

        return response.parsed.model_dump()  # type: ignore

    async def complete_chat(
        self,
        messages: Sequence[ChatMessage],
        response_schema: Type[BaseModel] = BaseLLMResponse,
    ) -> Dict[str, Any]:
        """Complete a batch of chat messages using the Google Generative AI API.

        This method handles more complex chat completions with multiple messages and
        additional configuration options.

        Args:
            messages (Sequence[ChatMessage]): List of chat messages to send to the LLM.
            response_schema (Type[BaseModel], optional): Response schema to use. Defaults to BaseLLMResponse.

        Returns:
            Dict[str, Any]: The LLM's response parsed as a dictionary.
        """
        user_messages = []
        system_prompt = ""

        for message in messages:
            if message.role == "system":
                system_prompt += str(message.content).strip()
            else:
                user_messages.append(str(message.content).strip())  # type: ignore

        combined_content = "\n\n".join(user_messages)

        if "gemini-2.5" in self.model:
            thinking_config = ThinkingConfig(thinking_budget=self.thinking_budget)
        else:
            thinking_config = None

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=combined_content,
            config=GenerateContentConfig(
                system_instruction=system_prompt if system_prompt else None,
                response_modalities=["TEXT"],
                temperature=self.temperature,
                response_mime_type="application/json",
                thinking_config=thinking_config,
                response_schema=response_schema,
            ),
        )

        if response.parsed is None:
            raise ValueError("Received empty response from Google")

        return response.parsed.model_dump()  # type: ignore
