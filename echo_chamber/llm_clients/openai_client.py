import os
from typing import Any, Dict, List, Literal, Optional, Sequence, Type

try:
    from openai import DEFAULT_MAX_RETRIES, AsyncAzureOpenAI, AsyncOpenAI
    from openai.types.chat import (
        ChatCompletionSystemMessageParam,
        ChatCompletionUserMessageParam,
    )
    from openai.types.shared.reasoning_effort import ReasoningEffort
except ImportError:
    raise ImportError(
        "openai is not installed. Please install it with `uv sync --extra openai`."
    )

from pydantic import BaseModel

from ..logger import get_logger
from .base import BaseLLMResponse, ChatMessage, LLMClient, RetryConfig

LOGGER = get_logger(__name__)


class OpenAiClient(LLMClient):
    """Client for interacting with OpenAI's API.

    This class implements the LLMClient interface and provides methods to interact
    with OpenAI's API for both standard and Azure deployments. It supports single
    and batch completions with various configuration options.

    Attributes:
        client: An instance of either OpenAI or AzureOpenAI client for making API calls.
    """

    def __init__(
        self,
        model: str,
        llm_api: Literal["openai", "azure", "vllm", "deepseek"] = "openai",
        temperature: float = 0.2,
        reasoning_effort: ReasoningEffort = "low",
        retry_config: Optional[RetryConfig | dict[str, Any]] = None,
    ):
        """Initialize the OpenAI client.

        Args:
            llm_api (Literal["openai", "azure", "vllm", "deepseek"], optional): The API type to use - either "openai" for standard
                OpenAI API or any other value for Azure OpenAI. Defaults to "openai".
            model (str, optional): The model to use.
            temperature (float, optional): Sampling temperature. Defaults to 0.2.
            reasoning_effort (ReasoningEffort, optional): Reasoning effort. Defaults to "low".
            retry_config (Optional[RetryConfig | dict[str, Any]], optional): Retry configuration. Defaults to None.
        """
        super().__init__(temperature=temperature, retry_config=retry_config)
        self.model = model
        self.reasoning_effort = reasoning_effort

        if self.retry_config:
            max_retries = self.retry_config.attempts
            if (
                self.retry_config.initial_delay
                or self.retry_config.max_delay
                or self.retry_config.exp_base
            ):
                LOGGER.warning(
                    "Initial delay, max delay, and exp base are not supported for OpenAI."
                )
        else:
            max_retries = DEFAULT_MAX_RETRIES

        if llm_api == "openai":
            self.client = AsyncOpenAI(max_retries=max_retries)
        elif llm_api == "deepseek":
            if not os.getenv("DEEPSEEK_API_KEY"):
                raise ValueError(
                    "DEEPSEEK_API_KEY environment variable is not set. Please set it with your DeepSeek API key."
                )

            if not os.getenv("DEEPSEEK_BASE_URL"):
                raise ValueError(
                    "DEEPSEEK_BASE_URL environment variable is not set. Please set it with your DeepSeek API base URL."
                )

            self.client = AsyncOpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=os.getenv("DEEPSEEK_BASE_URL"),
                max_retries=max_retries,
            )
        elif llm_api == "vllm":
            if not os.getenv("VLLM_API_KEY"):
                raise ValueError(
                    "VLLM_API_KEY environment variable is not set. Please set it with your VLLM API key."
                )

            if not os.getenv("VLLM_BASE_URL"):
                raise ValueError(
                    "VLLM_BASE_URL environment variable is not set. Please set it with your VLLM base URL."
                )

            self.client = AsyncOpenAI(
                api_key=os.getenv("VLLM_API_KEY"),
                base_url=os.getenv("VLLM_BASE_URL"),
                max_retries=max_retries,
            )
        elif llm_api == "azure":
            self.client = AsyncAzureOpenAI(max_retries=max_retries)
        else:
            raise ValueError(
                f"Invalid LLM_API value: {llm_api}. Please set it to one of: openai, azure, vllm, deepseek."
            )

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
        messages = self._generate_messages(instructions, system_prompt)

        if self.model == "o3-mini":
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_schema,
                reasoning_effort=self.reasoning_effort,  # type: ignore
            )
        else:
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_schema,
                temperature=self.temperature,
            )

        content = response.choices[0].message.parsed
        if content is None:
            if (
                hasattr(response.choices[0].message, "refusal")
                and response.choices[0].message.refusal
            ):
                raise ValueError(
                    f"Request refused due to content policy violation: {response.choices[0].message.refusal}"
                )
            raise ValueError(
                "Received empty response from OpenAI - this may be due to a malicious or inappropriate prompt"
            )

        return content.model_dump()

    @staticmethod
    def _generate_messages(
        instructions: str, system_prompt: Optional[str] = None
    ) -> List[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam]:
        """Generate the message list for the LLM API call.

        Creates a list of message dictionaries in the format expected by OpenAI's API,
        optionally including a system prompt.

        Args:
            instructions (str): The user instructions/prompt to include.
            system_prompt (Optional[str], optional): System prompt to prepend. Defaults to None.

        Returns:
            List[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam]: message dictionaries with 'role' and 'content' keys.
        """
        messages: List[
            ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam
        ] = []
        if system_prompt:
            messages = [
                ChatCompletionSystemMessageParam(
                    role="system", content=system_prompt.strip()
                ),
                ChatCompletionUserMessageParam(
                    role="user", content=instructions.strip()
                ),
            ]
        else:
            messages = [
                ChatCompletionUserMessageParam(
                    role="user", content=instructions.strip()
                ),
            ]
        return messages

    async def complete_chat(
        self,
        messages: Sequence[ChatMessage],
        response_schema: Type[BaseModel] = BaseLLMResponse,
    ) -> Dict[str, Any]:
        """Complete a batch of chat messages using the OpenAI API.

        This method handles more complex chat completions with multiple messages and
        additional configuration options.

        Args:
            messages (Sequence[ChatMessage]): List of chat messages to send to the LLM.
            response_schema (Type[BaseModel], optional): Response schema to use. Defaults to BaseLLMResponse.

        Returns:
            Dict[str, Any]: The LLM's response parsed as a dictionary.
        """
        openai_messages = [
            ChatCompletionSystemMessageParam(
                role="system", content=str(m.content).strip()
            )
            if m.role == "system"
            else ChatCompletionUserMessageParam(
                role="user", content=str(m.content).strip()
            )
            for m in messages
        ]

        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            response_format=response_schema,
        )

        content = response.choices[0].message.parsed
        if content is None:
            if (
                hasattr(response.choices[0].message, "refusal")
                and response.choices[0].message.refusal
            ):
                raise ValueError(
                    f"Request refused due to content policy violation: {response.choices[0].message.refusal}"
                )
            raise ValueError(
                "Received empty response from OpenAI - this may be due to a malicious or inappropriate prompt"
            )

        return content.model_dump()
