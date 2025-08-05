from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Type

from pydantic import BaseModel


class BaseLLMResponse(BaseModel):
    """Base class for LLM responses.

    This class defines the structure of the response from an LLM.
    """

    response: str


@dataclass
class ChatMessage:
    """Represents a message in a chat conversation.

    A ChatMessage contains a role (e.g. 'user', 'assistant', 'system') and optional content.
    This class is used to structure conversations with LLM models.

    Attributes:
        role (str): The role of the message sender (e.g. 'user', 'assistant', 'system')
        content (Optional[str]): The content of the message. Defaults to None.
    """

    role: str
    content: str | dict[str, Any] | None


@dataclass
class RetryConfig:
    """Configuration for retry options.

    Attributes:
        attempts: The number of attempts to make.
        initial_delay: The initial delay in seconds.
        max_delay: The maximum delay in seconds.
        exp_base: The base of the exponential delay.
    """

    attempts: int
    initial_delay: Optional[float] = None
    max_delay: Optional[float] = None
    exp_base: Optional[float] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients.

    This class defines the interface that all LLM clients must implement.
    """

    def __init__(
        self,
        temperature: float = 0.2,
        retry_config: Optional[RetryConfig | dict[str, Any]] = None,
    ):
        """Initialize the LLM client.

        Args:
            temperature (float, optional): Sampling temperature. Defaults to 0.2.
            retry_config (Optional[RetryConfig | dict[str, Any]], optional): Retry configuration. Defaults to None.
        """
        self.temperature = temperature
        self.retry_config: RetryConfig | None
        if isinstance(retry_config, dict):
            self.retry_config = RetryConfig(**retry_config)
        else:
            self.retry_config = retry_config

    @abstractmethod
    async def complete(
        self,
        instructions: str,
        system_prompt: Optional[str] = None,
        response_schema: Type[BaseModel] = BaseLLMResponse,
    ) -> Dict[str, Any]:
        """Complete a prompt using the LLM.

        Args:
            instructions (str): The prompt/instructions to send to the LLM.
            system_prompt (Optional[str], optional): System prompt to prepend. Defaults to None.
            response_schema (Type[BaseModel], optional): Response schema to use. Defaults to BaseLLMResponse.

        Returns:
            Dict[str, Any]: The LLM's response parsed as a dictionary.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented.
        """
        raise NotImplementedError

    @abstractmethod
    async def complete_chat(
        self,
        messages: Sequence[ChatMessage],
        response_schema: Type[BaseModel] = BaseLLMResponse,
    ) -> Dict[str, Any]:
        """Complete a series of chat messages using the LLM.

        Args:
            messages (Sequence[ChatMessage]): List of chat messages to send to the LLM.
            response_schema (Type[BaseModel], optional): Response schema to use. Defaults to BaseLLMResponse.

        Returns:
            Dict[str, Any]: The LLM's response parsed as a dictionary.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented.
        """
        raise NotImplementedError
