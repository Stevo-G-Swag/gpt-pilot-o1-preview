import datetime
import re
from typing import Optional

import tiktoken
from httpx import Timeout
from openai import AsyncOpenAI, RateLimitError

from core.config import LLMProvider
from core.llm.base import BaseLLMClient
from core.llm.convo import Convo
from core.log import get_logger

log = get_logger(__name__)
tokenizer = tiktoken.get_encoding("cl100k_base")


class OpenAIClient(BaseLLMClient):
    provider = LLMProvider.OPENAI
    stream_options = {"include_usage": True}

    def _init_client(self):
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=Timeout(
                max(self.config.connect_timeout, self.config.read_timeout),
                connect=self.config.connect_timeout,
                read=self.config.read_timeout,
            ),
        )

    def _adapt_messages(self, convo: Convo) -> list[dict[str, str]]:
        """
        Adapt messages for OpenAI models.

        Some models do not support the 'system' role. This method adjusts the messages
        accordingly based on the model being used.
        """
        messages = []

        # List of models that do not support 'system' role
        models_without_system_role = ["o1-preview", "o1-mini"]

        for msg in convo.messages:
            role = msg["role"]
            content = msg["content"]

            # If the model does not support 'system' role, adjust the message
            if self.config.model in models_without_system_role and role == "system":
                # Convert 'system' role to 'user' role
                role = "user"

            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return messages

    async def _make_request(
        self,
        convo: Convo,
        temperature: Optional[float] = None,
        json_mode: bool = False,
    ) -> tuple[str, int, int]:
        # Use the stream parameter from the configuration
        stream_value = self.config.stream

        completion_kwargs = {
            "model": self.config.model,
            "messages": convo.messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "stream": stream_value,
        }

        if self.stream_options and stream_value:
            completion_kwargs["stream_options"] = self.stream_options

        if json_mode:
            completion_kwargs["response_format"] = {"type": "json_object"}

        if stream_value:
            # Streaming response handling
            stream = await self.client.chat.completions.create(**completion_kwargs)
            response = []
            prompt_tokens = 0
            completion_tokens = 0

            async for chunk in stream:
                if chunk.usage:
                    prompt_tokens += chunk.usage.prompt_tokens
                    completion_tokens += chunk.usage.completion_tokens

                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content
                if not content:
                    continue

                response.append(content)
                if self.stream_handler:
                    await self.stream_handler(content)

            response_str = "".join(response)

            # Signal the end of streaming
            if self.stream_handler:
                await self.stream_handler(None)
        else:
            # Non-streaming response handling
            response = await self.client.chat.completions.create(**completion_kwargs)
            response_str = ""
            prompt_tokens = 0
            completion_tokens = 0

            if hasattr(response, 'usage') and response.usage:
                prompt_tokens += response.usage.prompt_tokens
                completion_tokens += response.usage.completion_tokens

            if response.choices:
                response_str = response.choices[0].message.content

                if self.stream_handler:
                    # Send the entire response at once
                    await self.stream_handler(response_str)
                    await self.stream_handler(None)
            else:
                log.warning("No choices returned in OpenAI response")

        if prompt_tokens == 0 and completion_tokens == 0:
            # Estimate token counts if not provided
            prompt_tokens = sum(3 + len(tokenizer.encode(msg["content"])) for msg in convo.messages)
            completion_tokens = len(tokenizer.encode(response_str))
            log.warning(
                "OpenAI response did not include token counts, estimating with tiktoken: "
                f"{prompt_tokens} input tokens, {completion_tokens} output tokens"
            )

        return response_str, prompt_tokens, completion_tokens

    def rate_limit_sleep(self, err: RateLimitError) -> Optional[datetime.timedelta]:
        """
        OpenAI rate limits docs:
        https://platform.openai.com/docs/guides/rate-limits/error-mitigation
        Limit reset times are in "2h32m54s" format.
        """

        headers = err.response.headers
        if "x-ratelimit-remaining-tokens" not in headers:
            return None

        remaining_tokens = headers["x-ratelimit-remaining-tokens"]
        time_regex = r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?"
        if remaining_tokens == 0:
            match = re.search(time_regex, headers["x-ratelimit-reset-tokens"])
        else:
            match = re.search(time_regex, headers["x-ratelimit-reset-requests"])

        if match:
            hours = int(match.group(1)) if match.group(1) else 0
            minutes = int(match.group(2)) if match.group(2) else 0
            seconds = int(match.group(3)) if match.group(3) else 0
            total_seconds = hours * 3600 + minutes * 60 + seconds
        else:
            # Default to 5 seconds if reset time is not provided
            total_seconds = 5

        return datetime.timedelta(seconds=total_seconds)


__all__ = ["OpenAIClient"]

