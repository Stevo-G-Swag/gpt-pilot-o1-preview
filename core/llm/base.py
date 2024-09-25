import asyncio
import datetime
import json
from enum import Enum
from time import time
from typing import Any, Callable, Optional, Tuple

import httpx

from core.config import LLMConfig, LLMProvider
from core.llm.convo import Convo
from core.llm.request_log import LLMRequestLog, LLMRequestStatus
from core.log import get_logger

log = get_logger(__name__)


class LLMError(str, Enum):
    KEY_EXPIRED = "key_expired"
    RATE_LIMITED = "rate_limited"
    GENERIC_API_ERROR = "generic_api_error"


class APIError(Exception):
    def __init__(self, message: str):
        self.message = message


class BaseLLMClient:
    provider: LLMProvider

    def __init__(
        self,
        config: LLMConfig,
        *,
        stream_handler: Optional[Callable] = None,
        error_handler: Optional[Callable] = None,
    ):
        self.config = config
        self.stream_handler = stream_handler
        self.error_handler = error_handler
        self._init_client()

    def _init_client(self):
        raise NotImplementedError()

    def _adapt_messages(self, convo: Convo) -> list[dict[str, str]]:
        """
        Default message adaptation method.

        Providers can override this method if they require different message formats.
        """
        messages = []
        for msg in convo.messages:
            messages.append(
                {
                    "role": msg["role"],
                    "content": msg["content"],
                }
            )
        return messages

    async def _make_request(
        self,
        convo: Convo,
        temperature: Optional[float] = None,
        json_mode: bool = False,
    ) -> tuple[str, int, int]:
        raise NotImplementedError()

    async def __call__(
        self,
        convo: Convo,
        *,
        temperature: Optional[float] = None,
        parser: Optional[Callable] = None,
        max_retries: int = 3,
        json_mode: bool = False,
    ) -> Tuple[Any, LLMRequestLog]:
        if temperature is None:
            temperature = self.config.temperature

        messages = self._adapt_messages(convo)
        convo = convo.fork()
        convo.messages = messages

        request_log = LLMRequestLog(
            provider=self.provider,
            model=self.config.model,
            temperature=temperature,
            prompts=convo.prompt_log,
        )

        prompt_length_kb = len(json.dumps(convo.messages).encode("utf-8")) / 1024
        log.debug(
            f"Calling {self.provider.value} model {self.config.model} (temp={temperature}), prompt length: {prompt_length_kb:.1f} KB"
        )
        t0 = time()

        remaining_retries = max_retries
        while True:
            if remaining_retries == 0:
                if request_log.error:
                    last_error_msg = f"Error connecting to the LLM: {request_log.error}"
                else:
                    last_error_msg = "Error parsing LLM response"

                if self.error_handler:
                    should_retry = await self.error_handler(LLMError.GENERIC_API_ERROR, message=last_error_msg)
                    if should_retry:
                        remaining_retries = max_retries
                        continue

                raise APIError(last_error_msg)

            remaining_retries -= 1
            request_log.messages = convo.messages[:]
            request_log.response = None
            request_log.status = LLMRequestStatus.SUCCESS
            request_log.error = None
            response = None

            try:
                response, prompt_tokens, completion_tokens = await self._make_request(
                    convo,
                    temperature=temperature,
                    json_mode=json_mode,
                )
            except Exception as err:
                await self._handle_exception(err, request_log)
                continue

            request_log.response = response
            request_log.prompt_tokens += prompt_tokens
            request_log.completion_tokens += completion_tokens

            if parser:
                try:
                    response = parser(response)
                    break
                except ValueError as err:
                    request_log.error = f"Error parsing response: {err}"
                    request_log.status = LLMRequestStatus.ERROR
                    log.debug(f"Error parsing LLM response: {err}, asking LLM to retry", exc_info=True)
                    convo.assistant(response)
                    convo.user(f"Error parsing response: {err}. Please output your response EXACTLY as requested.")
                    continue
            else:
                break

        t1 = time()
        request_log.duration = t1 - t0

        log.debug(
            f"Total {self.provider.value} response time {request_log.duration:.2f}s, "
            f"{request_log.prompt_tokens} prompt tokens, {request_log.completion_tokens} completion tokens used"
        )

        return response, request_log

    async def _handle_exception(self, err: Exception, request_log: LLMRequestLog):
        log.warning(f"LLM API error: {err}", exc_info=True)
        request_log.error = str(err)
        request_log.status = LLMRequestStatus.ERROR

        if isinstance(err, httpx.ReadTimeout):
            log.warning(f"Read timeout (set to {self.config.read_timeout}s): {err}", exc_info=True)
        elif isinstance(err, httpx.ReadError):
            log.warning(f"Read error: {err}", exc_info=True)
        elif isinstance(err, httpx.HTTPStatusError):
            log.warning(f"HTTP status error: {err}", exc_info=True)
            if err.response.status_code == 429:
                wait_time = self.rate_limit_sleep(err)
                if wait_time:
                    message = f"We've hit {self.config.provider.value} rate limit. Sleeping for {wait_time.seconds} seconds..."
                    if self.error_handler:
                        await self.error_handler(LLMError.RATE_LIMITED, message)
                    await asyncio.sleep(wait_time.seconds)
                else:
                    raise APIError("Rate limit exceeded") from err
            else:
                # Handle other HTTP status errors
                raise APIError(f"HTTP error {err.response.status_code}: {err.response.text}") from err
        else:
            raise APIError(f"LLM API error: {err}") from err

    async def api_check(self) -> bool:
        convo = Convo()
        msg = "This is a connection test. If you can see this, please respond only with 'START' and nothing else."
        convo.user(msg)
        try:
            resp, _log = await self(convo)
            return bool(resp)
        except Exception as e:
            log.error(f"API check failed: {e}")
            return False

    @staticmethod
    def for_provider(provider: LLMProvider) -> type["BaseLLMClient"]:
        from .anthropic_client import AnthropicClient
        from .azure_client import AzureClient
        from .groq_client import GroqClient
        from .openai_client import OpenAIClient

        if provider == LLMProvider.OPENAI:
            return OpenAIClient
        elif provider == LLMProvider.ANTHROPIC:
            return AnthropicClient
        elif provider == LLMProvider.GROQ:
            return GroqClient
        elif provider == LLMProvider.AZURE:
            return AzureClient
        else:
            raise ValueError(f"Unsupported LLM provider: {provider.value}")

    def rate_limit_sleep(self, err: Exception) -> Optional[datetime.timedelta]:
        return None


__all__ = ["BaseLLMClient"]

