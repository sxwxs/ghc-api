"""Provider adapters by API name."""

from ..models import ApiName
from ..provider import ProviderFn
from . import anthropic, openai_chat, openai_responses

PROVIDERS: dict[ApiName, ProviderFn] = {
    "openai_chat": openai_chat.stream,
    "openai_responses": openai_responses.stream,
    "anthropic": anthropic.stream,
}
