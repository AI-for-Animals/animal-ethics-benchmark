import os
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

from anai.apis import Agent, AnthropicAPI

load_dotenv()  # Load environment variables from .env file


@pytest.fixture
def mock_anthropic_client():
    with patch("anthropic.Anthropic") as mock_client:
        yield mock_client


def test_anthropic_api_initialization(mock_anthropic_client):
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
        api = AnthropicAPI()
    mock_anthropic_client.assert_called_once_with(api_key="test_key")
    assert api.model == "claude-3-5-sonnet-20240620"  # Check default model


def test_anthropic_api_initialization_with_custom_model(mock_anthropic_client):
    with patch.dict(
        os.environ, {"ANTHROPIC_API_KEY": "test_key", "ANTHROPIC_MODEL": "custom-model"}
    ):
        api = AnthropicAPI()
    assert api.model == "custom-model"


def test_generate_response(mock_anthropic_client):
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="Test response")]
    mock_anthropic_client.return_value.messages.create.return_value = mock_message

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
        api = AnthropicAPI()
        response = api.generate_response("Test prompt", system_prompt="Test system")

    assert response == "Test response"
    mock_anthropic_client.return_value.messages.create.assert_called_once_with(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0.0,
        messages=[{"role": "user", "content": "Test prompt"}],
        system="Test system",
    )


def test_create_agent():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
        api = AnthropicAPI()
        agent = api.create_agent("Test system prompt")

    assert isinstance(agent, Agent)
    assert agent.system_prompt == "Test system prompt"
    assert agent.api == api


def test_agent_send_message():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
        api = AnthropicAPI()
        agent = api.create_agent("Test system prompt")

    with patch.object(api, "generate_response", return_value="Test response"):
        response = agent.send_message("Test message")

    assert response == "Test response"
    assert len(agent.conversation_history) == 2
    assert agent.conversation_history[0] == {"role": "user", "content": "Test message"}
    assert agent.conversation_history[1] == {
        "role": "assistant",
        "content": "Test response",
    }


@pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("ANTHROPIC_MODEL")),
    reason="ANTHROPIC_API_KEY and ANTHROPIC_MODEL environment variables are not set",
)
@pytest.mark.live_api
def test_live_anthropic_api():
    api = AnthropicAPI()
    response = api.generate_response(
        "I'm testing an API, can you respond with a very short string that includes the name of the US president in 2024?"
    )

    assert isinstance(response, str)
    assert len(response) > 0
    assert "Biden" in response
