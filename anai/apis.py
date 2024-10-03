import os
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


class AnthropicAPI:
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

    def generate_response(
        self,
        prompt: str,
        model: str = os.environ.get(
            "ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"
        ),  # /claude-3-opus-20240229
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a response using the Anthropic API.

        Args:
            prompt (str): The input prompt for the model.
            model (str): The model to use (default: "claude-3-5-sonnet-20240620").
            max_tokens (int): Maximum number of tokens in the response (default: 1000).
            temperature (float): Controls randomness in output (default: 0.7).

        Returns:
            str: The generated response from the model.
        """
        try:
            message = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return ""

    def create_agent(self, system_prompt: Optional[str] = None):
        """
        Create an agent with an optional system prompt.

        Args:
            system_prompt (Optional[str]): The system prompt for the agent.

        Returns:
            Agent: An instance of the Agent class.
        """
        return Agent(self, system_prompt)


class Agent:
    def __init__(self, api: AnthropicAPI, system_prompt: Optional[str] = None):
        self.api = api
        self.system_prompt = system_prompt
        self.conversation_history = []

    def send_message(self, message: str) -> str:
        """
        Send a message to the agent and get a response.

        Args:
            message (str): The message to send to the agent.

        Returns:
            str: The response from the agent.
        """
        full_prompt = self._build_prompt(message)
        response = self.api.generate_response(full_prompt)
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def _build_prompt(self, message: str) -> str:
        """
        Build the full prompt including system prompt and conversation history.

        Args:
            message (str): The current message to be added to the prompt.

        Returns:
            str: The full prompt for the API call.
        """
        prompt = ""
        if self.system_prompt:
            prompt += f"System: {self.system_prompt}\n\n"

        for entry in self.conversation_history:
            prompt += f"{entry['role'].capitalize()}: {entry['content']}\n\n"

        prompt += f"Human: {message}\n\nAssistant:"
        return prompt
