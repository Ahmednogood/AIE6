import os
import requests

class ChatModel:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set.")

    def run(self, messages):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "messages": messages
        }

        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]
