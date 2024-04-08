import os
import time
import json

from openai import AzureOpenAI

client = AzureOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
azure_endpoint="xxx",
api_version="xxx")

from .chatbot_template import ChatBotTemplate


class AzureGPTChat(ChatBotTemplate):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        self.model = cfg.AZURE_GPT.MODEL
        self.frequency_penalty = cfg.AZURE_GPT.FREQUENCY_PENALTY
        self.max_tokens = cfg.AZURE_GPT.MAX_TOKENS
        self.top_p = cfg.AZURE_GPT.TOP_P
        self.presence_penalty = cfg.AZURE_GPT.PRESENCE_PENALTY
        self.stop_tokens = cfg.AZURE_GPT.STOP_TOKENS
        self.temperature = cfg.AZURE_GPT.TEMPERATURE

    def _ask(self, content, **kwargs):
        message = {'role': 'user', 'content': content}

        model = kwargs.get('model', self.model)
        try:
            response = client.chat.completions.create(model=model,
                messages=[message],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=self.stop_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
        except Exception as e:
            print(e)
            error_message = json.loads(e.http_body)
            status = error_message['statusCode']
            message = error_message['message']
            if status == 429:
                wait_time = int(message.split(' ')[-2])
                print(f"Retry in {wait_time} seconds to avoid AzureGPT rate limit")
                time.sleep(wait_time)
                return self._ask(content, **kwargs)
            else:
                print(f"Retry in {self.retry_time} seconds to avoid AzureGPTGPT rate limit")
                time.sleep(self.retry_time)
                return self._ask(content, **kwargs)

        response_text = response.choices[0].message.content.strip()
        return response_text
