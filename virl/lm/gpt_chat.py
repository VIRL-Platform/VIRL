import os
import time

import openai

from .chatbot_template import ChatBotTemplate


class GPTChat(ChatBotTemplate):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        self.model = cfg.GPT.MODEL
        self.max_tokens = cfg.GPT.MAX_TOKENS
        self.temperature = cfg.GPT.TEMPERATURE
        # self.frequency_penalty = cfg.GPT.FREQUENCY_PENALTY
        # self.presence_penalty = cfg.GPT.PRESENCE_PENALTY
        self.retry_time = cfg.GPT.RETRY_TIME

    @staticmethod
    def set_prior():
        openai.api_type = "openai"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = 'https://api.openai.com/v1'
        openai.api_version = None

    def _ask(self, content, **kwargs):
        self.set_prior()
        message = {'role': 'user', 'content': content}

        model = kwargs.get('model', self.model)
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[message],
                max_tokens=self.max_tokens,
                n=1,
                stop=None,
                temperature=self.temperature,
                # frequency_penalty=self.frequency_penalty,
                # presence_penalty=self.presence_penalty
            )
        except Exception as e:
            print(f"Retry in {self.retry_time} seconds to avoid GPT rate limit")
            time.sleep(self.retry_time)
            return self._ask(content, **kwargs)

        response_text = response.choices[0].message['content'].strip()
        return response_text
