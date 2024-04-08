import os
import time

from openai import OpenAI

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

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _ask(self, content, **kwargs):
        message = {'role': 'user', 'content': content}

        model = kwargs.get('model', self.model)
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[message],
                max_tokens=self.max_tokens,
                n=1,
                stop=None,
                temperature=self.temperature
            )
        except Exception as e:
            print(f"Retry in {self.retry_time} seconds to avoid GPT rate limit")
            time.sleep(self.retry_time)
            return self._ask(content, **kwargs)

        response_text = response.choices[0].message.content.strip()
        return response_text
