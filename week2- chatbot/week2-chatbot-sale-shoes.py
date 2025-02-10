import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import gradio as gr

class Config:
    OLLAMA_BASE_URL = "http://localhost:11434/v1"
    OLLAMA_API_KEY = "ollama"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    }
    DEFAULT_MODEL = "llama3.2"
    MAX_CONTENT_LENGTH = 5_000
    MAX_LINKS = 3


class Chatbot:

    def __init__(self, system_message):
        self.system_message = system_message

    @staticmethod
    def chat(message, history):

        relevant_system_message = system_message

        # multi-shot prompting 
        if "belt" in message:
            relevant_system_message += " The store does not sell belts; if you are asked for belts, be sure to point out other items on sale."
        
        
        messages = (
            [{"role": "system", "content": system_message}]
            + history
            + [{"role": "user", "content": message}]
        )

        stream = openai.chat.completions.create(
            model=Config.DEFAULT_MODEL, messages=messages, stream=True
        )

        response = ""
        for chunk in stream:
            response += chunk.choices[0].delta.content or ""
            yield response

if __name__ == "__main__":

    openai = OpenAI(base_url=Config.OLLAMA_BASE_URL, api_key=Config.OLLAMA_API_KEY)

    system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage \
    the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. \
    For example, if the customer says 'I'm looking to buy a hat', \
    you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales event.'\
    Encourage the customer to buy hats if they are unsure what to get."

    gr.ChatInterface(fn=Chatbot(system_message).chat, type="messages").launch()
