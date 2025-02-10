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


class Website:
    url: str
    title: str
    text: str

    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        self.body = response.content
        soup = BeautifulSoup(self.body, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"

class Model:
    @staticmethod
    def stream_ollama(prompt, system_message):

        openai = OpenAI(base_url=Config.OLLAMA_BASE_URL, api_key=Config.OLLAMA_API_KEY)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        stream = openai.chat.completions.create(
            model=Config.DEFAULT_MODEL,
            messages=messages,
            stream=True,
            temperature=0.2,  
            max_tokens=300,  
        )
        result = ""
        for chunk in stream:
            result += chunk.choices[0].delta.content or ""
            yield result

class Brochure:
    @staticmethod
    def get_system_prompt():
        return """You are an assistant that analyzes the contents of several relevant pages from a company website 
                    and creates a short humorous, entertaining, jokey brochure about the company for prospective customers, investors and recruits. Respond in markdown.
                    Must include details of company culture, customers and careers/jobs if you have the information.
                    """

    @staticmethod
    def get_user_prompt(company_name, url):
        user_prompt = f"""Please generate a company brochure for {company_name} in traditional chinese. 
                            Response in markdown and should short and concise. Here is their landing page:\n"""
        user_prompt += Website(url).get_contents()
        print(user_prompt)
        return user_prompt

    def stream_brochure(self, company_name, url):

        user_prompt = self.get_user_prompt(company_name, url)
        system_prompt = self.get_system_prompt()

        result = Model().stream_ollama(user_prompt, system_prompt)

        yield from result

    def show_brochure(self):
        view = gr.Interface(
        fn=self.stream_brochure,
        inputs=[
            gr.Textbox(label="Company name:"),
            gr.Textbox(label="Landing page URL including http:// or https://"),
        ],
        outputs=[gr.Markdown(label="Brochure:")],
        flagging_mode="never",
    )
        view.launch()

def main():
    brochure = Brochure()
    brochure.show_brochure()


if __name__ == "__main__":
    main()
