import requests
from bs4 import BeautifulSoup
from openai import OpenAI

class Config:
    OLLAMA_BASE_URL = "http://localhost:11434/v1"
    OLLAMA_API_KEY = "ollama"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    }
    DEFAULT_MODEL = "llama3.2"

class Website:

    def __init__(self, url):

        """
        Create this Website object from the given url using the BeautifulSoup library
        """

        self.url = url
        response = requests.get(url, headers=Config.HEADERS)
        soup = BeautifulSoup(response.content, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)


class LLMSummarizer:
    def __init__(self):
        """Initialize OpenAI client with Ollama configuration."""
        self.client = OpenAI(
            base_url=Config.OLLAMA_BASE_URL, api_key=Config.OLLAMA_API_KEY
        )
        self.model = Config.DEFAULT_MODEL

    def _create_messages(self, website, system_prompt, user_prompt):

        user_prompt += website.text

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def display_summary(self, website: Website, system_prompt: str, user_prompt: str) -> str:

        messages = self._create_messages(website, system_prompt, user_prompt)
        response = self.client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content

def main():

    # Step 1: get the website content
    website = Website(
        "https://medium.com/@r41091113/%E5%9B%9E%E5%8F%B0%E7%81%A3%E6%89%BE%E5%B7%A5%E4%BD%9C%E5%9B%89-2022-%E7%B6%B2%E8%B7%AF%E7%94%A2%E6%A5%AD%E8%B3%87%E6%96%99%E7%A7%91%E5%AD%B8-%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90%E9%9D%A2%E8%A9%A6%E7%B6%93%E9%A9%97%E5%88%86%E4%BA%AB-1cd39b2b09b9"
    )
    print(f"Title: {website.title}")
    print(f"Text: {website.text[:200]}...")  # Print first 200 chars

    # Step 2: create the summarizer
    summarizer = LLMSummarizer()

    # Step 3: create the prompts
    system_prompt = (
            "Imagine you are a professional AI engineer, teaching people AI with "
            "simple analogy and examples. Always follow user's format requirements "
            "strictly. Never include references unless specifically requested. "
            "Respond in markdown."
        )

    user_prompt = (
        "This is the website that tells the trend of the LLM. "
        "Summarize what it talks about in the article. "
        "Don't contain reference in your summary."
    ) 

    # Step 4: get the summary
    summary = summarizer.display_summary(website, system_prompt, user_prompt)
    print("\nSummary:")
    print(summary)


__name__ == "__main__" and main()
