!ollama pull llama3.2
import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI

# Some websites need you to use proper headers when fetching them:
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

openai = OpenAI(
        base_url="http://localhost:11434/v1",  # Ollama 的本地服務器地址
        api_key="ollama",  # 這裡的 "ollama" 只是一個佔位符,因為本地運行不需要真實的 API key

    )

class Website:

    def __init__(self, url):

        """
        Create this Website object from the given url using the BeautifulSoup library
        """

        self.url = url
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)


# Step1: set user prompt and system prompt
def user_prompt_for(website):

    user_prompt = "This is the website that tells the trend of the llm. \
    summarize what he talks about in the article.\
    don't contain reference in your summary."
    user_prompt += website.text
    return user_prompt

def display_summary(messages):

    response = openai.chat.completions.create(model="llama3.2", messages=messages)
    return response.choices[0].message.content


def main():
    # set the OpenAI
    

    system_prompt = "imagine you are a professional ai engineer, you are teaching poeple ai with simple analogy and examples.Always follow user's format requirements strictly. Never include references unless specifically requested. Respond in markdown."

    # Step 2: get the website content
    website = Website(
        "https://medium.com/@cch.chichieh/llm-%E8%A9%95%E4%BC%B0%E6%96%B9%E6%B3%95%E6%8C%87%E5%8D%97-%E8%B6%A8%E5%8B%A2-%E6%8C%87%E6%A8%99%E8%88%87%E6%9C%AA%E4%BE%86%E6%96%B9%E5%90%91-e81616d30e53"
    )
    print("title", website.title)
    print("text", website.text)

    # Step 3: set the messages
    messages =[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_for(website)},
        ]

    # Step 3: Call OpenAI

    summary = display_summary(messages)
    print(summary)

__name__ == "__main__" and main()