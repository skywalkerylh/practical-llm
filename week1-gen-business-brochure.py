import requests
import json
from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
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
    """
    A utility class to represent a Website that we have scraped, now with links
    """

    def __init__(self, url):
        self.url = url
        response = requests.get(url, headers=Config.HEADERS)
        self.body = response.content
        soup = BeautifulSoup(self.body, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        links = [link.get("href") for link in soup.find_all("a")]
        self.links = [link for link in links if link]

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"


# First step: Have model figure out which links are relevant
class Link:
    def __init__(self, client):
        self.client =  client

    @staticmethod
    def get_links_system_prompt():
        link_system_prompt = "You are provided with a list of links found on a webpage. \
        You are able to decide which of the links would be most relevant to include in a brochure about the company, \
        such as links to an About page, or a Company page, or Careers/Jobs pages.\n"
        link_system_prompt += "You should respond in JSON as in this example:"
        link_system_prompt += """
        {
            "links": [
                {"type": "about page", "url": "https://full.url/goes/here/about"},
                {"type": "careers page": "url": "https://another.full.url/careers"}
            ]
        }
        """
        return link_system_prompt

    @staticmethod
    def get_links_user_prompt(website):
        user_prompt = f"Here is the list of links on the website of {website.url} - "
        user_prompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. \
    Do not include Terms of Service, Privacy, email links.\n"
        user_prompt += "Links (some might be relative links):\n"
        user_prompt += "\n".join(website.links)
        return user_prompt

    def get_links(self,url):
        website = Website(url)

        response = self.client.chat.completions.create(
            model=Config.DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": self.get_links_system_prompt()},
                {"role": "user", "content": self.get_links_user_prompt(website)},
            ],
            response_format={"type": "json_object"},
        )
        result = response.choices[0].message.content

        return json.loads(result)

    def get_all_details(self, url):
        result = "Landing page:\n"
        result += Website(url).get_contents()
        links = self.get_links(url)
        print("Found links:", links)
        i = 0
        for link in links["links"]:
            result += f"\n\n{link['type']}\n"
            result += Website(link["url"]).get_contents()
            if i > Config.MAX_LINKS:
                break
        return result


## Second step: make the brochure!

class Brochure:

    def __init__(self, client):
        self.client = client
        self.link = Link(client)

    @staticmethod
    def get_brochure_system_prompt():
        return "You are an assistant that analyzes the contents of several relevant pages from a company website \
                and creates a short humorous, entertaining, jokey brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
                Include details of company culture, customers and careers/jobs if you have the information."

    def get_brochure_user_prompt(self, company_name,url ):
        user_prompt = f"You are looking at a company called: {company_name}\n"
        user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
       
        user_prompt += self.link.get_all_details(url)
        user_prompt = user_prompt[:Config.MAX_CONTENT_LENGTH]  # Truncate if more than 5,000 characters
        return user_prompt

    def create_brochure(self, company_name, url):
        response = self.client.chat.completions.create(
            model=Config.DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": self.get_brochure_system_prompt()},
                {"role": "user", "content": self.get_brochure_user_prompt(company_name, url)}
            ],
        )
        result = response.choices[0].message.content
        # display(Markdown(result))
        return result

    def stream_brochure(self, company_name, url):
        stream = self.client.chat.completions.create(
            model=Config.DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": self.get_brochure_system_prompt()},
                {
                    "role": "user",
                    "content": self.get_brochure_user_prompt(company_name, url),
                },
            ],
            stream=True, # user can see the response as it is being generated
        )

        #Display the response using markdown
        # response = ""
        # display_handle = display(Markdown(""), display_id=True)
        # for chunk in stream:
        #     response += chunk.choices[0].delta.content or ''
        #     response = response.replace("```","").replace("markdown", "")
        #     update_display(Markdown(response), display_id=display_handle.display_id)

        result = ""
        for chunk in stream:
            result += chunk.choices[0].delta.content or ""
            yield result

def show_brochure(create_brochure):
    
    view = gr.Interface(
        fn=create_brochure,
        inputs=[
            gr.Textbox(label="Company name:"),
            gr.Textbox(label="Landing page URL including http:// or https://"),
            
        ],
        outputs=[gr.Markdown(label="Brochure:")],
        flagging_mode="never",
    )
    view.launch()

def main():
    client = OpenAI(
        base_url=Config.OLLAMA_BASE_URL, api_key=Config.OLLAMA_API_KEY
    )

    #result= Brochure(client).create_brochure("Nuli", "https://nuli.app")
    #print(result)

    # with ui 
    show_brochure(Brochure(client).stream_brochure)
    


__name__ == "__main__" and main()
