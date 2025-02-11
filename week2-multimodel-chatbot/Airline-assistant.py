import json
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


class PriceTool:

    def __init__(self):
        self.ticket_prices = {
            "london": "$799",
            "paris": "$899",
            "tokyo": "$1400",
            "berlin": "$499",
        }
    
    @staticmethod
    def price_function():
        '''There's a particular dictionary structure that's required to describe our function'''

        return {
            "name": "get_ticket_price",
            "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination_city": {
                        "type": "string",
                        "description": "The city that the customer wants to travel to",
                    },
                },
                "required": ["destination_city"],
                "additionalProperties": False,
            },
        }

    def get_ticket_price(self, destination_city):
        print(f"Tool get_ticket_price called for {destination_city}")
        city = destination_city.lower()
        return self.ticket_prices.get(city, "Unknown")

    def handle_tool_call(self, message):

        # parse the message to get the arguments
        tool_call = message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)

        # get city from the arguments
        city = arguments.get("destination_city")
        # get price of the ticket for the city
        price = self.get_ticket_price(city)

        # capsule the response in the required format
        response = {
            "role": "tool",
            "content": json.dumps({"destination_city": city, "price": price}),
            "tool_call_id": tool_call.id,
        }
        return response, city

class Chatbot:
    def __init__(self):
        self.openai = OpenAI(
            base_url=Config.OLLAMA_BASE_URL, api_key=Config.OLLAMA_API_KEY
        )
        # tools means the functions that the chatbot can call when llm doesn't know the answer
        self.tools =  [{"type": "function", "function": PriceTool.price_function()}]

        self.system_message = """You are a helpful assistant for an Airline called FlightAI. 
                                Give short, courteous answers, no more than 1 sentence. 
                                Always be accurate. If you don't know the answer, say so."""

    def chat(self, message, history):
        """
        Handles the chat interaction with the user.
        """
        messages = (
            [{"role": "system", "content": self.system_message}]
            + history
            + [{"role": "user", "content": message}]
        )
        response = self.openai.chat.completions.create(
            model=Config.DEFAULT_MODEL, messages=messages, tools=self.tools
        )

        # when model doesn't know the answer, it calls the tool
        if response.choices[0].finish_reason == "tool_calls":

            # unpack the message and call the tool
            message = response.choices[0].message

            # tool provide the response based on the message
            response, city = PriceTool().handle_tool_call(message)

            # add the response to the messages
            messages.append(message)
            messages.append(response)

            # call the model again with the updated messages and get the response
            response = self.openai.chat.completions.create(
                model=Config.DEFAULT_MODEL, messages=messages
            )

        return response.choices[0].message.content

def main():

    gr.ChatInterface(fn=Chatbot().chat, type="messages").launch()

if __name__ == "__main__":
    main()
