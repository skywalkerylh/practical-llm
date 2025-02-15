import json
import gradio as gr
import base64
from openai import OpenAI
from io import BytesIO
from PIL import Image
from Utils.ImgGenerator import artist

class Config:
    BASE_URL = "http://localhost:11434/v1"
    API_KEY = "nokeyneeded"
    DEFAULT_MODEL = "llama3.1:8b"


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
            base_url=Config.BASE_URL, api_key=Config.API_KEY
        )
        # tools means the functions that the chatbot can call when llm doesn't know the answer
        self.tools =  [{"type": "function", "function": PriceTool.price_function()}]

        self.system_message = """You are a helpful assistant for an Airline called FlightAI. 
                                Give short, courteous answers, no more than 1 sentence. 
                                Always be accurate. If you don't know the answer, say so.
                                'Greetings from FlightAI! How can I help you today?' is the greeting message.
                                Notes and murmurs are not allowed. e.g. Note: This is a general greeting message. I don't know what actions to take."""

        

    def text_only(self, message, history):
        messages = [{"role": "system", "content": self.system_message}] + history + [{"role": "user", "content": message}]
        response = self.openai.chat.completions.create(
            model=Config.DEFAULT_MODEL, messages=messages, tools=self.tools
        )

        if response.choices[0].finish_reason=="tool_calls":
            message = response.choices[0].message
            response, city = PriceTool().handle_tool_call(message)
            messages.append(message)
            messages.append(response)
            response = self.openai.chat.completions.create(
                model=Config.DEFAULT_MODEL, messages=messages
            )

        return response.choices[0].message.content

    def text_and_img(self, history):
        """
        Handles the chat interaction with the user.
        """
        messages = (
            [{"role": "system", "content": self.system_message}]
            + history 
            
        )
        response = self.openai.chat.completions.create(
            model=Config.DEFAULT_MODEL, messages=messages, tools=self.tools
        )
        image = None 
       
        print(response.choices[0].finish_reason)
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

            # gen img
            image = artist(city)

        reply = response.choices[0].message.content
        history += [{"role":"assistant", "content":reply}]
        
        return history, image
        

class UI:
    def text_only():
        gr.ChatInterface(fn=Chatbot().text_only, type="messages").launch()
        
    def text_and_image():
       with gr.Blocks() as ui:
        with gr.Row():
            chatbot = gr.Chatbot(height=500, type="messages")
            image_output = gr.Image(height=500)
        with gr.Row():
            entry = gr.Textbox(label="Chat with our AI Assistant:")
        with gr.Row():
            clear = gr.Button("Clear")

        def do_entry(message, history):
            history += [{"role":"user", "content":message}]
            return "", history

        entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
            Chatbot().text_and_img, inputs=chatbot, outputs=[chatbot, image_output]
        )
        clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

        ui.launch(inbrowser=True)

def main():
    #UI.text_only()
    UI.text_and_image()


if __name__ == "__main__":
    main()
