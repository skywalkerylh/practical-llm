#!pip install -q requests torch bitsandbytes transformers sentencepiece accelerate openai httpx==0.27.2
import os
import requests
from IPython.display import Markdown, display, update_display
from openai import OpenAI
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig,AutoModelForSpeechSeq2Seq , pipeline, AutoProcessor
import torch
from Config import HF_TOKEN

# Constants
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
audio_filename = (
    "/content/drive/MyDrive/LLM/Data/How I use AI to save 10 hours per week.mp3"
)
login(HF_TOKEN, add_to_git_credential=True)

# Sign in to OpenAI using Secrets in Colab

AUDIO_MODEL = "openai/whisper-medium"
speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(AUDIO_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)
speech_model.to('cuda')
processor = AutoProcessor.from_pretrained(AUDIO_MODEL)

pipe = pipeline(
    "automatic-speech-recognition",
    model=speech_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device='cuda',
)
# Use the Whisper OpenAI model to convert the Audio to Text
result = pipe(audio_filename,
              return_timestamps= True)
transcription = result["text"]
print(transcription)


# text summary
system_message = "You are an assistant that produces minutes of meetings from transcripts, with summary, key discussion points, takeaways and action items with owners, in markdown."
user_prompt = f"Below is an extract transcript of a Denver council meeting. Please write minutes in markdown, including a summary with attendees, location and date; discussion points; takeaways; and action items with owners.\n{transcription}"

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
  ]

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto", quantization_config=quant_config)
outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)

response = tokenizer.decode(outputs[0])

display(Markdown(response))
