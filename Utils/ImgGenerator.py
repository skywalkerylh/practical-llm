#!nvidia-smi
#!pip install diffusers transformers torch accelerate

import requests
import base64
from PIL import Image
import io
from diffusers import StableDiffusionPipeline
import torch

def artist(city):
    
    weight_path = "C:/Users/annie/Desktop/LLM/model/scenicroad.safetensors"

    print(weight_path)
    image_gen = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    )

    image_gen.load_lora_weights(weight_path, strict=False)  
    # start the model on GPU
    image_gen = image_gen.to("cuda")
    # if no gpu available, use cpu
    image_gen.enable_model_cpu_offload()

    
    text = f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant romantic style"
    negative_prompt = "blurry, bad quality, text, human, face"
    image = image_gen(prompt=text,
                    negative_prompt= negative_prompt,
                    width= 512,
                    height=512,
                    num_inference_step=20,
                    guidance_scale=8.5,
                    ).images[0]
    
    return image 



    
    
