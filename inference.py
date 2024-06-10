from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import os
import json
from datetime import datetime
from dict_to_list import *

# Importing required libraries for model and processor
from transformers import AutoModelForCausalLM, AutoProcessor

# Initialize the FastAPI app
app = FastAPI()

# Define the paths and model ID
OUTPUT_FILE = "/media/aditya/Projects/Logo_Matching/journal_watch/Logomatchingdemo/LLaVA/user_logo.json"
FOLDER_PATH = "/media/aditya/Projects/Logo_Matching/journal_watch/Logomatchingdemo/LLaVA/user_logo/logos"
MODEL_ID = "microsoft/Phi-3-vision-128k-instruct"

# Load the model and processor
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda", trust_remote_code=True, torch_dtype="auto", cache_dir="/media/aditya/Projects/Logo_Matching/journal_watch/Logomatchingdemo/phi3/cache_phi")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir="/media/aditya/Projects/Logo_Matching/journal_watch/Logomatchingdemo/phi3/cache_phi")


def generate_description(img, output_file):
    image = Image.open(img).convert('RGB')


    messages = [ 
        {"role": "user", "content": "<|image_1|>\nExplain the above image, divide your explanation in four parts, the shape of the image, the texture of the image, the text of the images and the colours of the image."}, 
        {"role": "assistant", "content": ""} 
    ] 

    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda:0")

    generation_args = {
        "max_new_tokens": 500,
        "temperature": 0.0,
        "do_sample": False,
    }

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # date_str = datetime.now().strftime("%d/%m/%Y")
    date_str = "01/05/2024"

    data = {"Date": date_str, "filename": os.path.basename(img), "outputs": response}

    return data

@app.post("/process/")
async def process_images_endpoint():
    with open(OUTPUT_FILE, 'w') as file:
        for filename in os.listdir(FOLDER_PATH):
            if filename.lower().endswith((".jpg",".jpeg", ".png", ".jfif")):
                image_file = os.path.join(FOLDER_PATH, filename)
                data = generate_description(image_file, OUTPUT_FILE)
                json.dump(data, file)
                file.write('\n')
    
    write_json_file(OUTPUT_FILE)
    # Prepare a response
    response_content = {"message": "Images processed successfully."}
    return JSONResponse(content=response_content)

