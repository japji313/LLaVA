import os
import json
import torch
from fastapi import FastAPI
import requests
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from datetime import datetime
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

app = FastAPI()

# Global variables for the model, tokenizer, and image processor
model = None
tokenizer = None
image_processor = None

# Paths
MODEL_PATH = "/media/aditya/Projects/Logo_Matching/journal_watch/try/LLaVA/models/models--liuhaotian--llava-v1.6-34b/snapshots/e2a1f782a20d26b855072029738bfd0107d85e96"
CACHE_DIR = "/media/aditya/Projects/Logo_Matching/journal_watch/try/LLaVA/models"
OUTPUT_FILE = "user_logo.json"

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def process_image(image_file):
    image = load_image(image_file)
    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    return image_tensor, image_size

def generate_prompt(conv_mode, roles, image, inp):
    conv = conv_templates[conv_mode].copy()
    if image is not None:
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(roles[0], inp)
    else:
        conv.append_message(roles[0], inp)
    conv.append_message(roles[1], None)
    return conv

def infer_conv_mode(model_path):
    if "llama-2" in model_path.lower():
        return "llava_llama_2"
    elif "mistral" in model_path.lower():
        return "mistral_instruct"
    elif "v1.6-34b" in model_path.lower():
        return "chatml_direct"
    elif "v1" in model_path.lower():
        return "llava_v1"
    elif "mpt" in model_path.lower():
        return "mpt"
    else:
        return "llava_v0"

def save_results(output_file, data):
    with open(output_file, 'a') as file:
        json.dump(data, file)
        file.write('\n')

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, image_processor
    disable_torch_init()
    torch.manual_seed(10)
    
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    model_cache_path = os.path.join(CACHE_DIR, os.path.basename(MODEL_PATH) + ".pth")
    
    if os.path.exists(model_cache_path):
        print(f"Loading model from cache: {model_cache_path}")
        model_state_dict = torch.load(model_cache_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=MODEL_PATH, model_base=None, model_name=MODEL_PATH,
            load_8bit=False, load_4bit=True, device='cuda'
        )
        model.load_state_dict(model_state_dict)
    else:
        print(f"Downloading and caching model to: {model_cache_path}")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=MODEL_PATH, model_base=None, model_name=MODEL_PATH,
            load_8bit=False, load_4bit=True, device='cuda'
        )
        torch.save(model.state_dict(), model_cache_path)

@app.post("/process/")
async def process_image_endpoint():
    folder_path = "user_logo"

    with open(OUTPUT_FILE, 'w') as file:
        pass
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".jfif")):
            image_file = os.path.join(folder_path, filename)
            image_tensor, image_size = process_image(image_file)
            inp = "Explain the above image, divide your explanation in four parts: shape, texture, text, and colors."
            conv_mode = infer_conv_mode(MODEL_PATH)
            roles = ('user', 'assistant') if "mpt" in MODEL_PATH.lower() else conv_templates[conv_mode].roles
            conv = generate_prompt(conv_mode, roles, image_tensor, inp)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=512,
                    streamer=streamer,
                    top_p=0.2,
                    use_cache=True
                )

            outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            conv.messages[-1][-1] = outputs

            # date_str = datetime.now().strftime("%d/%m/%Y")
            date_str = "01/05/2024"
            data = {"Date": date_str, "filename": os.path.basename(image_file), "outputs": outputs}
            save_results(OUTPUT_FILE, data)
    
    with open(OUTPUT_FILE, "r") as file:
        data = [json.loads(line) for line in file]
        return data[0]


# import subprocess
# import json
# import os
# from fastapi import FastAPI, File, UploadFile
# from shutil import copyfileobj

# import sys
# from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parent.parent))

# app = FastAPI()


# def extract_text_from_json_file(json_file_path):
#     with open(json_file_path, 'r') as file:
#         json_data = json.load(file)

#     if isinstance(json_data, dict):
#         filename = json_data.get("filename")
#         text = json_data.get("outputs")

#         if filename is not None and text is not None:
#             keywords1 = ["Shape of the Image", "Texture of the Image", "Text of the Image", "Colors of the Image"]
#             keywords2 = ["Shape:", "Texture:", "Text:", "Colors:"]

#             def extract_text(text, keywords):
#                 split_text = text.split("\n\n")
#                 formatted_text = {}
#                 current_category = None
#                 for part in split_text:
#                     for keyword in keywords:
#                         if keyword in part:
#                             current_category = keyword
#                             formatted_text[current_category] = []
#                             break
#                     if current_category:
#                         if current_category == "Shape of the Image" and "Shape of the Image" not in part:
#                             formatted_text[current_category].extend(part.split())
#                         else:
#                             formatted_text[current_category].append(part)
#                 return formatted_text

#             formatted_text1 = extract_text(text, keywords1)
#             formatted_text2 = extract_text(text, keywords2)

#             # If output from keywords1 is not generated, use the formatted text from keywords2
#             if not formatted_text1:
#                 formatted_text = formatted_text2
#             else:
#                 formatted_text = formatted_text1

#             # Write the extracted data to a new JSON file
#             extracted_data = {"filename": filename, "extracted_text": formatted_text}
#             with open("user_logo.json", 'w') as outfile:
#                 json.dump(extracted_data, outfile, indent=4)


# @app.post("/process/")
# async def process_image():
#     model_path = "/media/aditya/Projects/Logo_Matching/journal_watch/LLaVA/models/models--liuhaotian--llava-v1.6-34b/snapshots/e2a1f782a20d26b855072029738bfd0107d85e96"
#     folder_path = "user_logo"
#     os.environ['LAVA_MODEL'] = model_path
#     subprocess.run(["python", "-m", "llava.serve.desc_generate_res", "--model-path", model_path, "--folder_path", folder_path, "--load-4bit"])
        
#     with open("user_logo.json", "r") as file:
#         data = json.load(file)
#         #for llama index appraoch
#         if data:
#             with open('user_logo.json', 'w') as file:
#                 pass  
#         return data
    
#         #for simple embedding approach

#         # if data:
#         #     extract_text_from_json_file("user_logo.json")
#         #     with open("user_logo.json", "r") as refactored_file:
#         #         refactored_data = json.load(refactored_file)
#         #     os.remove("user_logo.json")
#         # return refactored_data



