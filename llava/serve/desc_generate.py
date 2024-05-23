import sys
dict_to_list_path = '/media/aditya/Projects/Logo_Matching/journal_watch/Logomatchingdemo/LLaVA/'
sys.path.append(dict_to_list_path)

import argparse
import torch
import os
import json
from dict_to_list import *
from datetime import datetime
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args, tokenizer, model, image_processor, image_file):
    # Model
    disable_torch_init()

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image(image_file)
    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    inp = "Explain the above image, divide your explanation in four parts, the shape of the image, the texture of the image, the text of the images and the colours of the image. "
    
    if not inp:
        print("exit...")
        return {}

    print(f"{roles[1]}: ", end="")

    if image is not None:
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            top_p=args.top_p,
            use_cache=True)

    outputs = tokenizer.decode(output_ids[0]).strip()
    conv.messages[-1][-1] = outputs

    if args.debug:
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    
    data = {"filename": os.path.basename(image_file), "outputs": outputs}
    
    with open(args.output_file, 'a') as file:
        json.dump(data, file)
        file.write('\n')


def get_last_journal_number(file_path):
    try:
        with open(file_path, 'r+') as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-1].strip()
                last_journal_number = int(last_line)
                return last_journal_number
            else:
                raise ValueError("Journal file is empty.")
    except Exception as e:
        print(f"Error reading  journal file: {e}")
        raise

def update_last_journal_number(file_path):
    try:
        with open(file_path, 'r+') as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-1].strip()
                last_journal_number = int(last_line)
                new_journal_number = last_journal_number + 1
                file.seek(0, os.SEEK_END)
                file.write(f"{new_journal_number}\n")
            else:
                raise ValueError("Journal file is empty.")
    except Exception as e:
        print(f"Error updating journal file: {e}")
        raise

if __name__ == "__main__":

    journal_file_path = "journal.txt"
    journal_number = get_last_journal_number(journal_file_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/models--liuhaotian--llava-v1.6-34b/snapshots/e2a1f782a20d26b855072029738bfd0107d85e96")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str)
    parser.add_argument("--output-file", type=str, default=f"journal_no_{journal_number}.json")
    parser.add_argument("--folder_path", type=str, default=f"journal_no_{journal_number}")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--top_p", type=float, default=0.2)
    args = parser.parse_args()
    disable_torch_init()
    torch.manual_seed(10)

    model_name = args.model_path

    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    # Write the initial dictionary with the date
    date_str = datetime.now().strftime("%d/%m/%Y")
    initial_data = {"Date": date_str}
    with open(args.output_file, 'w') as file:
        json.dump(initial_data, file)
        file.write('\n')

    folder_path = args.folder_path
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jfif"):
            image_file = os.path.join(folder_path, filename)
            main(args, tokenizer, model, image_processor, image_file)
    
    write_json_file(f"journal_no_{journal_number}.json")
    update_last_journal_number(journal_file_path)
