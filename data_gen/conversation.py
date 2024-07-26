import pickle
import os
from openai import OpenAI
import numpy as np
import mmengine
import json
from os import path as osp
from PIL import Image, ImageOps
import base64
from io import BytesIO
import re
import argparse

def replace_newlines_in_json_string(s):
    pattern = re.compile(r'\"(.*?)\"', re.DOTALL)
    
    def replace_newline(match):
        return match.group(0).replace('\n', '\\n')
    
    replaced_string = re.sub(pattern, replace_newline, s)
    return replaced_string

def preprocess_images(image_paths_or_images, layout='horizontal', flip=False):
    images = [Image.open(x) if isinstance(x, str) else x for x in image_paths_or_images]

    if flip:
        images = [ImageOps.mirror(image) for image in images]

    if layout == 'horizontal':
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
    elif layout == 'vertical':
        widths, heights = zip(*(i.size for i in images))
        max_width = max(widths)
        total_height = sum(heights)
        new_im = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]
    return new_im

def create_combined_image(front_image_paths, back_image_paths):
    front_image = preprocess_images(front_image_paths)
    back_image = preprocess_images(back_image_paths, flip=True)
    return front_image, back_image

def encode_image(image):
    with BytesIO() as buffer:
        image.save(buffer, format='JPEG')  
        return base64.b64encode(buffer.getvalue()).decode('utf-8') 

def preprocess_single(args):
    data, output_dir, desc_path, api_key, sys_prompt = args

    client = OpenAI(base_url="https://api.close2openai.com/v1/", api_key=api_key)
    output_file_path = osp.join(output_dir, data['token'] + ".json")
    os.makedirs(osp.dirname(output_file_path), exist_ok=True)
    if not osp.isfile(osp.join(output_dir, data['token']+'.json')):
        with open(osp.join(desc_path, data['token'] + ".json"), 'r') as f:
            scene_keywords = json.load(f)
        user_prompt = f"""
Description:
{scene_keywords["description"]}

Action:
{scene_keywords["action"]}        
        """

        front_image_paths = [data['cams']['CAM_FRONT_LEFT']['data_path'], data['cams']['CAM_FRONT']['data_path'], data['cams']['CAM_FRONT_RIGHT']['data_path']]
        back_image_paths = [data['cams']['CAM_BACK_LEFT']['data_path'], data['cams']['CAM_BACK']['data_path'], data['cams']['CAM_BACK_RIGHT']['data_path']]

        front_image, back_image = create_combined_image(front_image_paths, back_image_paths)
        
        front_image = front_image.resize((1536, 512))
        back_image = back_image.resize((1536, 512))

        encoded_front_image = encode_image(front_image)
        encoded_back_image = encode_image(back_image)
        
        while True:
            try:
                hat_completion = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_front_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "The first image depicts scenes to the left-front, directly in front, and right-front of the vehicle.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_back_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "The second image, displays views of the left-rear, directly behind, and right-rear of the vehicle.",
                        },
                        ]},
                        ],
                        temperature=0.9,
                        top_p=0.7,
                        max_tokens=2000,
                    )
                result = json.loads(replace_newlines_in_json_string(hat_completion.choices[0].message.content))
                with open(osp.join(output_dir, data['token']+'.json'), 'w') as f:
                    json.dump(result, f, indent=4)
            except Exception as e:
                print(e)
                continue
            else:
                break

def main(info_file, desc_path, output_dir, n_process, api_key_arg):
    api_key = api_key_arg

    key_infos = pickle.load(open(info_file, 'rb'))
    data = key_infos['infos']

    sys_prompt = f"""Given two panoramic images that encapsulates the surroundings of a vehicle in a 360-degree view, your task is to analyze and interpret the current driving behavior and the associated driving scene.

You should design a conversation between you and a person asking about this driving scenario.
Ask diverse questions and give corresponding answers.

Include questions asking about the visual content of the image, including the particular settings (parking lot, intersection with/without traffic light, roundabout), \
object types, counting the objects, object actions, relative positions between objects, etc. Only include questions that have definite answers:
(1) one can see the content in the images that the question asks about and can answer confidently;
(2) one can determine confidently from the images that it is not in the images.
Do not ask any question that cannot be answered confidently.

Also include complex questions that are relevant to the content in the images, for example, asking about background knowledge of the objects in the scenario, asking to discuss about events happening in the scenario.
Provide detailed answers when answering complex questions. For example, give detailed examples or reasoning steps to make the content more convincing and well-organized.

In all the sentences:
- Do not mention "the first/second image", ""the front-left/rear-center view" describes xxx. Instead, replace it with what is present at specific vehicle positions (front, back, left, right, etc.). \
Always answer as if you are directly in the driving scene.
- When describing the traffic elements, please specify their location or appearance characteristics to make them more distinguishable.
- Each panoramic image is a composite of three smaller images. The first image depicts scenes to the left-front, directly in front, and right-front of the vehicle. \
The second image, displays views of the left-rear, directly behind, and right-rear of the same vehicle.

You should refer to the following example and format each QA pair like {{"question": "xxx", "answer": "xxx"}}:
[
  {{
    "question": "What is the current status of the traffic light?",
    "answer": "Currently, the traffic light is red for vehicles, indicating a need to stop."
  }},
  {{
    "question": "Where is the pedestrian who is wearing a blue coat?",
    "answer": "He is crossing the zebra from the right to the left."
  }},
  {{
    "question": "What are the characteristics of the road you are on?",
    "answer": "It's a two-lane road, one lane for each direction that weaves left."
  }},
  {{
    "question": "Can you describe the vehicle directly in front of you, including any distinctive features or decals it might have?",
    "answer": "The vehicle in front is a silver SUV with tinted windows and a ski rack on top. It has a bumper sticker that reads “Coexist” and a small dent on the left side of the rear bumper, possibly from a previous minor collision."
  }},
  {{
    "question": "Considering the vehicle ahead and its features, can you infer anything about its owner or the type of person who might drive such a vehicle?",
    "answer": "Given the vehicle's practical yet luxurious make, combined with lifestyle indicators like a ski rack, and a bumper sticker advocating for harmony, the owner likely enjoys outdoor activities and values messages of peace and coexistence. This might suggest they are adventurous yet socially and environmentally conscious, possibly reflecting a preference for vehicles that offer both comfort and utility without compromising on personal values."
  }}
]
"""

    # Prepare the arguments for each task
    tasks = [(d, output_dir, desc_path, api_key, sys_prompt) for d in data]

    # Call track_parallel_progress
    mmengine.track_parallel_progress(
        func=preprocess_single, 
        tasks=tasks, 
        nproc=n_process, 
        keep_order=False,  # Results do not need to be in the order tasks were given
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NuScenes data.")
    parser.add_argument('--info_file', type=str, default='data/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl', help='Path to the info file (e.g., nuscenes2d_ego_temporal_infos_train.pkl).')
    parser.add_argument('--desc_path', type=str, default='./desc/train/', help='Path to the description files directory.')
    parser.add_argument('--output_dir', type=str, default='./conv/train/', help='Directory to save the output JSON files.')
    parser.add_argument('--n_process', type=int, default=8, help='Number of processes to use.')
    parser.add_argument('--api_key', type=str, required=True, help='API key for OpenAI.')

    args = parser.parse_args()
    main(args.info_file, args.desc_path, args.output_dir, args.n_process, args.api_key)
