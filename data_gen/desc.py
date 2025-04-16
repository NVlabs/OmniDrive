# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import pickle
import os
from openai import OpenAI
from prompt_utils import describe_expertv2
import numpy as np
from planning_utils import Traj_Generator
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
    data, lane_infos, traj_gen, output_dir, api_key, sys_prompt = args
    
    client = OpenAI(base_url="https://api.close2openai.com/v1/", api_key=api_key)
    
    output_file_path = osp.join(output_dir, data['token'] + ".json")
    os.makedirs(osp.dirname(output_file_path), exist_ok=True)
    
    if not osp.isfile(output_file_path):
        if 'lane_info' in data.keys():
            lane_info = lane_infos[data['lane_info']]
            lane_pts = [lane['points'] for lane in lane_info['annotation']['lane_centerline']]
            traj, mask = data['gt_planning'][0], data['gt_planning_mask'][0]
            gt_fut_traj, gt_fut_traj_mask = data['gt_fut_traj'], data['gt_fut_traj_mask']
            planning_trajs, full_paths = traj_gen.generate_traj(lane_pts)
            expert_info = describe_expertv2(traj, mask, lane_pts, full_paths, gt_fut_traj, gt_fut_traj_mask, data['gt_fullnames'], data['gt_boxes'], data['gt_attrs'])

            user_prompt = f"""
Planning Info:
{expert_info}
                    """
        else:
            user_prompt = ""
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
                        temperature=0.7,
                        top_p=0.7,
                        max_tokens=2000,
                    )
                result = json.loads(replace_newlines_in_json_string(hat_completion.choices[0].message.content))
                print(result)
                with open(output_file_path, 'w') as f:
                    json.dump(result, f, indent=4)
            except Exception as e:
                print(e)
            else:
                break

def main(base_path, lane_info_path, info_file, output_dir, n_process, api_key_arg):
    api_key = api_key_arg
    
    key_infos = pickle.load(open(info_file, 'rb'))
    traj_gen = Traj_Generator()
    lane_infos = pickle.load(open(lane_info_path, 'rb'))
    data = key_infos['infos']

    sys_prompt = f"""Given two panoramic images that encapsulates the surroundings of a vehicle in a 360-degree view, your task is to analyze and interpret the current driving behavior and the associated driving scene.

Your task is divided into two parts:
1. Summarize the driving scenario in a paragraph.
- In this task, you should provide a detailed description of the driving scene. For example, specify the road condition, \
noting any particular settings (parking lot, intersection, roundabout), traffic elements (pedestrain, vehicle, traffic sign/light), time of the day and weather.

2. Analyze the driving action.
- The task is to use the given image to shortly explain the driving intentions, assuming you are driving in a real scene.
- You should understand the provided image, first identify the proper driving decision/intension. \
Then based on your background knowledge to reason what the driver should be particularly mindful of in this scenario and list them in the point form.
- Do not directly copy the provided planning infomation; instead, make the action description sound more natural.

In both tasks:
- Do not mention the "first/second image", ""front-left/rear-center view" respectively describes xxx. Instead, replace it with what is present at specific vehicle positions (front, back, left, right, etc.). \
Always answer as if you are directly in the driving scene.
- When describing the traffic elements, please specify their location or appearance characteristics to make them more distinguishable. \
Do not merely mention generic traffic rules; integrate the information from the image.
- Each panoramic image is a composite of three smaller images. The first image depicts scenes to the left-front, directly in front, and right-front of the vehicle. \
The second image, displays views of the left-rear, directly behind, and right-rear of the same vehicle.
- Answer based only on the content determined in the image, and do not speculate on uncertain content.

You should refer to the following example and format the results like {{"description": "xxx", "action": "xxx"}}:

  {{
    "description": "The scene captures a moment of urban life framed by a red traffic light in mid-transition. To the right, a pedestrian crossing, marked by bright white zebra stripes, lies momentarily empty, waiting for the signal to change. \
Directly ahead, a lineup of vehicles—a mix of sedans, a motorcycle, and a delivery van—pauses obediently at the red light, their headlights beginning to flicker on against the dimming light. \
On the left, the sidewalk bustles with people of all ages, indicating a neighborhood that thrives on its mix of residential and commercial energy. \
Behind this foreground of orderly traffic and pedestrian movement, the cityscape reveals a patchwork of modern and older buildings. A parked truck is behind us, with a construction worker standing beside it.",
    "action": "In this scenario, the vehicle should move slowly and make a right lane change. 
- The decision to change lanes is influenced by the need to overtake the stop bus in front of the vehicle. 
- There are no traffic behind the vehicle and ensure a gap large enough for a safe lane change. 
- Pedestrians are visible on the sidewalk to the right, it is necessary to observe their movements when changing lanes."
  }},
"""
    # Prepare the arguments for each task
    tasks = [(d, lane_infos, traj_gen, output_dir, api_key, sys_prompt) for d in data]

    # Call track_parallel_progress
    mmengine.track_parallel_progress(
        func=preprocess_single, 
        tasks=tasks, 
        nproc=n_process, 
        keep_order=True,  # Results will be in the order tasks were given
    )
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NuScenes data.")
    parser.add_argument('--base_path', type=str, default='data/nuscenes/', help='Base path to the NuScenes data.')
    parser.add_argument('--lane_info_path', type=str, default='data/nuscenes/data_dict_sample.pkl', help='Path to the lane info pickle file.')
    parser.add_argument('--info_file', type=str, default='data/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl', help='Path to the info file (e.g., nuscenes2d_ego_temporal_infos_train.pkl).')
    parser.add_argument('--output_dir', type=str, default='./desc/train/', help='Directory to save the output JSON files.')
    parser.add_argument('--n_process', type=int, default=8, help='Number of processes to use.')
    parser.add_argument('--api_key', type=str, required=True, help='API key for OpenAI.')


    args = parser.parse_args()
    main(args.base_path, args.lane_info_path, args.info_file, args.output_dir, args.n_process, args.api_key)
