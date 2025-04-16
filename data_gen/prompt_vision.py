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
import argparse
import pickle
import os
from openai import OpenAI
from prompt_utils import describe_expert, get_crosswalks, scene_description, describe_simulated
import numpy as np
from planning_utils import PlanningMetric, Traj_Generator
from nuscenes.eval.common.utils import Quaternion
import json
from os import path as osp
import mmengine
from PIL import Image, ImageOps

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

def make_context(area, side):
    prompt = f"""\
You are a certified professional driving instructor and currently demonstrating driving in a residential area of {area}.

- The traffic participant information is represented in the 2D top-down view under the corresponding lane/crosswalk, \
with attributes like class, orientation, position (x, y) in meters, and velocity (vx, vy) in meters per second. 

- Definitions for x and y directions: If x > 0, it means that the object is in front of your own car, and vice versa. \
If y > 0, it means that the object is to the left of your own car, and vice versa.

- The Lane centerline for vehicle navigation, is defined by a cubic Bezier curve's control points [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]. \
Lanes are separated into different instances if and only if in the cases of intersection, fork, and merge. \
The connection of two lanes means that the ending point of a lane is connected to the starting point of another lane. \
with-flow: A lane designated for vehicles that are traveling in the same direction as you. \
opposite-flow: A lane designated for vehicles that are traveling in the opposite direction as you. \
allowing from right to left driving: A lane that is perpendicular to your current direction, allowing left movement. \
allowing from left to right driving: A lane that is perpendicular to your current direction, allowing right movement.

- The crosswalk is represented by the four vertices of a rectangular, which encompasses the area of the pedestrian crossing.

- Traffic sign attributes are assigned to the corresponding lane.

- The description and action briefly describe the scene you are in.

- Your status is represented as "your own car" under the corresponding centerline.

- There is one expert trajectory and multiple example trajectories. The expert trajectory meets safety requirements, while the example trajectories might be dangerous. \
If a trajectory is considered dangerous, we will provide details on which traffic rules were violated and with which objects it may collide. Some trajectories are marked as relatively safe, \
but you still need to confirm again based on the scene information and description, because the danger information under the example trajectory is not complete.

Now design 4 QA pairs about the current driving scenario.
Ask diverse questions, paraphrase the following questions in a more natural way, and give corresponding answers.
Generate detailed answers and include numerical outputs (based on the current given input observations) as much as possible.

Q1: Are there traffic elements that may affect your driving behavior? If so, what are they?
Q2: What's your next action and why?

You should refer to the expert/example trajectories and design 2 QA pairs of this kind:
Q3: If you follow the trajectory [PT, (x1, y1), (x2, y2), (x3, y3)] [replace here with example trajectory], what would happen?

You should obey the following rules when answering questions:
- In {area}, You should drive on the {side[area]} side of the road according to the {area} driving rules.
- By analyzing the differences between the expert and the example trajectories, you can determine specific factors that contribute to safety or risk.
- Do not mention proprietary terms like 'expert trajectory/decision', 'example trajectory/decision'. Always answer as if you are directly in the driving scene.
- Pay attention to the subordinate relationship between objects and lanes/crosswalks, and list them out. For example, a truck traveling ahead is driving on a left turning lane.
- When answering the questions, please rephrase the attributes and categories of objects to make the expression more natural. For example, "movable_object.barriers" -> "barriers", "vehicle.truck.moving" -> "moving truck".
- When answering the questions, add numerical information based on the given scene description and action (e.g., objects near your path, you may collide with, traffic signs and signals, etc.), always with their locations, states and belonging lanes if possible.
- When answering Q1, avoid overreporting the threatened traffic elements. For example, objects (stationary or not moving in the same direction) behind you, objects in the distance (having large location values and don't hinder your path), even in the same lane or crosswalk, usually pose no threat. \
You could say "In this scenario, only xxx is far behind the vehicle/there are no objects around us, which won't affect the driving behavior."
- Always replace the trajectory with the format [PT, (x1, y1), (x2, y2), (x3, y3)] in the Q3 (question part).

You should refer to the following example and format each QA pair like {{"question": "xxx", "answer": "xxx"}}:
[
  {{
    "question": "Are there traffic elements that may affect your driving behavior?",
    "answer": "In this scenario, ..."
  }},
  {{
    "question": "If you follow the trajectory [PT, (x1, y1), (x2, y2), (x3, y3)], what would happen?",
    "answer": "This action is safe as ..."
  }},
  {{
    "question": "What's your next action and why?",
    "answer": "Given the current scenario, the next action could be ..."
  }},
]
"""
    return prompt

def process_single(tasks):
    data, lane_infos, planning_metric, output_dir, traj_gen, side, desc_path = tasks
    client = OpenAI(base_url="https://api.close2openai.com/v1/", api_key=args.api_key)  # Initialize client here
    output_file_path = osp.join(output_dir, data['token'] + ".json")
    os.makedirs(osp.dirname(output_file_path), exist_ok=True)
    if 'lane_info' in data.keys() and not osp.isfile(f'{output_dir}/{data["token"]}.json'):
        lane_info = lane_infos[data['lane_info']]
        lane_pts = [lane['points'] for lane in lane_info['annotation']['lane_centerline']]
        traj, mask = data['gt_planning'][0], data['gt_planning_mask'][0]

        with open(f"{desc_path}/{data['token']}.json", 'r') as f:
            scene_keywords = json.load(f)
        
        scene_keywords = f"""
Description:
{scene_keywords["description"]}

Action:
{scene_keywords["action"]}        
        """
        
        gt_fut_traj, gt_fut_traj_mask = data['gt_fut_traj'], data['gt_fut_traj_mask']
        crosswalks = get_crosswalks(data['map_geoms'])
        planning_trajs, full_paths = traj_gen.generate_traj(lane_pts)
        scene_info, lanes_red = scene_description(traj, mask, lane_info, data['gt_fullnames'], data['gt_boxes'], data['gt_velocity'], data['gt_attrs'], lane_pts, crosswalks)
        expert_info = describe_expert(traj, mask, lane_pts, full_paths, gt_fut_traj, gt_fut_traj_mask, data['gt_fullnames'], data['gt_boxes'], data['gt_attrs'])

        ego_boxes = np.array([[1.5, 0.0, 0.0, 4.08, 1.73, 0.0, 0.0, 0.0, 0.0]])
        step = 6

        light_seg = planning_metric.red_light_area(lanes_red)

        gt_agent_boxes = np.concatenate([data['gt_boxes'], data['gt_velocity']], -1)
        gt_agent_feats = np.concatenate([data['gt_fut_traj'][:, :6].reshape(-1, 12), data['gt_fut_traj_mask'][:, :6], data['gt_fut_yaw'][:, :6], data['gt_fut_idx']], -1)
        bev_seg = planning_metric.get_birds_eye_view_label(gt_agent_boxes, gt_agent_feats)

        e2g_r_mat = Quaternion(data['ego2global_rotation']).rotation_matrix
        e2g_t = data['ego2global_translation']
        drivable_seg = planning_metric.get_drivable_area(e2g_t, e2g_r_mat, data)

        all_coll_objs = []
        all_red_lights = []
        all_drivable = []
        for traj in planning_trajs:
            ego_seg = planning_metric.get_ego_seg(ego_boxes, traj, add_rec=True)
            coll_index, red_light, out_of_drivable = planning_metric.traj_check(ego_seg, bev_seg, light_seg, drivable_seg)
            all_red_lights.append(red_light)
            all_drivable.append(out_of_drivable)
            coll_obj = [(data['gt_fullnames'][idx], data['gt_attrs'][idx], data['gt_boxes'][idx]) for idx in coll_index]
            all_coll_objs.append(coll_obj)

        simulated_info = describe_simulated(step, planning_trajs, lane_pts, all_coll_objs, all_red_lights, all_drivable, full_paths)

        area = data['location'].split("-")[0]
        sys_prompt = make_context(area, side)
        user_prompt = f"""
Scene Info:
{scene_info}

{scene_keywords}

Planning Info:
{expert_info}

Simulated Info:
{simulated_info}
        """
        
        while True:
            try:
                hat_completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
                    ],
                    temperature=0.7,
                    top_p=0.7,
                    max_tokens=2000,
                )

                result = json.loads(hat_completion.choices[0].message.content)
                with open(f"{output_dir}/{data['token']}.json", 'w') as f:
                    json.dump(result, f, indent=4)
            except Exception as e:
                print(e)
            else:
                break

def main(args):
    # Load data
    key_infos = pickle.load(open(os.path.join(args.base_path, args.nuscenes_info_file), 'rb'))
    lane_infos = pickle.load(open(os.path.join(args.base_path, args.data_dict_file), 'rb'))

    planning_metric = PlanningMetric(args.base_path)
    traj_gen = Traj_Generator()

    side = {
        'singapore': 'left',
        'boston': 'right',
    }

    data = key_infos['infos']

    # Prepare the arguments for each task
    tasks = [(d, lane_infos, planning_metric, args.output_dir, traj_gen, side, args.desc_path) for d in data]
    
    # Call track_parallel_progress
    mmengine.track_parallel_progress(
        func=process_single, 
        tasks=tasks, 
        nproc=args.n_process, 
        keep_order=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process driving scenarios and generate QA pairs.")
    parser.add_argument('--base_path', type=str, default='./data/nuscenes/', help='Base path to the data directory.')
    parser.add_argument('--nuscenes_info_file', type=str, default='nuscenes2d_ego_temporal_infos_train.pkl', help='Nuscenes info file.')
    parser.add_argument('--data_dict_file', type=str, default='data_dict_sample.pkl', help='Data dictionary file.')
    parser.add_argument('--output_dir', type=str, default='./vqa/train', help='Output directory for results.')
    parser.add_argument('--desc_path', type=str, default='./desc/train/', help='Path to the description files directory.')
    parser.add_argument('--api_key', type=str, required=True, help='API key for OpenAI.')
    parser.add_argument('--n_process', type=int, default=8, help='Number of parallel processes to use.')

    args = parser.parse_args()
    main(args)
