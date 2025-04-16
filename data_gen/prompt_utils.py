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
import numpy as np
from math import factorial
import math
import cv2
import torch

te_category =  ['traffic_light', 'road_sign']

te_attribute = ['unknown', 'red','green','yellow','go_straight','turn_left', 'turn_right','no_left_turn','no_right_turn', \
    'u_turn','no_u_turn','slight_left','slight_right']

def format_number(n, decimal_places=1):
    if abs(round(n, decimal_places)) <= 1e-2:
         return 0.0
    else:
        format_string = f"{{n:+.{decimal_places}f}}"
        return format_string.format(n=n)

def point_in_rotated_rect(bbox, rect):
    
    rect = np.array(rect)
    point = np.array(bbox[:2])

    
    edge1 = rect[1] - rect[0]  
    edge2 = rect[3] - rect[0]  

    
    axis1 = edge1 / np.linalg.norm(edge1)
    axis2 = edge2 / np.linalg.norm(edge2)

    
    point_vector = point - rect[0]

    
    proj_on_axis1 = point_vector.dot(axis1)
    proj_on_axis2 = point_vector.dot(axis2)
    
    
    in_rect = (
        0 <= proj_on_axis1 <= np.linalg.norm(edge1) and
        0 <= proj_on_axis2 <= np.linalg.norm(edge2)
    )
    
    return in_rect

def calculate_speed(traj, mask):
    traj_roll = np.zeros_like(traj)
    traj_roll[1:, :] = traj[:-1, :]
    velos = (traj - traj_roll) / 0.5
    return velos[mask]

def judge_speed_changes(speeds, accel_threshold=0.2, accel_absolute=5, decel_threshold=0.2, decel_absolute=5):
    if len(speeds) == 0:
        return "Unknown"
    if all(abs(speed) <= 5e-2 for speed in speeds):
        return "Stopped"  
    
    initial_speed = speeds[0]
    if initial_speed < 1.5:
        speed_state = "Crawling"
    elif initial_speed < 5.5:
        speed_state = "Moving Slowly"
    elif initial_speed < 14:
        speed_state = "Moderate Speed"
    else:
        speed_state = "Moving Fastly"

    current_state = "" 
    closest_change_distance = float('inf')  

    for i, speed in enumerate(speeds[1:], start=1):
        change = speed - initial_speed
        percent_change = abs(change) / (initial_speed if initial_speed != 0 else 1)

        
        if abs(initial_speed) < 0.1 and abs(speed) > 0.2:
            current_state = "Vehicle Starting"  
            break  
        
        
        elif change > 0 and ((percent_change > accel_threshold and change > 2.0) or change > accel_absolute):
            
            if i < closest_change_distance:
                closest_change_distance = i
                current_state = "Accelerate"

        
        elif change < 0 and ((percent_change > decel_threshold and abs(change) > 2.0) or abs(change) > decel_absolute):
            
            if i < closest_change_distance:
                closest_change_distance = i
                current_state = "Decelerate"

        initial_speed = speed  
        
    if current_state == "":
        return speed_state
    else:
        return speed_state + ", " + current_state

def determine_turning_behavior(steering_angles):
    steering_angles = (steering_angles + np.pi) % (2 * np.pi) - np.pi
    LEFT_TURN_THRESHOLD = 20
    RIGHT_TURN_THRESHOLD = -20
    LEFT_UTURN_THRESHOLD = 150
    RIGHT_UTURN_THRESHOLD = -150
    current_behavior = 'Go Straight'

    # 遍历转向角列表
    for index, angle in enumerate(steering_angles):
        angle = angle * (180 / np.pi)
        # 检查U-turn条件
        if angle > LEFT_UTURN_THRESHOLD:
            current_behavior = 'Left U-turn'
        elif angle < RIGHT_UTURN_THRESHOLD:
            current_behavior = 'Right U-turn'
        # 检查左转或右转条件
        elif angle > LEFT_TURN_THRESHOLD:
            if current_behavior != 'Left U-turn':  
                current_behavior = 'Left Turn'
        elif angle < RIGHT_TURN_THRESHOLD:
            if current_behavior != 'Right U-turn':  
                current_behavior = 'Right Turn'

    return current_behavior

def control_points_to_lane_points(lanes, t):
    if isinstance(lanes, np.ndarray):
        lanes = torch.tensor(lanes, dtype=torch.float32)
    lanes = lanes.reshape(-1, lanes.shape[0], lanes.shape[-1])
    n_control = lanes.shape[1]
    n_points = len(t)
    A = np.zeros((n_points, n_control))
    for i in range(n_points):
        for j in range(n_control):
            A[i, j] = comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
    bezier_A = torch.tensor(A, dtype=torch.float32).to(lanes.device)
    lanes = torch.einsum('ij,njk->nik', bezier_A, lanes)
    lanes = lanes.reshape(-1, lanes.shape[-1])
    return lanes

def interpolate_lane_points(lane_points, n_points=100):
    t = np.arange(n_points) / (n_points - 1)
    interpolated_points = control_points_to_lane_points(lane_points, t)
    return interpolated_points

def minimum_distance_to_bezier_curve(lane_pts, target_pt):
    distance = np.linalg.norm(target_pt - lane_pts, axis=1)
    min_distance_index = np.argmin(distance)  
    return distance[min_distance_index], min_distance_index

def find_closest_point_and_tangent(lane_pts, target_pt, n_points=100):
    distances = np.linalg.norm(target_pt - lane_pts, axis=1)
    min_index = np.argmin(distances)
    min_distance = distances[min_index]

    t_list = np.linspace(0, 1, n_points)
    closest_t = t_list[min_index]
    angles = bezier_tangent_angles(lane_pts, [closest_t])

    return min_distance, angles[0], min_index

def angle_difference(angle1, angle2):
    # 计算两个角度之间的差值
    diff = angle2 - angle1
    # 调整差值使其在(-pi, pi)范围内
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff

def check_indices_in_sublists(lst, index1, index2):
    for sublist in lst:
        if index1 in sublist and index2 in sublist:
            return True
    return False

def closest_curve(vehicle, curves):
    min_distance_angle = float('inf')
    min_distance = float('inf')
    min_angle_diff = float('inf')
    point_index = None
    closest_curve_index = None
    for i, curve in enumerate(curves):
        inter_curve = interpolate_lane_points(fit_bezier_Endpointfixed(curve[..., :2], 4), 100).numpy()
        distance, angle, point_index = find_closest_point_and_tangent(inter_curve, vehicle[..., :2])
        angle_diff = np.abs(angle_difference(angle, vehicle[-1]))
        distance_angle = distance + angle_diff
        if distance_angle < min_distance_angle:
            min_distance_angle = distance_angle
            min_distance = distance
            min_angle_diff = angle_diff
            closest_curve_index = i
            min_point_index = point_index
    
    return closest_curve_index, min_distance, min_point_index, min_angle_diff

def check_indices_in_sublists(lst, index1, index2):
    # 遍历所有子list
    for sublist in lst:
        # 检查子list是否同时包含两个index
        if index1 in sublist and index2 in sublist:
            # 如果找到，返回True
            return True
    # 如果遍历结束都没有找到，返回False
    return False


def detect_lane_change(trajs, lane_centers_list, full_paths):
    # Step 1: Find which lane center is nearest to the origin
    origin = np.array([0, 0, 0]) # x, y ,yaw
    nearest_lane_center_indices, _, _, _ = closest_curve(origin, lane_centers_list)
    ref_index = nearest_lane_center_indices

    # Step 2 and 3: Check if the nearest lane center changes along the trajectory
    for coord in trajs:
        cur_index, _, _, _ = closest_curve(coord, lane_centers_list)
        if ref_index != cur_index:
            if not check_indices_in_sublists(full_paths, ref_index, cur_index) and \
            (np.abs(lane_centers_list[ref_index][0, 1] - lane_centers_list[cur_index][0, 1]) > 1.5) and \
            (np.abs(lane_centers_list[ref_index][-1, 1] - lane_centers_list[cur_index][-1, 1]) > 1.5):
                if coord[1] > 0.1:
                   return "Left Lane Changing"
                elif coord[1] < -0.1:
                    return "Right Lane Changing"
            else:
                ref_index = cur_index
  
    return "Lane Keeping"

def calculate_vector_angle(vector):
    """计算向量与水平轴的夹角（以度为单位）"""
    angle = np.arctan2(vector[1], vector[0]) * (180 / np.pi)
    return angle

def classify_lane_direction(lane_points):
    """根据向量角度变化判断车道线的走向"""
    vectors = np.diff(lane_points, axis=0)
    angles_deg = np.array([calculate_vector_angle(vector) for vector in vectors])

    # 统计落在各个角度区间的向量数量
    fwd_count = np.sum((angles_deg >= -45) & (angles_deg <= 45))
    left_count = np.sum((angles_deg > 45) & (angles_deg <= 135))
    opp_count = np.sum((angles_deg > 135) | (angles_deg <= -135))
    right_count = np.sum((angles_deg < -45) & (angles_deg > -135))

    # 根据最多的落在哪个区间来判断车道方向
    max_count = max(fwd_count, left_count, opp_count, right_count)
    if max_count == fwd_count:
        result = "with-flow"
    elif max_count == left_count:
        result = "allowing from right to left driving"
    elif max_count == opp_count:
        result = "opposite-flow"
    elif max_count == right_count:
        result = "allowing from left to right driving"
    
    directions = (angles_deg - angles_deg[0])

    for i in range(len(directions)):
        if directions[i] < -180:
            directions[i] += 360
        elif directions[i] > 180:
            directions[i] -= 360

    # 防止异常数据，以平均值为准，乘2，看总体趋势
    direction = directions.mean() * 2
    
    if direction > 30:
        result += ", left turning lane"
    elif direction < -30:
        result += ", right turning lane"
    else:
        result += ", straight lane"
    
    return result

def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def fit_bezier_Endpointfixed(points, n_control):
    n_points = len(points)
    A = np.zeros((n_points, n_control))
    t = np.arange(n_points) / (n_points - 1)
    for i in range(n_points):
        for j in range(n_control):
            A[i, j] = comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
    A_BE = A[1:-1, 1:-1]
    _points = points[1:-1]
    _points = _points - A[1:-1, 0].reshape(-1, 1) @ points[0].reshape(1, -1) - A[1:-1, -1].reshape(-1, 1) @ points[-1].reshape(1, -1)

    conts = np.linalg.lstsq(A_BE, _points, rcond=None)

    control_points = np.zeros((n_control, points.shape[1]))
    control_points[0] = points[0]
    control_points[-1] = points[-1]
    control_points[1:-1] = conts[0]

    return control_points

def bezier_tangent_angles(control_points, t_list):
    n = len(control_points)
    derivative_points = [(n - 1) * (control_points[i + 1] - control_points[i]) for i in range(n - 1)]
    angles = []  # 存储每个t值对应的角度
    
    for t in t_list:
        # 计算t时刻的贝塞尔曲线的一阶导数
        derivative_at_t = np.zeros(2)  # 假设是二维的情况
        for i, point in enumerate(derivative_points):
            coefficient = comb(n - 2, i) * ((1 - t) ** (n - 2 - i)) * (t ** i)
            derivative_at_t += point * coefficient
        
        # 计算导数向量的角度
        angle = np.arctan2(derivative_at_t[1], derivative_at_t[0])
        
        # 将计算出的角度添加到列表中
        angles.append(angle)
    
    return np.array(angles)

def expand_lane(control_points, width=2, n_points=100, res=(0.5, 0.5), dim=(200, 200)):
    t_list = np.linspace(0, 1, n_points)
    lane_points = interpolate_lane_points(control_points, n_points)
    angles = bezier_tangent_angles(control_points, t_list)
    
    left_lane = []
    right_lane = []
    for point, angle in zip(lane_points, angles):
        dx = np.cos(angle + math.pi/2) * width
        dy = np.sin(angle + math.pi/2) * width
        left_lane.append([point[0] + dx, point[1] + dy])
        right_lane.append([point[0] - dx, point[1] - dy])
    
    # 转换为图像坐标系并进行平移
    left_lane = np.array(left_lane) / res + [dim[1] // 2, dim[0] // 2]
    right_lane = np.array(right_lane) / res + [dim[1] // 2, dim[0] // 2]
    
    return np.round(left_lane).astype(np.int32), np.round(right_lane).astype(np.int32)

def calculate_distance(bbox):
    """计算bounding box中心点的欧式距离"""
    x, y, z = bbox[:3]
    return math.sqrt(x**2 + y**2 + z**2)

def describe_crosswalks(crosswalks):
    crosswalks_part = []
    for i in range (len(crosswalks)):
        crosswalk = (
            f"Crosswalk ["
            f"({format_number(crosswalks[i][0][0])}, {format_number(crosswalks[i][0][1])}), "
            f"({format_number(crosswalks[i][1][0])}, {format_number(crosswalks[i][1][1])}), "
            f"({format_number(crosswalks[i][2][0])}, {format_number(crosswalks[i][2][1])}), "
            f"({format_number(crosswalks[i][3][0])}, {format_number(crosswalks[i][3][1])})"
            "]"
        )
        crosswalks_part.append(crosswalk)
    return crosswalks_part

def describe_lanes(lane_info):
    lanes_part = []
    lanes_red = []
    for i in range (len(lane_info['annotation']['lane_centerline'])):
        bezier_lane = fit_bezier_Endpointfixed(lane_info['annotation']['lane_centerline'][i]['points'], 4)
        desc = classify_lane_direction(lane_info['annotation']['lane_centerline'][i]['points'])
        bezier_lane_desc = (
            f"{desc} ["
            f"({format_number(bezier_lane[0][0])}, {format_number(bezier_lane[0][1])}), "
            f"({format_number(bezier_lane[1][0])}, {format_number(bezier_lane[1][1])}), "
            f"({format_number(bezier_lane[2][0])}, {format_number(bezier_lane[2][1])}), "
            f"({format_number(bezier_lane[3][0])}, {format_number(bezier_lane[3][1])})"
            "]"
        )
        category = ['']
        for j in range (len(lane_info['annotation']['topology_lcte'][i])):
            if lane_info['annotation']['topology_lcte'][i][j] != 0:
                if te_attribute[lane_info['annotation']['traffic_element'][j]['attribute']] not in category:
                    category += [te_category[lane_info['annotation']['traffic_element'][j]['category']-1]]
                    category += [te_attribute[lane_info['annotation']['traffic_element'][j]['attribute']]]
                    if te_attribute[lane_info['annotation']['traffic_element'][j]['attribute']] == 'red':
                            lanes_red.append(bezier_lane)
        lanes_part.append(' '.join([bezier_lane_desc]+category))
    return lanes_part, lanes_red

def describe_tl(lane_info):
    desc_tl = "Traffic Light Existing: False"
    for i in range (len(lane_info['annotation']['traffic_element'])):
        if lane_info['annotation']['traffic_element'][i]['category'] == 1:
            desc_tl = "Traffic Light Existing: True"
            break
        
    return desc_tl

def describe_expert(gt_planning, planning_mask, lane_pts, full_paths, pred_traj, pred_traj_mask, names, bboxes, attrs):
    planning_traj = gt_planning[..., :2]
    planning_yaw = gt_planning[..., 2]
    mask = planning_mask.any(axis=1)

    combined_data = list(zip(names, bboxes, attrs, pred_traj, pred_traj_mask))
    
    filtered_data = [(name, bbox, attr, traj, traj_mask) for name, bbox, attr, traj, traj_mask in combined_data if abs(bbox[0]) <= 50 and abs(bbox[1]) <= 50]
    all_names = []
    all_dists = []
    all_xy = []
    for name, bbox, attr, traj, traj_mask in filtered_data:
        if attr == '':
            full_name = name
        else:
            attr = attr.split('.')[1]
            full_name = name + f'.{attr}'
        traj = np.cumsum(traj, axis=1)
        traj += bbox[:2]
        masked_planning = gt_planning[mask]
        masked_traj = traj[traj_mask.astype(bool)][:6]
        dist_rec = np.linalg.norm(bbox[:2])

        # 检查是否有空数组，如果有，则不能计算距离
        if masked_planning.size == 0 or masked_traj.size == 0:
            l2_norm = dist_rec
        else:
            # 若两数组长度不同，取较小的长度来计算L2 Norm
            min_len = min(len(masked_planning), len(masked_traj))
            
            # 计算L2 Norm
            l2_norm = np.linalg.norm(masked_planning[:min_len][..., :2] - masked_traj[:min_len], axis=1).min()
        dist = min(dist_rec, l2_norm)
    
        if dist <= 10.0:
            all_names.append(full_name)
            all_dists.append(dist)
            all_xy.append(bbox[:2])

    ego_vel = calculate_speed(planning_traj, mask)
    speed_state = judge_speed_changes(ego_vel[..., 0])
    self_action = f"Expert decision: {speed_state}"
    lane_change = detect_lane_change(gt_planning[mask], lane_pts, full_paths)
    turning_behavior = determine_turning_behavior(planning_yaw)
    if speed_state not in ["Stopped", "Unknown"]:
        if turning_behavior == "Go Straight":
            self_action = self_action + ", " + lane_change
        if not (lane_change != "Lane Keeping" and turning_behavior == "Go Straight"):
            self_action = self_action + ", " + turning_behavior
    
    formatted_points = ', '.join(f"({format_number(point[0], 2)}, {format_number(point[1], 2)})" for point in planning_traj[mask])
    self_traj = f"Expert trajectory: [PT, {formatted_points}]."
    ego_state = [self_action, self_traj]
    description = '\n'.join(ego_state)

    if len(all_dists):
        desc_near = f"Objects near your path: "
        for i, obj in enumerate(all_names):
            desc_near += f"{all_names[i]} at ({format_number(all_xy[i][0])}, {format_number(all_xy[i][1])})"
            if i != len(all_dists) -1:
                desc_near += ", "
            else:
                desc_near += "."
        description = description + "\n" + desc_near
    return description

def describe_expertv2(gt_planning, planning_mask, lane_pts, full_paths, pred_traj, pred_traj_mask, names, bboxes, attrs):
    planning_traj = gt_planning[..., :2]
    planning_yaw = gt_planning[..., 2]
    mask = planning_mask.any(axis=1)

    combined_data = list(zip(names, bboxes, attrs, pred_traj, pred_traj_mask))
    
    filtered_data = [(name, bbox, attr, traj, traj_mask) for name, bbox, attr, traj, traj_mask in combined_data if abs(bbox[0]) <= 50 and abs(bbox[1]) <= 50]
    all_names = []
    all_dists = []
    all_xy = []
    for name, bbox, attr, traj, traj_mask in filtered_data:
        if attr == '':
            full_name = name
        else:
            attr = attr.split('.')[1]
            full_name = name + f'.{attr}'
        traj = np.cumsum(traj, axis=1)
        traj += bbox[:2]
        masked_planning = gt_planning[mask]
        masked_traj = traj[traj_mask.astype(bool)][:6]
        dist_rec = np.linalg.norm(bbox[:2])

        if masked_planning.size == 0 or masked_traj.size == 0:
            l2_norm = dist_rec
        else:
            min_len = min(len(masked_planning), len(masked_traj))
            
            l2_norm = np.linalg.norm(masked_planning[:min_len][..., :2] - masked_traj[:min_len], axis=1).min()
        dist = min(dist_rec, l2_norm)
    
        if dist <= 10.0:
            all_names.append(full_name)
            all_dists.append(dist)
            all_xy.append(bbox[:2])

    ego_vel = calculate_speed(planning_traj, mask)
    speed_state = judge_speed_changes(ego_vel[..., 0])
    self_action = f"Expert decision: {speed_state}"
    lane_change = detect_lane_change(gt_planning[mask], lane_pts, full_paths)
    turning_behavior = determine_turning_behavior(planning_yaw)
    if speed_state not in ["Stopped", "Unknown"]:
        if turning_behavior == "Go Straight":
            self_action = self_action + ", " + lane_change
        if not (lane_change != "Lane Keeping" and turning_behavior == "Go Straight"):
            self_action = self_action + ", " + turning_behavior
    
    formatted_points = ', '.join(f"({format_number(point[0], 2)}, {format_number(point[1], 2)})" for point in planning_traj[mask])
    self_traj = f"Expert trajectory: [PT, {formatted_points}]."
    ego_state = [self_action]
    description = '\n'.join(ego_state)

    return description

def get_vectorized_lines(map_geoms):
    ''' Vectorize map elements. Iterate over the input dict and apply the 
    specified sample funcion.
    
    Args:
        line (LineString): line
    
    Returns:
        vectors (array): dict of vectorized map elements.
    '''

    vectors = {}
    for label, geom_list in map_geoms.items():
        vectors[label] = []
        for geom in geom_list:
            if geom.geom_type == 'LineString':
                line = geom.simplify(0.2, preserve_topology=True)
                line = np.array(line.coords)
                vectors[label].append(line)

            elif geom.geom_type == 'Polygon':
                # polygon objects will not be vectorized
                continue
            else:
                raise ValueError('map geoms must be either LineString or Polygon!')
    return vectors

def get_crosswalks(vectorized_lines):
    crosswalks = []
    for points in vectorized_lines[0]:
        points = np.array(points, dtype=np.float32)
        hull = cv2.convexHull(points)
        rect = cv2.minAreaRect(hull) # (cx, cy), (width, height), angle 
        box = cv2.boxPoints(rect)
        crosswalks.append(box)
    return crosswalks

def add_ego2lane(gt_planning, planning_mask, lane_pts, lane_objects):
    planning_traj = gt_planning[..., :2]
    mask = planning_mask.any(axis=1)
    vel = calculate_speed(planning_traj, mask)
    index, dist, _, _ = closest_curve(np.array([0.0, 0.0, 0.0]), lane_pts)

    ego_info = f"your own car"

    if dist >= 3.5:
        lane_objects['others'].insert(0, ego_info)
        return lane_objects, None
    else:
        if index not in lane_objects:
            lane_objects[index] = []
        lane_objects[index].insert(0, ego_info)
        return lane_objects, index
        
def is_approaching_or_moving_away(attr,ego_vel, position, velocity):
    if len(ego_vel):
        ego_vel = ego_vel[0]
    else:
        ego_vel = (0.0, 0.0)
    relative_velocity = (velocity[0] - ego_vel[0], velocity[1] - ego_vel[1])
    
    future_position = (position[0] + 5 * relative_velocity[0], position[1] + 5 * relative_velocity[1])
    
    distance_future = (future_position[0] ** 2 + future_position[1] ** 2) ** 0.5
    
    dot_product = position[0] * relative_velocity[0] + position[1] * relative_velocity[1]

    result = ""
    
    if attr in ["moving", "with_rider"]:
        if dot_product > 0:
            result = "moving away from you"
        elif dot_product < 0:
            if distance_future <= 10:
                result = "approaching you"
            
    return result

def analyze_position(x, y, angle_deg, desc_direction):
    direction = ''
    if x > 0:
        direction += 'front'
    elif x < 0:
        direction += 'back'

    if y > 2.5:
        direction += ' left'
    elif y < -2.5:
        direction += ' right'

    
    if desc_direction:
        if abs(angle_deg) < 45:
            direction += ", same direction as you, "
        elif abs(abs(angle_deg) - 180) < 45:
            direction += ", opposite direction from you, "
        elif abs(angle_deg - 90) < 45:
            direction += ", heading from right to left, "
        elif abs(angle_deg + 90) < 45:
            direction += ", heading from left to right, "

    return direction.strip()

def format_det_answer(full_name, bbox, vel, desc_direction):
    x = bbox[0]
    y = bbox[1]
    z = bbox[2]
    l = bbox[3]
    w = bbox[4]
    h = bbox[5]
    yaw = math.degrees(bbox[6])
    vx = vel[0]
    vy = vel[1]

    position = analyze_position(x, y, yaw, desc_direction)

    answer = f"{full_name} in the {position} "
    answer += f"location: ({format_number(x)}, {format_number(y)})"
    # answer += f"length: {l:.1f}, width: {w:.1f}, height: {h:.1f}, "
    # answer += f"angles in degrees: {format_number(yaw)}"
    if np.sqrt(vx**2 + vy**2) > 0.2:
        answer += f", velocity: ({format_number(vx)}, {format_number(vy)}).  "
    else:
        answer += "."

    return answer
    
def describe_objects2lane(gt_planning, planning_mask, objects_list, bboxes, velocity, attrs, lane_pts, crosswalks):
    lane_objects = {}
    lane_objects['others'] = []
    crosswalk_objects = {}

    combined_data = list(zip(objects_list, bboxes, velocity, attrs))

    filtered_data = [(name, bbox, vel, attr) for name, bbox, vel, attr in combined_data if abs(bbox[0]) <= 50 and abs(bbox[1]) <= 50]
    sorted_data = sorted(filtered_data, key=lambda item: calculate_distance(item[1]))

    for name, bbox, vel, attr in sorted_data:
        desc_direction = True
        if attr == '':
            full_name = name
            desc_direction = False
        else:
            attr = attr.split('.')[1]
            full_name = name + f'.{attr}'
        
        text = format_det_answer(full_name, bbox, vel, desc_direction)
        
        if "pedestrian" in name:
            for i, crosswalk in enumerate(crosswalks):
                if point_in_rotated_rect(bbox, crosswalk):
                    if i not in crosswalk_objects:
                        crosswalk_objects[i] = []
                    crosswalk_objects[i].append(text)
                # continue

        index, dist, _, _ = closest_curve(np.concatenate([bbox[:2], bbox[6:7]], -1), lane_pts)
        if dist >= 3.0:
            lane_objects['others'].append(text)
        else:
            if index not in lane_objects:
                lane_objects[index] = []
            lane_objects[index].append(text)
    
    return lane_objects, crosswalk_objects

def scene_description(gt_planning, planning_mask, lane_info, objects_list, bboxes, velocity, attrs, lane_pts, crosswalks):
    output_lines = []
    
    tl_description = describe_tl(lane_info)
    output_lines.append(tl_description)
    
    lane_description, lanes_red = describe_lanes(lane_info)

    crosswalk_description = describe_crosswalks(crosswalks)
    
    lane_objects, crosswalk_objects = describe_objects2lane(gt_planning, planning_mask, objects_list, bboxes, velocity, attrs, lane_pts, crosswalks)
    if len(objects_list) == 0:
        output_lines.append(f"No traffic participants observed in the scene.")
    lane_objects, ego_index = add_ego2lane(gt_planning, planning_mask, lane_pts, lane_objects)

    for i, crosswalk_desc in enumerate(crosswalk_description):
        output_lines.append(f"├── {crosswalk_desc}")
        if i in crosswalk_objects.keys():
            for obj_desc in crosswalk_objects[i]:
                output_lines.append(f"│   ├── {obj_desc}")

    for i, lane_desc in enumerate (lane_description):
        if i == ego_index:
            lane_desc = lane_desc.replace("with-flow, ", "your current ")
        output_lines.append(f"├── {lane_desc}")
        if i in lane_objects.keys():
            for obj_desc in lane_objects[i]:
                            output_lines.append(f"│   ├── {obj_desc}")

    if 'others' in lane_objects:
        output_lines.append("├── Other Lanes/Roadside")
        for obj_desc in lane_objects['others']:
            output_lines.append(f"│   ├── {obj_desc}")

    return '\n'.join(output_lines), lanes_red

def describe_simulated(T, planning_trajs, lane_pts, all_coll_objs, red_light, out_of_drivable, full_paths):
    all_desc = []
    for i, traj in enumerate(planning_trajs):
        coll_obj = all_coll_objs[i]
        comment = []
        
        planning_traj = traj[..., :T*2].reshape(T, 2)
        planning_yaw= traj[..., T*3:T*4].reshape(T, 1)
        mask = np.ones_like(planning_yaw, dtype=np.bool8).reshape(T)
        ego_vel = calculate_speed(planning_traj, mask)
        speed_state = judge_speed_changes(ego_vel[..., 0])
        self_action = f"Example decision: {speed_state}"

        if red_light[i] and speed_state not in ["Stopped", "Unknown"]:
            comment += ["Run the red light"]

        lane_change = detect_lane_change(np.concatenate([planning_traj, planning_yaw], -1), lane_pts, full_paths)
        turning_behavior = determine_turning_behavior(planning_yaw)
        if speed_state not in ["Stopped", "Unknown"]:
            if turning_behavior == "Go Straight":
                self_action = self_action + ", " + lane_change
            if not (lane_change != "Lane Keeping" and turning_behavior == "Go Straight"):
                self_action = self_action + ", " + turning_behavior
        
        formatted_points = ', '.join(f"({format_number(point[0], 2)}, {format_number(point[1], 2)})" for point in planning_traj[mask])
        self_traj = f"Example trajectory: [PT, {formatted_points}]."

        if out_of_drivable[i]:
            comment += ["Out of the drivable area"]

        if len(coll_obj):
            desc_coll = f"Collide with: "
            for i, obj in enumerate(coll_obj):
                name, attr, box = obj
                if attr == '':
                    full_name = name
                else:
                    attr = attr.split('.')[1]
                    full_name = name + f'.{attr}'
                desc_coll += f"{full_name} at ({format_number(box[0])}, {format_number(box[1])})"
                if i != len(coll_obj) -1:
                    desc_coll += ", "
                else:
                    desc_coll += "."
            comment += [desc_coll]
        
        if len(comment) == 0:
            comment += ["Relatively safe"]
        ego_state = [self_action, self_traj] + comment
        description = '\n'.join(ego_state)
        all_desc.append(description)
    all_desc = '\n\n'.join(all_desc)
    return all_desc
    