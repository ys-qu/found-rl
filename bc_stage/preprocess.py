import h5py
import glob
import torch as th
from PIL import Image
import json
import carla_gym.utils.transforms as trans_utils
import carla_gym.core.task_actor.common.navigation.route_manipulation as gps_util
import carla
import numpy as np
from torchvision import transforms as T
import uuid
import os
from tqdm import tqdm


COMMANDS = [
    'turn left at the intersection',
    'turn right at the intersection',
    'go straight at the intersection',
    'follow the current lane',
    'change to the left lane',
    'change to the right lane'
]


def process_obs(obs, speed_factor=12.0):
    ev_gps = obs['gnss']['gnss']
    # imu nan bug
    compass = 0.0 if np.isnan(obs['gnss']['imu'][-1]) else obs['gnss']['imu'][-1]

    gps_point = obs['gnss']['target_gps']
    target_vec_in_global = gps_util.gps_to_location(gps_point) - gps_util.gps_to_location(ev_gps)
    ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass) - 90.0)
    loc_in_ev = trans_utils.vec_global_to_ref(target_vec_in_global, ref_rot_in_global)

    # VOID = -1
    # LEFT = 1
    # RIGHT = 2
    # STRAIGHT = 3
    # LANEFOLLOW = 4
    # CHANGELANELEFT = 5
    # CHANGELANERIGHT = 6
    command = obs['gnss']['command'][0]
    if command < 0:
        command = 4
    command -= 1

    assert command in [0, 1, 2, 3, 4, 5]
    command_prompt = COMMANDS[command]

    speed = obs['speed']['forward_speed'][0]
    forward_vector_x = loc_in_ev.x
    forward_vector_y = loc_in_ev.y

    im = obs['central_rgb']['data'][:]

    policy_input = {
        'speed': speed,
        'forward_vector': np.array([forward_vector_x, forward_vector_y], dtype=np.float32),
        'command': command_prompt,
        'image': im
    }
    return policy_input


def process_act(action):
    throttle, steer, brake = action[0], action[1], action[2]
    throttle = np.clip(throttle, 0, 1)
    steer = np.clip(steer, -1, 1)
    brake = np.clip(brake, 0, 1)
    return np.array([throttle, steer, brake], dtype=np.float32)


def getitem(data):
    # image
    image_array = data["image"]
    image = Image.fromarray(image_array)

    # get command, speed, gps, etc.
    command = data["command"]
    speed = data["speed"].item()
    forward_vector = data["forward_vector"]

    # action
    action = data["action"]

    # format prompt
    prompt = (
        "<image>\n"
        "You are a driving agent in a simulated environment.\n"
        f"Current speed: {speed:.1f} m/s.\n"
        f"Forward vector: ({forward_vector[0]:.2f}, {forward_vector[1]:.2f}).\n"
        f"Navigation command: \"{command}\".\n"
        "Based on the visual input and state information, predict the next action.\n"
        "Return the action in the following JSON format:\n"
        "{\"throttle\": float, \"steer\": float, \"brake\": float}"
    )

    # format answer
    answer = {
        "throttle": round(action[0].item(), 2),
        "steer": round(action[1].item(), 2),
        "brake": round(action[2].item(), 2),
    }

    return {
        "id": uuid.uuid4().hex[:16],
        "image": image,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": str(json.dumps(answer))}
        ]
    }


def h5_group_to_dict(group):
    result = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            result[key] = item[()]  # Convert dataset to numpy array or scalar
        elif isinstance(item, h5py.Group):
            result[key] = h5_group_to_dict(item)  # Recursively process sub-groups
    return result


ROOT_PATH = '/path/to/raw_data'
ROOT_DIR = '/dir/to/processed_data'

json_list = []
json_list_bev = []
for data_type in ['roach_cc', 'roach_lb', 'roach_cc']:  #
    path_list = sorted(glob.glob(f'{ROOT_PATH}/{data_type}/expert/*.h5'), key=lambda x: int(x.split('/')[-1].split('.')[0]))
    OUTPUT_DIR = ROOT_DIR + '/' + data_type
    json_list_type = []
    json_list_bev_type = []
    for path in tqdm(path_list):
        bev_scenario = []
        with h5py.File(path, 'r') as f:
            for step_name in sorted(f.keys(), key=lambda x: int(x.split('_')[1])):
                step_data = f[step_name]
                obs = step_data['obs']
                supervision = step_data['supervision']
                data_dict = process_obs(obs)
                speed = data_dict['speed']
                action = supervision['action']
                action = process_act(action)
                data_dict.update({'action': action})

                data_dict = getitem(data_dict)

                scenario = path.split("/")[-1].split(".")[0]
                # save_path = f'{OUTPUT_DIR}/rgb/{scenario}/{scenario}_{step_name}.jpg'
                # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # data_dict['image'].save(save_path)
                # data_dict['image'] = f'{data_type}/rgb/{scenario}/{scenario}_{step_name}.jpg'
                # json_list.append(data_dict)
                # json_list_type.append(data_dict)

                im = obs['birdview']['masks'][:]   #########################rendered
                # save_path = f'{OUTPUT_DIR}/birdview/{scenario}/{scenario}_{step_name}.jpg'
                # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                reward_done = step_data['reward_done']
                reward_done = h5_group_to_dict(reward_done)
                reward = reward_done['reward']
                done = reward_done['done']
                bev_dict = {}
                bev_dict['image'] = im
                bev_dict['reward'] = reward
                bev_dict['done'] = done
                bev_dict['action'] = action
                bev_dict['speed'] = speed
                bev_scenario.append(bev_dict)

                # im_ = Image.fromarray(im)
                # save_path = f'{OUTPUT_DIR}/birdview/{scenario}/{scenario}_{step_name}.jpg'
                # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # im_.save(save_path)
                # prompt = (
                #     "<image>\n"
                #     "You are a driving agent in a simulated environment.\n"
                #     f"Current speed: {speed:.1f} m/s.\n"
                #     "Based on the visual input and state information, predict the next action.\n"
                #     "Return the action in the following JSON format:\n"
                #     "{\"throttle\": float, \"steer\": float, \"brake\": float}"
                # )
                # data_dict_bev = {
                #     "id": uuid.uuid4().hex[:16],
                #     "image": f'{data_type}/birdview/{scenario}/{scenario}_{step_name}.jpg',
                #     "conversations": [
                #         {"from": "human", "value": prompt},
                #         {"from": "gpt", "value": data_dict['conversations'][1]['value']}
                #     ]
                # }
                # json_list_bev.append(data_dict_bev)
                # json_list_bev_type.append(data_dict_bev)

        bev_chunk = {
            "image": np.stack([d["image"] for d in bev_scenario]),  # shape: (T, H, W, 3)
            "reward": np.array([d["reward"] for d in bev_scenario]),  # shape: (T,)
            "done": np.array([d["done"] for d in bev_scenario]),  # shape: (T,)
            "action": np.stack([d["action"] for d in bev_scenario]),  # shape: (T, action_dim)
            'speed': np.array([d["speed"] for d in bev_scenario]),  # shape: (T,)
        }
        # print(bev_chunk['image'].shape)
        save_path = f'{OUTPUT_DIR}/bev_masks_npz/{scenario}_{step_name}.npz'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, **bev_chunk)

#     with open(f'{ROOT_DIR}/{data_type}/rgb/carla_roach_rgb_{data_type}.json', "w") as f:
#         json.dump(json_list_type, f, indent=2)
#     with open(f'{ROOT_DIR}/{data_type}/birdview/carla_roach_bev_{data_type}.json', "w") as f:
#         json.dump(json_list_bev_type, f, indent=2)
#
# with open(f'{ROOT_DIR}/carla_roach_rgb.json', "w") as f:
#     json.dump(json_list, f, indent=2)
# with open(f'{ROOT_DIR}/carla_roach_bev.json', "w") as f:
#     json.dump(json_list_bev, f, indent=2)