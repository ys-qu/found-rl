"""
Optional cleaner VLM inference implementation (vLLM and transformers). Not used in default experiments.
"""
import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
import ast
import os
import gym
import math
from pathlib import Path
import json
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import subprocess
import sys

from stable_baselines3.common.vec_env.base_vec_env import tile_images
from gym.wrappers.monitoring.video_recorder import ImageEncoder

from carla_gym.utils import config_utils
from carla_gym.utils.expert_noiser import ExpertNoiser
from utils import saving_utils, server_utils
from agents.rl_birdview.utils.wandb_callback import WandbCallback
# from agents.rl_birdview.utils.rl_birdview_wrapper import RlBirdviewWrapper
import carla_gym.utils.transforms as trans_utils
import carla_gym.core.task_actor.common.navigation.route_manipulation as gps_util
import uuid
from rwkv_v7.src.utils import Conversation
from rwkv_v7.src.dataset import process_image_tokens_in_conversations, preprocess
from rwkv_v7.src.dataset import DEFAULT_IMAGE_TOKEN, DEFAULT_STOP_TOKEN, STOP_TOKEN_INDEX
import torch
from qwen_vl_utils import process_vision_info
from rwkv_v7.tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
from rwkv_v7.src.config import VISION_TOWER_CHECKPOINT_NAMES
from pytorch_lightning.utilities import rank_zero_info
import carla
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

COMMANDS = [
    'turn left at the intersection',
    'turn right at the intersection',
    'go straight at the intersection',
    'follow the current lane',
    'change to the left lane',
    'change to the right lane'
]

from typing import NamedTuple, Optional

try:

    from vllm import LLM, EngineArgs, SamplingParams

    class ModelRequestData(NamedTuple):
        engine_args: EngineArgs
        prompts: list[str]
        stop_token_ids: Optional[list[int]] = None
except:
    print('[INFO] Will not use vllm for inference')


import os
os.environ["VLLM_SKIP_VIDEO_PROFILE"] = "1"


def load_rwkv(args=None, model_device='cuda'):
    from rwkv_v7.src.model import VisualRWKV
    model_path = Path(args.model_path)
    model_name = model_path.parent.name
    args.vision_tower_path = {name: Path(args.vision_tower_dir) / path for name, path in
                              VISION_TOWER_CHECKPOINT_NAMES.items()}
    # Model
    model = VisualRWKV(args).bfloat16().to(model_device)
    if args.model_path:
        raw_model = torch.load(args.model_path, map_location='cpu', weights_only=True)
        # use pos_embed from pretrained model
        if "vit.dino_featurizer.pos_embed" in raw_model:
            del raw_model["vit.dino_featurizer.pos_embed"]
        if "vit.siglip_featurizer.pos_embed" in raw_model:
            del raw_model["vit.siglip_featurizer.pos_embed"]
        msg = model.load_state_dict(raw_model, strict=False)
        rank_zero_info(f"loading visual rwkv model from {args.model_path}: {msg}")
    if args.freeze_rwkv > 0:
        model.freeze_rwkv(args.freeze_rwkv)
    if args.freeze_proj > 0:
        model.freeze_proj()
    model.freeze_emb()  # freeze emb all the time

    # init training data
    tokenizer_path = f"{PROJECT_ROOT}/rwkv_v7/tokenizer/rwkv_vocab_v20230424.txt"
    print(f'[INFO] Found tokenizer at {tokenizer_path}')
    args.tokenizer = TRIE_TOKENIZER(tokenizer_path)
    args.image_processor = model.vit.get_image_transform()
    return model, args


def load_qwen(vlm_path_or_url, model_device='cuda'):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model_device = 'auto' if model_device is None else model_device
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_path_or_url, torch_dtype="auto", device_map=model_device
    )
    processor = AutoProcessor.from_pretrained(vlm_path_or_url)
    return model, processor


def load_qwen_vllm(vlm_path_or_url):
    from vllm import LLM
    from transformers import AutoProcessor
    llm = LLM(
        model=vlm_path_or_url,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=2048,
    )
    processor = AutoProcessor.from_pretrained(vlm_path_or_url, trust_remote_code=True)
    return llm, processor


def load_llava(vlm_path_or_url, model_device='cuda'):
    from transformers import LlavaForConditionalGeneration, LlavaProcessor
    model_device = 'auto' if model_device is None else model_device
    model = LlavaForConditionalGeneration.from_pretrained(
        vlm_path_or_url,
        torch_dtype="auto",
        device_map=model_device
    )
    processor = LlavaProcessor.from_pretrained(vlm_path_or_url)
    return model, processor


def load_llava_vllm(vlm_path_or_url):
    from vllm import LLM
    from transformers import LlavaProcessor
    llm = LLM(
        model=vlm_path_or_url,
        trust_remote_code=True,
        dtype="auto"
    )
    processor = LlavaProcessor.from_pretrained(vlm_path_or_url, trust_remote_code=True)
    return llm, processor


def load_internvl(vlm_path_or_url, model_device='cuda'):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    model_device = 'auto' if model_device is None else model_device
    processor = AutoProcessor.from_pretrained(vlm_path_or_url)
    if 'full' in vlm_path_or_url:
        model = AutoModelForImageTextToText.from_pretrained(vlm_path_or_url, device_map=model_device,
                                                            torch_dtype="float32")
    else:
        model = AutoModelForImageTextToText.from_pretrained(vlm_path_or_url, device_map=model_device,
                                                            torch_dtype="auto")
    return model, processor


def load_internvl_vllm(vlm_path_or_url):
    from vllm import LLM
    from transformers import AutoProcessor
    torch_dtype = "float16" if 'full' in vlm_path_or_url else "auto"
    llm = LLM(
        model=vlm_path_or_url,
        trust_remote_code=True,
        dtype=torch_dtype
    )
    processor = AutoProcessor.from_pretrained(vlm_path_or_url, trust_remote_code=True)
    return llm, processor


def load_vlm(vlm_path_or_url, args=None, model_device='cuda', use_vllm=True):
    if 'rwkv' in vlm_path_or_url:
        model, args = load_rwkv(args, model_device)
        processor = None
    elif 'qwen' in vlm_path_or_url.lower():
        if use_vllm:
            model, processor = load_qwen_vllm(vlm_path_or_url)
        else:
            model, processor = load_qwen(vlm_path_or_url, model_device)
    elif 'llava' in vlm_path_or_url.lower():
        if use_vllm:
            model, processor = load_llava_vllm(vlm_path_or_url)
        else:
            model, processor = load_llava(vlm_path_or_url, model_device)
    elif 'intern' in vlm_path_or_url.lower():
        # https://github.com/vllm-project/vllm/issues/17463
        # so far, the internvl-hf is not supported by vllm
        if use_vllm:
            model, processor = load_internvl_vllm(vlm_path_or_url)
        else:
            model, processor = load_internvl(vlm_path_or_url, model_device)
    else:
        raise NotImplementedError(f'{vlm_path_or_url} is not supported!')

    return model, processor, args


def infer_rwkv(vlm_model, image, prompt, image_processor=None, tokenizer=None, ctx_len=2048, num_token_per_image=1024):
    messages = {
        "id": uuid.uuid4().hex[:16],
        "image": Image.fromarray(image),
        "conversations": [
            {
                "from": "human",
                "value": prompt
            },
            {
                "from": "gpt",
                "value": ""
            }
        ]
    }
    def get_question_id(line):
        if "question_id" in line:
            return line["question_id"]
        elif "id" in line:
            return line["id"]
        elif "index" in line:
            return line["index"]
        else:
            raise ValueError("Cannot find question id in line: {}".format(line))

    def get_single_image_dict(line, image_processor):
        image_dict = {}
        if "image" in line:
            image = line['image'].convert("RGB")
            pixel_values = image_processor(image)  # dict with keys 'dino' and 'siglip' and 'sam'
            # unsqueeze to add batch dimension
            for key in pixel_values:
                image_dict[key] = pixel_values[key].unsqueeze(0)
        else:
            raise ValueError("no key 'image' in line: {}".format(line))
        return image_dict

    def get_input_text(line, num_images):
        input_text = line["text"] if "text" in line else line["conversations"][0]["value"]
        # remove DEFAULT_IMAGE_TOKEN
        input_text = input_text.replace(DEFAULT_IMAGE_TOKEN, "").strip()
        # add <image> tokens
        image_prifix = "\n".join(num_images * [DEFAULT_IMAGE_TOKEN])
        input_text = image_prifix + "\n" + input_text
        return input_text

    idx = get_question_id(messages)
    image_dict = get_single_image_dict(messages, image_processor)
    for k in image_dict:
        image_dict[k] = image_dict[k].bfloat16().to(vlm_model.device)
        num_images = image_dict[k].shape[0]
    input_text = get_input_text(messages, num_images=num_images)
    conv = Conversation(id=idx, roles=["human", "gpt"], conversations=[])
    conv.append_message(conv.roles[0], input_text)
    conv.append_message(conv.roles[1], "")
    conversations = process_image_tokens_in_conversations(
        conv.conversations,
        num_image_paths=num_images, )

    data_dict = preprocess(
        conversations,
        tokenizer,
        has_image=True,
        ctx_len=ctx_len,
        num_token_per_image=num_token_per_image,
        pad_token_id=0,
        do_pad_to_max_length=False)
    input_ids = data_dict['input_ids'].unsqueeze(0).to(vlm_model.device)

    with torch.inference_mode():
        output_ids, output_logits, output_probs = vlm_model.generate(
            input_ids,
            images=image_dict,
            do_sample=False,
            temperature=0.2,  # args.temperature,
            top_p=None,  # args.top_p,
            max_new_tokens=128,  # args.max_new_tokens,
            stop_token_idx=STOP_TOKEN_INDEX)

    if isinstance(output_ids, list):
        output_ids = torch.tensor(output_ids).to(dtype=torch.long, device=input_ids.device)
    if output_ids.dim() == 1:
        output_ids = output_ids.unsqueeze(0)

    output_text = tokenizer.decode(output_ids[0].tolist()).split(DEFAULT_STOP_TOKEN)[0].strip()
    try:
        control_dict = json.loads(output_text)
    except Exception:
        try:
            control_dict = ast.literal_eval(output_text)
        except Exception:
            control_dict = {"throttle": 0.0, "steer": 0.0, "brake": 0.0}

    control = carla.VehicleControl(
        throttle=control_dict.get("throttle", 0.0),
        steer=control_dict.get("steer", 0.0),
        brake=control_dict.get("brake", 0.0)
    )
    return control


def infer_qwen(vlm_model, image, prompt, processor=None):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Image.fromarray(image).convert("RGB"),
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(vlm_model.device)

    # Inference: Generation of the output
    generated_ids = vlm_model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    control_dict = json.loads(output_text[0])

    control = carla.VehicleControl(
        throttle=control_dict.get("throttle", 0.0),
        steer=control_dict.get("steer", 0.0),
        brake=control_dict.get("brake", 0.0)
    )
    return control


def infer_llava(vlm_model, image, prompt, processor=None):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Image.fromarray(image).convert("RGB"),  # central_rgb: np.array(H, W, 3)
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image = messages[0]["content"][0]["image"]

    inputs = processor(
        text=[text],
        images=[image],  # MUST be list of PIL.Image
        padding=True,
        return_tensors="pt"
    ).to(vlm_model.device)

    generated_ids = vlm_model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    control_dict = json.loads(output_text[0])
    control = carla.VehicleControl(
        throttle=control_dict.get("throttle", 0.0),
        steer=control_dict.get("steer", 0.0),
        brake=control_dict.get("brake", 0.0)
    )
    return control


def infer_internvl(vlm_model, image, prompt, processor=None):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Image.fromarray(image).convert("RGB"),  # central_rgb: np.array(H, W, 3)
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image = messages[0]["content"][0]["image"]

    inputs = processor(
        text=[text],
        images=[image],  # MUST be list of PIL.Image
        padding=True,
        return_tensors="pt"
    ).to(vlm_model.device)

    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(dtype=vlm_model.dtype)

    generated_ids = vlm_model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    control_dict = json.loads(output_text[0])
    control = carla.VehicleControl(
        throttle=control_dict.get("throttle", 0.0),
        steer=control_dict.get("steer", 0.0),
        brake=control_dict.get("brake", 0.0)
    )
    return control


def infer_vllm(vlm_model, image, prompt, processor=None):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Image.fromarray(image),
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = [{
        "prompt": text,
        "multi_modal_data": {
            "image": [Image.fromarray(image)]
        },
    }]
    sampling_params = SamplingParams(
        max_tokens=128,
    )
    results = vlm_model.generate(inputs, sampling_params)
    output_text = [result.outputs[0].text for result in results]
    control_dict = json.loads(output_text[0])
    control = carla.VehicleControl(
        throttle=control_dict.get("throttle", 0.0),
        steer=control_dict.get("steer", 0.0),
        brake=control_dict.get("brake", 0.0)
    )
    return control


def generate_vlm_agent_action(vlm_model, model_name_or_path, obs_, image_processor=None, tokenizer=None, processor=None,
                              use_bev_obs=True, use_vllm=True):
    """
    obs: {
        'hero':{
            'birdview': {
                'rendered'
                'masks'
            },
            'speed': {
                'speed',
                'speed_xy',
                'forward_speed'
            }
            'control':{
                'throttle',
                'steer',
                'brake',
                'gear',
                'speed_limit'
            }
            'velocity':{
                'acc_xy',
                'vel_xy',
                'vel_ang_z'
            }
            'gnss':{
                'gnss',
                'imu',
                'target_gps',
                'command'
            }
            'central_rgb':{
                'frame',
                'data'
            }
            'route_plan':{
                'location',
                'command',
                'road_id',
                'lane_id',
                'is_junction'
            }
        }
    }

    """
    # policy_input = RlBirdviewWrapper.process_obs(obs, ['control', 'vel_xy'])
    # birdview = policy_input['birdview']
    obs = obs_['hero']

    speed = obs['speed']['forward_speed'][0]

    ev_gps = obs['gnss']['gnss']
    compass = 0.0 if np.isnan(obs['gnss']['imu'][-1]) else obs['gnss']['imu'][-1]
    gps_point = obs['gnss']['target_gps']
    target_vec_in_global = gps_util.gps_to_location(gps_point) - gps_util.gps_to_location(ev_gps)
    ref_rot_in_global = carla.Rotation(yaw=float(np.rad2deg(compass)) - 90.0)
    loc_in_ev = trans_utils.vec_global_to_ref(target_vec_in_global, ref_rot_in_global)
    forward_vector_x = loc_in_ev.x
    forward_vector_y = loc_in_ev.y

    command = obs['gnss']['command'][0]
    if command < 0:
        command = 4
    command -= 1
    assert command in [0, 1, 2, 3, 4, 5]
    command_prompt = COMMANDS[command]

    if use_bev_obs:
        image_np = obs['birdview']['rendered']
        prompt = (
            "<image>\n"
            "You are a driving agent in a simulated environment.\n"
            f"Current speed: {speed:.1f} m/s.\n"
            f"Forward vector: ({forward_vector_x:.2f}, {forward_vector_y:.2f}).\n"
            f"Navigation command: \"{command_prompt}\".\n"
            "Based on the visual input and state information, predict the next action.\n"
            "Return the action in the following JSON format:\n"
            "{\"throttle\": float, \"steer\": float, \"brake\": float}"
        )
    else:
        image_np = obs['central_rgb']['data']
        prompt = (
            "<image>\n"
            "You are a driving agent in a simulated environment.\n"
            f"Current speed: {speed:.1f} m/s.\n"
            f"Forward vector: ({forward_vector_x:.2f}, {forward_vector_y:.2f}).\n"
            f"Navigation command: \"{command_prompt}\".\n"
            "Based on the visual input and state information, predict the next action.\n"
            "Return the action in the following JSON format:\n"
            "{\"throttle\": float, \"steer\": float, \"brake\": float}"
        )

    if 'rwkv' in model_name_or_path.lower():
        control = infer_rwkv(vlm_model, image_np, prompt, image_processor, tokenizer)
        return control
    elif 'qwen' in model_name_or_path.lower():
        prompt = prompt.replace("<image>\n", "").replace("<image>", "")
        if use_vllm:
            control = infer_vllm(vlm_model, image_np, prompt, processor)
        else:
            control = infer_qwen(vlm_model, image_np, prompt, processor)
        return control
    elif 'llava' in model_name_or_path.lower():
        prompt = prompt.replace("<image>\n", "").replace("<image>", "")
        if use_vllm:
            control = infer_vllm(vlm_model, image_np, prompt, processor)
        else:
            control = infer_llava(vlm_model, image_np, prompt, processor)
        return control
    elif 'intern' in model_name_or_path.lower():
        prompt = prompt.replace("<image>\n", "").replace("<image>", "")
        if use_vllm:
            control = infer_vllm(vlm_model, image_np, prompt, processor)
        else:
            control = infer_internvl(vlm_model, image_np, prompt, processor)
        return control
    else:
        raise NotImplementedError


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python vlm_utils.py <vlm_model_path> <image_path>")
        sys.exit(1)
    vlm_path = sys.argv[1]
    image_path = sys.argv[2]
    vlm_model, processor = load_llava(vlm_path)
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    prompt = (
        "<image>\n"
        "You are a driving agent in a simulated environment.\n"
        f"Current speed: 1 m/s.\n"
        f"Forward vector: (1, 1).\n"
        f"Navigation command: \"{COMMANDS[0]}\".\n"
        "Based on the visual input and state information, predict the next action.\n"
        "Return the action in the following JSON format:\n"
        "{\"throttle\": float, \"steer\": float, \"brake\": float}"
    )
    prompt = prompt.replace("<image>\n", "").replace("<image>", "")

    # image = Image.fromarray(image)
    infer_vllm(vlm_model, image, prompt, processor)


    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "image": Image.fromarray(image),
    #             },
    #             {"type": "text", "text": prompt},
    #         ],
    #     }
    # ]
    #
    # text = processor.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )
    # image_inputs, video_inputs = process_vision_info(messages)
    # print(image_inputs)
    # print(text)
    # inputs = processor(
    #     text=[text],
    #     images=image_inputs,
    #     videos=video_inputs,
    #     padding=True,
    #     return_tensors="pt",
    # )
    # inputs = inputs.to(vlm_model.device)
    # print(inputs.keys())

