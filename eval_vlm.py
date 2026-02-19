"""VLM-only evaluation: run vision-language models as driving agents without RL training."""
import torch
from carla import VehicleControl

"""
    If you are using old gpus, please enable following instructions:
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
"""
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
from transformers.utils import logging
logging.set_verbosity_error()
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
from agents.rl_vlm.utils.wandb_callback import ImageEncoder

from carla_gym.utils import config_utils
from carla_gym.utils.expert_noiser import ExpertNoiser
from utils import saving_utils, server_utils
from agents.rl_vlm.utils.wandb_callback import WandbCallback
# from agents.rl_birdview.utils.rl_birdview_wrapper import RlBirdviewWrapper
import carla_gym.utils.transforms as trans_utils
import carla_gym.core.task_actor.common.navigation.route_manipulation as gps_util
import uuid
import torch
from qwen_vl_utils import process_vision_info
from pytorch_lightning.utilities import rank_zero_info
import carla
from PIL import Image



log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

COMMANDS = [
    'turn left at the intersection',
    'turn right at the intersection',
    'go straight at the intersection',
    'follow the current lane',
    'change to the left lane',
    'change to the right lane'
]


def eval_single(run_name, env, driver_log_dir, log_video,
                   vlm_model=None, image_processor=None, tokenizer=None,
                   processor=None, model_name_or_path=None):

    list_debug_render = []
    list_data_render = []
    ep_stat_dict = {}
    ep_event_dict = {}
    actor_id = 'hero'
    log_dir = driver_log_dir / actor_id
    log_dir.mkdir(parents=True, exist_ok=True)

    obs = env.reset()
    timestamp = env.timestamp
    done = {'__all__': False}
    valid = True
    while not done['__all__']:
        driver_control = {}
        driver_supervision = {}

        vlm_agent_action = generate_vlm_agent_action(vlm_model, model_name_or_path, obs,
                                    image_processor=image_processor, tokenizer=tokenizer,
                                    processor=processor)
        driver_control[actor_id] = vlm_agent_action

        new_obs, reward, done, info = env.step(driver_control)

        obs = new_obs

        debug_imgs = []
        if done[actor_id] and (actor_id not in ep_stat_dict):
            episode_stat = info[actor_id]['episode_stat']
            ep_stat_dict[actor_id] = episode_stat
            ep_event_dict[actor_id] = info[actor_id]['episode_event']

        timestamp = env.timestamp

    return list_debug_render, ep_stat_dict, ep_event_dict, timestamp


def load_vlm(vlm_path_or_url, args=None, model_device='cuda'):
    if 'rwkv' in vlm_path_or_url:
        try:
            from rwkv_v7.src.model import VisualRWKV
            from rwkv_v7.src.utils import Conversation
            from rwkv_v7.src.dataset import process_image_tokens_in_conversations, preprocess
            from rwkv_v7.src.dataset import DEFAULT_IMAGE_TOKEN, DEFAULT_STOP_TOKEN, STOP_TOKEN_INDEX
            from rwkv_v7.tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
            from rwkv_v7.src.config import VISION_TOWER_CHECKPOINT_NAMES
        except:
            raise FileNotFoundError('If you want to use visual rwkv, you should download source codes first!')
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
    elif 'qwen' in vlm_path_or_url.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_path_or_url, torch_dtype="float16", device_map="cuda"
        )
        processor = AutoProcessor.from_pretrained(vlm_path_or_url)
        return model, processor
    elif 'llava' in vlm_path_or_url.lower():
        from transformers import LlavaForConditionalGeneration, LlavaProcessor
        model = LlavaForConditionalGeneration.from_pretrained(
            vlm_path_or_url,
            torch_dtype="float16",
            device_map="auto"
        )
        processor = LlavaProcessor.from_pretrained(vlm_path_or_url)
        return model, processor
    elif 'intern' in vlm_path_or_url.lower():
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(vlm_path_or_url)
        if 'full' in vlm_path_or_url:
            model = AutoModelForImageTextToText.from_pretrained(vlm_path_or_url, device_map="cuda",
                                                                torch_dtype="float16")
        else:
            model = AutoModelForImageTextToText.from_pretrained(vlm_path_or_url, device_map="cuda",
                                                                torch_dtype="auto")
        return model, processor
    else:
        raise NotImplementedError


def generate_vlm_agent_action(vlm_model, model_name_or_path, obs_, image_processor=None, tokenizer=None, processor=None,
                              ctx_len=2048, num_token_per_image=1024, device='cuda'):
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

    if 'bev' in model_name_or_path.lower() or 'birdview' in model_name_or_path.lower():
        central_rgb = obs['birdview']['rendered'].transpose(1, 2, 0)
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
        central_rgb = obs['central_rgb']['data']
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

        messages = {
            "id": uuid.uuid4().hex[:16],
            "image": Image.fromarray(central_rgb),
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
        idx = get_question_id(messages)
        image_dict = get_single_image_dict(messages, image_processor)
        for k in image_dict:
            image_dict[k] = image_dict[k].bfloat16().to(device)
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
    elif 'qwen' in model_name_or_path.lower():
        prompt = prompt.replace("<image>\n", "").replace("<image>", "")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.fromarray(central_rgb),
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
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
        try:
            control_dict = json.loads(output_text[0])
            control = carla.VehicleControl(
                throttle=control_dict.get("throttle", 0.0),
                steer=control_dict.get("steer", 0.0),
                brake=control_dict.get("brake", 0.0)
            )
        except Exception as e:
            print(f"Error parsing JSON: {e}, input str is {output_text[0]}")
            control = carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=0.0
            )
        return control
    elif 'llava' in model_name_or_path.lower():
        prompt = prompt.replace("<image>\n", "").replace("<image>", "")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.fromarray(central_rgb).convert("RGB"),
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
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        generated_ids = vlm_model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        try:
            control_dict = json.loads(output_text[0])
            control = carla.VehicleControl(
                throttle=control_dict.get("throttle", 0.0),
                steer=control_dict.get("steer", 0.0),
                brake=control_dict.get("brake", 0.0)
            )
        except Exception as e:
            print(f"Error parsing JSON: {e}, input str is {output_text[0]}")
            control = carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=0.0
            )
        return control

    elif 'intern' in model_name_or_path.lower():
        prompt = prompt.replace("<image>\n", "").replace("<image>", "")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.fromarray(central_rgb).convert("RGB"),
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
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to("cuda")

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
    else:
        raise NotImplementedError

    return {'throttle': 0, 'steer': 0, 'brake': 0}


@ hydra.main(config_path='config', config_name='eval_vlm')
def main(cfg: DictConfig):
    if cfg.host == 'localhost' and cfg.kill_running:
        server_utils.kill_carla()
    log.setLevel(getattr(logging, cfg.log_level.upper()))

    server_manager = server_utils.CarlaServerManager(cfg.carla_sh_path, port=cfg.port)
    server_manager.start()

    # single actor, place holder for multi actors
    obs_configs = {}
    reward_configs = {}
    terminal_configs = {}
    for ev_id, ev_cfg in cfg.actors.items():  # ev_id: hero
        cfg_driver = cfg.agent[ev_cfg.driver]  # ev_cfg.driver: ppo  --> cfg_driver: ppo agent config
        OmegaConf.save(config=cfg_driver, f='config_driver.yaml')
        obs_configs[ev_id] = cfg_driver['obs_configs']

        reward_configs[ev_id] = OmegaConf.to_container(ev_cfg.reward)
        terminal_configs[ev_id] = OmegaConf.to_container(ev_cfg.terminal)

    config_utils.check_h5_maps(cfg.test_suites, obs_configs, cfg.carla_sh_path)

    last_checkpoint_path = f'{hydra.utils.get_original_cwd()}/outputs/checkpoint.txt'
    if cfg.resume and os.path.isfile(last_checkpoint_path):
        with open(last_checkpoint_path, 'r') as f:
            env_idx = int(f.read())
    else:
        env_idx = 0

    ep_state_buffer_json = f'{hydra.utils.get_original_cwd()}/outputs/ep_stat_buffer_{env_idx}.json'
    if cfg.resume and os.path.isfile(ep_state_buffer_json):
        ep_stat_buffer = json.load(open(ep_state_buffer_json, 'r'))
        ckpt_task_idx = len(ep_stat_buffer['hero'])
    else:
        ckpt_task_idx = 0
        ep_stat_buffer = {}
        ep_stat_buffer['hero'] = []

    wb_checkpoint_path = f'{hydra.utils.get_original_cwd()}/outputs/wb_run_id.txt'
    if cfg.resume and os.path.isfile(wb_checkpoint_path):
        with open(wb_checkpoint_path, 'r') as f:
            wb_run_id = f.read()
    else:
        wb_run_id = None

    log.info(f'Start from env_idx: {env_idx}, task_idx {ckpt_task_idx}')

    diags_dir = Path('diagnostics')
    driver_log_dir = Path('driver_log')
    video_dir = Path('videos')
    diags_dir.mkdir(parents=True, exist_ok=True)
    driver_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    # init wandb
    wandb.init(project=cfg.wb_project, name=cfg.wb_name, group=cfg.wb_group, notes=cfg.wb_notes, tags=cfg.wb_tags,
               id=wb_run_id, resume="allow")
    wandb.config.update(OmegaConf.to_container(cfg))
    wandb.save('./config_agent.yaml')
    with open(wb_checkpoint_path, 'w') as f:
        f.write(wandb.run.id)

    if env_idx >= len(cfg.test_suites):
        log.info(f'Finished! env_idx: {env_idx}, resave to wandb')
        return

    print('========================Making env========================')
    env_setup = OmegaConf.to_container(cfg.test_suites[env_idx])
    env = gym.make(env_setup['env_id'], obs_configs=obs_configs, reward_configs=reward_configs,
                   terminal_configs=terminal_configs, host=cfg.host, port=cfg.port,
                   seed=cfg.seed, no_rendering=cfg.no_rendering, light_always_green=cfg.light_always_green,
                    disbale_bg_actors=cfg.disbale_bg_actors, **env_setup['env_configs'])

    print('========================Loading vlm========================')
    model_name_or_path = cfg.model_name_or_path
    if 'rwkv' in model_name_or_path:
        class RwkvConfig:
            def __init__(self, cfg: dict):
                self.__dict__.update(cfg)
        args_rwkv = {
            'model_path': model_name_or_path,
            'vision_tower_dir': f'{PROJECT_ROOT}/rwkv_v7/huggingface_models/',
            'load_model': "",
            "n_layer": 12,
            "n_embd": 768,  # 2048 for 1B5
            "ctx_len": 2048,
            "num_token_per_image": 1024,
            "proj_type": "mlp",  # or "linear"
            "freeze_rwkv": 1,
            "freeze_proj": 1,
            "vocab_size": 65536,
            "dropout": 0,
            "grad_cp": 0,
            "weight_decay": 0,
            "head_size_a": 64,
            "head_size_divisor": 8,
            "pre_ffn": 0,
            "dim_att": 0,
            "dim_ffn": 0,
        }
        args_rwkv = RwkvConfig(args_rwkv)
        if args_rwkv.dim_att <= 0:
            args_rwkv.dim_att = args_rwkv.n_embd
        if args_rwkv.dim_ffn <= 0:
            args_rwkv.dim_ffn = int((args_rwkv.n_embd * 3.5) // 32 * 32)  # default = 3.5x emb size
        os.environ["RWKV_JIT_ON"] = "1"
        os.environ["RWKV_CTXLEN"] = str(args_rwkv.ctx_len)
        os.environ["RWKV_HEAD_SIZE_A"] = str(args_rwkv.head_size_a)
        vlm_model, args_rwkv = load_vlm(model_name_or_path, args=args_rwkv)
        processor = None
        image_processor = args_rwkv.image_processor
        tokenizer = args_rwkv.tokenizer
    else:
        vlm_model, processor = load_vlm(model_name_or_path)
        image_processor, tokenizer = None, None

    n_episodes_per_env = math.ceil(cfg.n_episodes/len(cfg.test_suites))

    for task_idx in range(ckpt_task_idx, n_episodes_per_env):
        idx_episode = task_idx + n_episodes_per_env * env_idx
        run_name = f'{idx_episode:04}'

        env.set_task_idx(task_idx % env.num_tasks)
        log.info(f'Start episode {run_name}, {env_setup}')
        list_debug_render, ep_stat_dict, ep_event_dict, timestamp = eval_single(
            run_name, env, driver_log_dir, cfg.log_video,
            vlm_model=vlm_model, image_processor=image_processor,
            tokenizer=tokenizer, processor=processor, model_name_or_path=model_name_or_path)

        if cfg.log_video:
            debug_video_path = (video_dir / f'debug_{run_name}.mp4').as_posix()
            encoder = ImageEncoder(debug_video_path, list_debug_render[0].shape, 30)
            for im in list_debug_render:
                encoder.capture_frame(im)
            encoder.close()
            wandb.log({f'video/debug_{run_name}': wandb.Video(debug_video_path)}, step=idx_episode)
            encoder = None

        diags_json_path = (diags_dir / f'{run_name}.json').as_posix()
        with open(diags_json_path, 'w') as fd:
            json.dump(ep_event_dict, fd, indent=4, sort_keys=False)

        wandb.save(diags_json_path)
        wandb.save(f'{driver_log_dir.as_posix()}/*/*')

        wandb.log({'time/total_step': timestamp['step'],
                   'time/simulation_time': timestamp['simulation_time'],
                   'time/fps': timestamp['step'] / timestamp['simulation_time'],
                   }, step=idx_episode)

        # save statistics
        for actor_id, ep_stat in ep_stat_dict.items():
            ep_stat_buffer[actor_id].append(ep_stat)
            log_dict = {}
            for k, v in ep_stat.items():
                k_actor = f'{actor_id}/{k}'
                log_dict[k_actor] = v
            wandb.log(log_dict, step=idx_episode)

        with open(ep_state_buffer_json, 'w') as fd:
            json.dump(ep_stat_buffer, fd, indent=4, sort_keys=True)
        list_debug_render.clear()
        ep_stat_dict = None
        ep_event_dict = None

    # close env
    env.close()
    env = None
    server_manager.stop()

    table_data = []
    ep_stat_keys = None
    for actor_id, list_ep_stat in json.load(open(ep_state_buffer_json, 'r')).items():
        avg_ep_stat = WandbCallback.get_avg_ep_stat(list_ep_stat)
        data = [actor_id, cfg.actors[actor_id].driver, env_idx, str(len(list_ep_stat))]
        if ep_stat_keys is None:
            ep_stat_keys = list(avg_ep_stat.keys())
        data += [f'{avg_ep_stat[k]:.4f}' for k in ep_stat_keys]
        table_data.append(data)

    table_columns = ['actor_id', 'driver', 'env_idx',  'n_episode'] + ep_stat_keys
    wandb.log({f'table/summary_{env_idx}': wandb.Table(data=table_data, columns=table_columns)})

    with open(last_checkpoint_path, 'w') as f:
        f.write(f'{env_idx+1}')

    log.info(f"Finished eval vlm env_idx {env_idx}, {env_setup['env_id']}.")
    if env_idx+1 == len(cfg.test_suites):
        log.info(f"Finished, {env_idx+1}/{len(cfg.test_suites)}")
        return
    else:
        log.info(f"Not finished, {env_idx+1}/{len(cfg.test_suites)}")
        sys.exit(1)

    return


if __name__ == '__main__':
    main()
    log.info("eval_vlm.py DONE!")
