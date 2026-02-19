"""
Main training entry point for Found-RL. Launches CARLA servers, VLM/CLIP async inference servers,
and trains RL agents with foundation model guidance (VMR, AWAG) or CLIP reward shaping.
"""
import threading
from collections import defaultdict
import warnings, logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

import torch
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gym
from pathlib import Path
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList
from agents.rl_vlm.utils.wandb_callback import WandbCallback
from carla_gym.utils import config_utils
from utils import server_utils
import asyncio
import time
import queue
import torch
import multiprocessing as mp
from functools import partial
from PIL import Image
import re
from agents.rl_vlm.utils.llm_wrapper import LLMWrapper, OpenCLIPWrapper
import numpy as np
import torch.nn.functional as F


log = logging.getLogger(__name__)


def tolerant_parse_vlm_output(output_text):
    """Parse throttle, steer, brake from VLM text output. Returns dict or None if parsing fails."""
    numbers = re.findall(r'-?\d+(?:\.\d+)?', output_text)

    if len(numbers) >= 3:
        try:
            throttle, steer, brake = map(float, numbers[:3])
            return {"throttle": np.clip(throttle, 0, 1), "steer": np.clip(steer, -1, 1), "brake": np.clip(brake, 0, 1)}
        except:
            pass
    return None


async def llm_server_loop(request_queue, shared_output_dict, model_path, batch_size=64, wait_time=10, ready_event=None):
    """Async VLM server loop. Batches requests and returns control dicts via shared_output_dict."""
    from transformers.utils import logging
    logging.set_verbosity_error()
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(model_path, device_map="cuda:0",
                                                        torch_dtype="float16", low_cpu_mem_usage=True,
                                                        attn_implementation="sdpa")

    if ready_event is not None:
        ready_event.set()

    while True:
        batch = []
        start = time.time()
        while len(batch) < batch_size or (time.time() - start) < wait_time:
            try:
                req = request_queue.get_nowait()
                batch.append(req)
            except queue.Empty:
                await asyncio.sleep(0.01)

        if not batch:
            continue

        messages = []
        for sample in batch:
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.fromarray(sample.image).convert("RGB"),  # central_rgb: np.array(H, W, 3)
                    },
                    {"type": "text", "text": sample.prompt},
                ]
            }
            messages.append(message)

        text_list = [
            processor.apply_chat_template(
                [message],
                tokenize=False,
                add_generation_prompt=True
            ) for message in messages
        ]
        image_list = [message["content"][0]["image"] for message in messages]
        inputs = processor(
            text=text_list,
            images=image_list,  # MUST be list of PIL.Image
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device=model.device, dtype=model.dtype) if torch.is_floating_point(v) else v.to(model.device)
                  for k, v in inputs.items()}

        generated_ids = model.generate(**inputs, max_new_tokens=32)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        for r, output_text in zip(batch, output_texts):
            control_dict = tolerant_parse_vlm_output(output_text)
            if control_dict:
                control_dict["id"] = r.id
            else:
                continue
            shared_output_dict.setdefault(r.env_index, manager.list()).append(control_dict)


class RunningMeanStd:
    """Running mean and variance for normalization of CLIP bonus scores."""

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
            x = np.array([x])

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = 1 if np.isscalar(x) else len(x)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count


async def open_clip_server_loop(request_queue, shared_output_dict, model_path=None, ready_event=None,
                                model_arch="ViT-B-16",
                                batch_size=32, wait_time=0.01,
                                reward_scale=0.25):
    import open_clip
    device = "cuda:0"
    print(f"[OpenCLIP Server] Initializing {model_arch}...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_arch,
        pretrained=model_path if model_path else "laion2b_s34b_b88k",
        device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_arch)
    clip_rms = RunningMeanStd(shape=())

    long_opts = ["braking hard", "braking", "accelerating fast", "accelerating", "accelerating gently", "idling"]
    lat_opts = ["going straight", "turning right sharply", "turning right", "turning left sharply", "turning left"]

    long_neighbors = {
        "braking hard": ["braking"], "braking": ["braking hard"],
        "accelerating fast": ["accelerating"], "accelerating": ["accelerating fast", "accelerating gently"],
        "accelerating gently": ["accelerating"], "idling": []
    }
    lat_neighbors = {
        "turning right sharply": ["turning right"], "turning right": ["turning right sharply"],
        "turning left sharply": ["turning left"], "turning left": ["turning left sharply"],
        "going straight": []
    }

    actions_flat = []
    action_str_to_idx = {}
    idx_counter = 0
    for lon in long_opts:
        for lat in lat_opts:
            act_str = f"{lon} and {lat}"
            actions_flat.append(act_str)
            action_str_to_idx[act_str] = idx_counter
            idx_counter += 1

    neighbor_mask_matrix = torch.ones((30, 30), dtype=torch.bool, device=device)

    for i, target_str in enumerate(actions_flat):
        parts = target_str.split(" and ")
        t_lon, t_lat = parts[0], parts[1]
        valid_lons = long_neighbors.get(t_lon, []) + [t_lon]
        valid_lats = lat_neighbors.get(t_lat, []) + [t_lat]

        for n_lon in valid_lons:
            for n_lat in valid_lats:
                neighbor_str = f"{n_lon} and {n_lat}"
                if neighbor_str in action_str_to_idx:
                    n_idx = action_str_to_idx[neighbor_str]
                    neighbor_mask_matrix[i, n_idx] = False

    commands = [
        'turn left at the intersection', 'turn right at the intersection', 'go straight at the intersection',
        'follow the current lane', 'change to the left lane', 'change to the right lane'
    ]
    speed_bins = [
        "The car is currently stopped", "The car is moving slowly",
        "The car is driving at a moderate speed", "The car is driving at a high speed"
    ]

    all_prompts = []
    for cmd in commands:
        for spd in speed_bins:
            for lon in long_opts:
                for lat in lat_opts:
                    prompt = f"Command is to {cmd}. {spd}. Action behavior: The car is {lon} and {lat}."
                    all_prompts.append(prompt)

    with torch.no_grad():
        all_tokens = tokenizer(all_prompts).to(device)
        all_feats = model.encode_text(all_tokens)
        all_feats /= all_feats.norm(dim=-1, keepdim=True)
        # Reshape: [Command(6), Speed(4), Action(30), Dim(512)]
        structured_anchors = all_feats.view(len(commands), len(speed_bins), 30, -1)

    logit_scale = model.logit_scale.exp().item()
    print(f"[OpenCLIP Server] Ready! Anchors: {structured_anchors.shape}")

    if ready_event is not None:
        ready_event.set()

    while True:
        batch = []
        start = time.time()
        while len(batch) < batch_size or (time.time() - start) < wait_time:
            try:
                req = request_queue.get_nowait()
                batch.append(req)
            except queue.Empty:
                await asyncio.sleep(0.001)

        if not batch:
            await asyncio.sleep(0.005)
            continue

        image_list = [Image.fromarray(s.image).convert("RGB") for s in batch]
        inputs_img = torch.stack([preprocess(img) for img in image_list]).to(device)

        with torch.no_grad():
            img_feats = model.encode_image(inputs_img)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)

            final_scores = []

            for i, req in enumerate(batch):
                c_idx, s_idx = req.command_idx, req.speed_idx
                if c_idx >= 6 or c_idx < 0: c_idx = 3
                if s_idx >= 4 or s_idx < 0: s_idx = 0

                context_anchors = structured_anchors[c_idx, s_idx]
                logits = (img_feats[i].unsqueeze(0) @ context_anchors.T) * logit_scale
                probs = torch.softmax(logits, dim=-1)[0]

                act_idx = action_str_to_idx.get(req.action_desc)

                if act_idx is not None:
                    pos_score = probs[act_idx].item()
                    neg_mask = neighbor_mask_matrix[act_idx]
                    if neg_mask.any():
                        max_neg_score = probs[neg_mask].max().item()
                    else:
                        max_neg_score = 0.0

                    margin = max(0.0, pos_score - max_neg_score)
                    raw_bonus = margin
                else:
                    raw_bonus = 0.0

                clip_rms.update(raw_bonus)
                norm_bonus = (raw_bonus - clip_rms.mean) / np.sqrt(clip_rms.var + 1e-8)
                norm_bonus = np.clip(norm_bonus, -1.0, 1.0)
                final_bonus = norm_bonus * reward_scale

                final_scores.append({'final_bonus': final_bonus,
                 'raw_bonus': raw_bonus,
                 'clip_rms_mean': clip_rms.mean,
                 'clip_rms_std': np.sqrt(clip_rms.var + 1e-8)})

            for r, res in zip(batch, final_scores):
                idx = r.env_index
                result_data = {"id": r.id,
                               "score": float(res['final_bonus']),
                               "raw_bonus": float(res['raw_bonus']),
                               "clip_rms_mean": float(res['clip_rms_mean']),
                               "clip_rms_std": float(res['clip_rms_std'])}

                if idx in shared_output_dict:
                    l = shared_output_dict[idx]
                    l.append(result_data)
                    shared_output_dict[idx] = l
                else:
                    shared_output_dict[idx] = [result_data]


@hydra.main(config_path='config', config_name='train_rl_asycn_vlm')
def main(cfg: DictConfig):
    if cfg.kill_running:
        server_utils.kill_carla()
    set_random_seed(cfg.seed, using_cuda=True)

    if cfg.start_llm_server:
        mp.set_start_method("spawn", force=True)

    print('start carla servers...')
    server_manager = server_utils.CarlaServerManager(cfg.carla_sh_path, configs=cfg.test_suites, port=cfg.port)
    server_manager.start()
    print('carla server launched...')

    if cfg.start_llm_server:
        request_queue = manager.Queue()
        shared_output_dict = manager.dict({i: manager.list() for i in range(len(server_manager.env_configs))})
    if cfg.start_clip_server:
        request_queue_clip = manager.Queue()
        shared_output_dict_clip = manager.dict({i: manager.list() for i in range(len(server_manager.env_configs))})

    agent_name = cfg.actors[cfg.ev_id].agent

    last_checkpoint_path = Path(hydra.utils.get_original_cwd()) / 'outputs' / 'checkpoint.txt'
    if last_checkpoint_path.exists():
        with open(last_checkpoint_path, 'r') as f:
            cfg.agent[agent_name].wb_run_path = f.read()

    OmegaConf.save(config=cfg.agent[agent_name], f='config_agent.yaml')

    local_output_dir = Path(hydra.utils.get_original_cwd()) / 'outputs'
    AgentClass = config_utils.load_entry_point(cfg.agent[agent_name].entry_point)
    buffer_warmup_dir = ''
    agent = AgentClass('config_agent.yaml', local_output_dir=local_output_dir, buffer_warmup_dir=buffer_warmup_dir)
    cfg_agent = OmegaConf.load('config_agent.yaml')

    obs_configs = {cfg.ev_id: OmegaConf.to_container(cfg_agent.obs_configs)}
    reward_configs = {cfg.ev_id: OmegaConf.to_container(cfg.actors[cfg.ev_id].reward)}
    terminal_configs = {cfg.ev_id: OmegaConf.to_container(cfg.actors[cfg.ev_id].terminal)}

    # env wrapper
    EnvWrapper = config_utils.load_entry_point(cfg_agent.env_wrapper.entry_point)
    wrapper_kargs = cfg_agent.env_wrapper.kwargs
    EnvWrapper.set_birdview_key(wrapper_kargs['birdview_key'])
    EnvWrapper.set_compress_size(wrapper_kargs['compress_size'])

    config_utils.check_h5_maps(cfg.test_suites, obs_configs, cfg.carla_sh_path)

    def env_maker(config, env_index=0, light_always_green=False, disbale_bg_actors=False):
        log.info(f'making port {config["port"]}')
        config['env_configs'] = {**config['env_configs'], 'light_always_green': light_always_green, 'disbale_bg_actors': disbale_bg_actors}
        env = gym.make(config['env_id'], obs_configs=obs_configs, reward_configs=reward_configs,
                       terminal_configs=terminal_configs, host='localhost', port=config['port'],
                       seed=cfg.seed, no_rendering=True, **config['env_configs'])
        env = EnvWrapper(env, **wrapper_kargs)
        if cfg.start_clip_server:
            env = OpenCLIPWrapper(env, env_index, request_queue_clip, shared_output_dict_clip)
        if cfg.start_llm_server:
            env = LLMWrapper(env, env_index, request_queue, shared_output_dict)
        return env

    if cfg.dummy or len(server_manager.env_configs) == 1:
        env = DummyVecEnv([lambda config=config: env_maker(config, light_always_green=cfg.light_always_green, disbale_bg_actors=cfg.disbale_bg_actors) for config in server_manager.env_configs])
    else:
        env = SubprocVecEnv([
            partial(env_maker, config, env_index, light_always_green=cfg.light_always_green)
            for env_index, config in enumerate(server_manager.env_configs)
        ])

    if cfg.start_llm_server:
        print('[INFO] Launching async llm server...')
        llm_ready = threading.Event()
        def run_async_llm():
            asyncio.run(llm_server_loop(request_queue, shared_output_dict, cfg.llm_model_path, ready_event=llm_ready))
        thread = threading.Thread(target=run_async_llm)
        thread.daemon = True
        thread.start()
        llm_ready.wait()
        print('[INFO] llm server is ready.')

    if cfg.start_clip_server:
        print('[INFO] Launching async clip server...')
        clip_ready = threading.Event()
        def run_async_clip():
            asyncio.run(open_clip_server_loop(request_queue_clip, shared_output_dict_clip, cfg.clip_model_path, ready_event=clip_ready))
        thread_clip = threading.Thread(target=run_async_clip)
        thread_clip.daemon = True
        thread_clip.start()
        clip_ready.wait()
        print('[INFO] clip server is ready.')

    wb_callback = WandbCallback(cfg, env, local_output_dir=local_output_dir)
    callback = CallbackList([wb_callback])

    with open(last_checkpoint_path, 'w') as f:
        f.write(wandb.run.path)

    agent.learn(env, total_timesteps=cfg.total_timesteps, callback=callback)

    server_manager.stop()


if __name__ == '__main__':
    manager = mp.Manager()
    main()
    log.info("train_rl.py DONE!")
