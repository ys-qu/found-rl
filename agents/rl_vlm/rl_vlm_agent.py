"""RL+VLM agent: manages policy loading, training, and evaluation with foundation model guidance."""
import logging
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import wandb
import copy

from carla_gym.utils.config_utils import load_entry_point
from utils.os_utils import find_project_root
from stable_baselines3.common.utils import get_schedule_fn

import torch as th
device = 'cuda' if th.cuda.is_available() else 'cpu'


def linear_schedule(initial_value):
    def func(progress_remaining: float) -> float:
        return initial_value * progress_remaining
    return func


def linear_resume_schedule(init_lr, total_planned, num_done):
    """Linear LR schedule that resumes from partial training progress."""
    start = max(0.0, 1.0 - num_done / total_planned)
    def sched(progress_remaining):
        return init_lr * start * progress_remaining
    return sched


class RlVlmAgent():
    def __init__(self, path_to_conf_file='config_agent.yaml', local_output_dir=None, buffer_warmup_dir=None):
        self._logger = logging.getLogger(__name__)
        self._render_dict = None
        self.supervision_dict = None
        self.local_output_dir = local_output_dir
        self.buffer_warmup_dir = buffer_warmup_dir
        self.setup(path_to_conf_file)

    def setup(self, path_to_conf_file):
        cfg = OmegaConf.load(path_to_conf_file)
        print(cfg)

        if cfg.wb_run_path is not None:
            api = wandb.Api()
            run = api.run(cfg.wb_run_path)
            all_ckpts = [f for f in run.files() if 'ckpt' in f.name]

            if cfg.wb_ckpt_step is None:
                f = max(all_ckpts, key=lambda x: int(x.name.split('_')[1].split('.')[0]))
                self._logger.info(f'Resume checkpoint latest {f.name}')
            else:
                wb_ckpt_step = int(cfg.wb_ckpt_step)
                f = min(all_ckpts, key=lambda x: abs(int(x.name.split('_')[1].split('.')[0]) - wb_ckpt_step))
                self._logger.info(f'Resume checkpoint closest to step {wb_ckpt_step}: {f.name}')

            f.download(replace=True)
            run.file('config_agent.yaml').download(replace=True)
            cfg = OmegaConf.load('config_agent.yaml')
            self._buffer_path = self.local_output_dir / 'replay_buffer.pkl'
            self._ckpt = f.name
        else:
            self._ckpt = None
            self._buffer_path  = None

        cfg = OmegaConf.to_container(cfg)

        self._obs_configs = cfg['obs_configs']
        self._train_cfg = cfg['training']

        self._policy = 'MultiInputPolicy'  # 'MultiInputPolicy' is fixed

        self._wrapper_class = load_entry_point(cfg['env_wrapper']['entry_point'])
        self._wrapper_kwargs = cfg['env_wrapper']['kwargs']
        self._wrapper_class.set_birdview_key(self._wrapper_kwargs['birdview_key'])
        self._wrapper_class.set_compress_size(self._wrapper_kwargs['compress_size'])

    def init_eval_policy(self, env, local_ckpt_path=None):
        if 'replay_buffer_class' in self._train_cfg['kwargs']:
            self._train_cfg['kwargs']['replay_buffer_class'] = load_entry_point(
                self._train_cfg['kwargs']['replay_buffer_class'])
            self._train_cfg['kwargs']['buffer_size'] = 1
        elif 'rollout_buffer_class' in self._train_cfg['kwargs']:
            self._train_cfg['kwargs']['rollout_buffer_class'] = load_entry_point(
                self._train_cfg['kwargs']['rollout_buffer_class'])
        else:
            raise TypeError('A buffer class has to be assigned in config/agent/rl_vlm/training/*.yaml')

        if 'policy_kwargs' in self._train_cfg['kwargs'] and 'features_extractor_class' in self._train_cfg['kwargs']['policy_kwargs']:
            self._train_cfg['kwargs']['policy_kwargs']['features_extractor_class']= load_entry_point(
                self._train_cfg['kwargs']['policy_kwargs']['features_extractor_class'])
        model_class = load_entry_point(self._train_cfg['entry_point'])
        model = model_class(self._policy, env, **self._train_cfg['kwargs'])
        if local_ckpt_path is not None:
            model.set_parameters(local_ckpt_path)
            self._logger.info(f'Loading model parameters from: {local_ckpt_path}')
        else:
            model.set_parameters(self._ckpt)
            self._logger.info(f'Loading model parameters from: {self._ckpt}')
        self.eval_policy = model.policy

    def run_step(self, input_data, timestamp):
        input_data = copy.deepcopy(input_data)

        policy_input = input_data

        # actions, _, _ = self.eval_policy.forward(policy_input, deterministic=False)
        actions, _ = self.eval_policy.predict(policy_input, deterministic=True)
        self.supervision_dict = {
            'action': actions,
        }
        self.supervision_dict = copy.deepcopy(self.supervision_dict)

        self._render_dict = {
            'timestamp': timestamp,
            'obs': policy_input,
            'action': actions,
        }
        self._render_dict = copy.deepcopy(self._render_dict)

        return actions

    def reset(self, log_file_path):
        self._logger.handlers = []
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file_path, mode='w')
        fh.setLevel(logging.DEBUG)
        self._logger.addHandler(fh)

    def learn(self, env, total_timesteps, callback,
              log_interval: int = 4,
              tb_log_name: str = "SAC",
              reset_num_timesteps: bool = False,
              progress_bar: bool = False,
              ):
        model_class = load_entry_point(self._train_cfg['entry_point'])
        if 'replay_buffer_class' in self._train_cfg['kwargs']:
            self._train_cfg['kwargs']['replay_buffer_class'] = load_entry_point(
                self._train_cfg['kwargs']['replay_buffer_class'])
        elif 'rollout_buffer_class' in self._train_cfg['kwargs']:
            self._train_cfg['kwargs']['rollout_buffer_class'] = load_entry_point(
                self._train_cfg['kwargs']['rollout_buffer_class'])
        else:
            raise TypeError('A buffer class has to be assigned in config/agent/rl_vlm/training/*.yaml')

        if 'policy_kwargs' in self._train_cfg['kwargs'] and 'features_extractor_class' in self._train_cfg['kwargs']['policy_kwargs']:
            self._train_cfg['kwargs']['policy_kwargs']['features_extractor_class']= load_entry_point(
                self._train_cfg['kwargs']['policy_kwargs']['features_extractor_class'])

        if not self._ckpt and not self._buffer_path:
            lr = self._train_cfg['kwargs'].get("learning_rate", 1e-4)
            self._train_cfg['kwargs']['learning_rate'] = linear_schedule(lr)
            print(self._train_cfg['kwargs'])
            model = model_class(self._policy, env, **self._train_cfg['kwargs'])
            if self.buffer_warmup_dir and hasattr(model, 'replay_buffer'):
                self._logger.info(f'Loading warm-up data from: {self.buffer_warmup_dir}')
                try:
                    model.replay_buffer.warm_up(self.buffer_warmup_dir, env.n_envs)
                except:
                    model.replay_buffer.warm_up(self.buffer_warmup_dir, 1)  # dummy env

        else:
            with open(self.local_output_dir / "num_timesteps.txt", "r") as f:
                num_timesteps = int(f.read())
            lr = self._train_cfg['kwargs'].get("learning_rate", 1e-4)
            self._train_cfg['kwargs']['learning_rate'] = linear_resume_schedule(lr, total_timesteps, num_timesteps)
            self._logger.info(f'Loading model parameters from: {self._ckpt}')
            print(self._train_cfg['kwargs'])
            model = model_class.load(self._ckpt, env=env, **self._train_cfg['kwargs'])
            print("learning_rate  :", model.learning_rate)
            print("num_timesteps  :", model.num_timesteps)
            print('Model Loaded!')
            if self._buffer_path and os.path.exists(self._buffer_path) and hasattr(model, 'load_replay_buffer'):
                self._logger.info(f'Loading replay buffer from: {self._buffer_path}')
                model.load_replay_buffer(self._buffer_path)
            model.num_timesteps = num_timesteps
            self._logger.info(f'Resume at {model.num_timesteps}')

        remaining_steps = total_timesteps - model.num_timesteps

        if remaining_steps > 0:
            model.learn(remaining_steps, callback=callback, log_interval=log_interval,
                        tb_log_name=tb_log_name,
                        reset_num_timesteps=reset_num_timesteps,
                        progress_bar=progress_bar)

    def render(self, rendered, reward_debug, terminal_debug):
        self._render_dict['reward_debug'] = reward_debug
        self._render_dict['terminal_debug'] = terminal_debug
        self._render_dict['im_render'] = rendered

        return self._wrapper_class.im_render(self._render_dict)

    @property
    def obs_configs(self):
        return self._obs_configs
