import glob
import random
import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, Optional, Union, Dict, List, NamedTuple

import cv2
import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize
from tqdm import tqdm

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env.base_vec_env import tile_images
from agents.rl_vlm.models.torch_util import convert_jump_gap_to_actions

COLORS = [
    [46, 52, 54],
    [136, 138, 133],
    [255, 0, 255],
    [0, 255, 255],
    [0, 0, 255],
    [255, 0, 0],
    [255, 255, 0],
    [255, 255, 255]
]

from stable_baselines3.common.type_aliases import ReplayBufferSamples


class ReplayBufferSamplesVlm(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    vlm_actions: th.Tensor
    has_vlm_actions: th.Tensor
    discounts: Optional[th.Tensor] = None


class DictReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    next_observations: TensorDict
    dones: th.Tensor
    rewards: th.Tensor
    discounts: Optional[th.Tensor] = None
    env_indices: Optional[th.Tensor] = None


class DictReplayBufferSamplesVlm(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    next_observations: TensorDict
    dones: th.Tensor
    rewards: th.Tensor
    vlm_actions: Optional[th.Tensor] = None
    has_vlm_actions: Optional[th.Tensor] = None
    discounts: Optional[th.Tensor] = None
    env_indices: Optional[th.Tensor] = None
    clip_safety_scores: Optional[th.Tensor] = None
    costs: Optional[th.Tensor] = None


class ReplayBufferVlm(ReplayBuffer):
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = True,
            handle_timeout_termination: bool = True,
            use_vlm: bool = False
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs,
                         optimize_memory_usage=optimize_memory_usage,
                         handle_timeout_termination=handle_timeout_termination)

        self.use_vlm = use_vlm

        self.reward_debugs = [[] for i in range(self.n_envs)]
        self.terminal_debugs = [[] for i in range(self.n_envs)]

        if self.use_vlm:
            self.vlm_actions = np.zeros((self.buffer_size, self.n_envs) + self.action_space.shape, dtype=np.float32)
            self.has_vlm_actions = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
            self.vlm_pos = [0 for _ in range(self.n_envs)]

        if psutil is not None:
            obs_nbytes = self.observations.nbytes

            total_memory_usage = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if not self.optimize_memory_usage:
                if self.next_observations is not None:
                    total_memory_usage += self.next_observations.nbytes

            print(f"replay buffer {total_memory_usage / 1_073_741_824:.2f}GB")

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: list[dict[str, Any]],
    ) -> None:
        # Hard-coded, to make sure the saved obs is unit8
        obs = obs.astype(np.uint8)
        next_obs = next_obs.astype(np.uint8)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.use_vlm:
            for i in range(self.n_envs):
                control_dict_list = infos[i]['vlm_actions']
                if control_dict_list:
                    jump_gap_list = infos[i]['jump_gap_list']
                    limit = min(len(control_dict_list), len(jump_gap_list))
                    jump_gap_list = jump_gap_list[:limit]
                    control_dict_list = control_dict_list[:limit]
                    has_vlm_actions = np.array(convert_jump_gap_to_actions(jump_gap_list), dtype=np.int32)
                    self.has_vlm_actions[self.vlm_pos[i]: self.vlm_pos[i] + len(has_vlm_actions), i] = \
                        has_vlm_actions
                    for idx in range(len(control_dict_list)):
                        control_dict = control_dict_list[idx]
                        jump_gap = jump_gap_list[idx]
                        vlm_action = np.zeros(self.action_space.shape)  # (2, )
                        if len(vlm_action) == 2:
                            if control_dict['throttle'] > 0:
                                acc = control_dict['throttle']
                            elif control_dict['brake'] > 0:
                                acc = -abs(control_dict['brake'])
                            else:
                                acc = 0.
                            vlm_action[0] = acc
                            vlm_action[1] = control_dict['steer']
                        else:
                            vlm_action[0] = control_dict['throttle']
                            vlm_action[1] = control_dict['steer']
                            vlm_action[2] = control_dict['brake']
                        pos = int(self.vlm_pos[i] % self.buffer_size)
                        self.vlm_actions[pos, i] = vlm_action
                        self.has_vlm_actions[pos, i] = 1
                        advance = int(max(1, int(jump_gap) if jump_gap is not None else 1))
                        self.vlm_pos[i] = (pos + advance) % self.buffer_size

        for i in range(self.n_envs):
            self.reward_debugs[i].append(infos[i]['reward_debug']['debug_texts'])
            self.terminal_debugs[i].append(infos[i]['terminal_debug']['debug_texts'])

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
            self.reward_debugs = [[] for i in range(self.n_envs)]
            self.terminal_debugs = [[] for i in range(self.n_envs)]

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None) -> ReplayBufferSamples | ReplayBufferSamplesVlm:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        obs = self._normalize_obs(self.observations[batch_inds, env_indices, :])

        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self.next_observations[batch_inds, env_indices, :]

        if not self.use_vlm:
            data = (
                obs,
                self.actions[batch_inds, env_indices, :],
                next_obs,
                # Only use dones that are not due to timeouts
                # deactivated by default (timeouts is initialized as an array of False)
                (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
                self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            )
            return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
        else:
            data = (
                obs,
                self.actions[batch_inds, env_indices, :],
                next_obs,
                (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
                self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
                self.vlm_actions[batch_inds, env_indices],
                self.has_vlm_actions[batch_inds, env_indices],
            )
            return ReplayBufferSamplesVlm(*tuple(map(self.to_torch, data)))

    def render(self):
        list_render = []

        _, _, c, h, w = self.observations.shape
        if c == 15:
            vis_idx = np.array([0, 1, 2, 6, 10, 14])
        elif c == 12:
            vis_idx = np.array([0, 1, 2, 5, 8, 11])
        elif c == 9:
            vis_idx = np.array([0, 1, 2, 4, 6, 8])
        elif c == 6:
            vis_idx = np.array([0, 1, 2, 3, 4, 5])
        elif c == 16:
            vis_idx = np.array([0, 1, 2, 7, 11, 15, 3])
        else:
            raise NotImplementedError('history_idx [] should has at least one idx, for example, -1')

        lookback = 300
        for i in [(self.pos - lookback + i) % self.buffer_size for i in range(lookback)]:  # circular buffer
            im_envs = []
            for j in range(self.n_envs):

                masks = self.observations[i, j, vis_idx, :, :] > 100

                im_birdview = np.zeros([h, w, 3], dtype=np.uint8)
                for idx_c in range(len(vis_idx)):
                    im_birdview[masks[idx_c]] = COLORS[idx_c]

                im = np.zeros([h * 2, w * 2, 3], dtype=np.uint8)  # Double the height to fit all debug texts
                im[:h, :w] = im_birdview

                action_str = np.array2string(self.actions[i, j], precision=1, separator=',', suppress_small=True)

                reward = self.rewards[i, j]
                done = int(self.dones[i, j])

                txt_1 = f'a{action_str}'
                im = cv2.putText(im, txt_1, (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                txt_2 = f'{done}'
                im = cv2.putText(im, txt_2, (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                txt_3 = f'rw:{reward:5.2f}'
                im = cv2.putText(im, txt_3, (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

                try:
                    for i_txt, txt in enumerate(self.reward_debugs[j][i] + self.terminal_debugs[j][i]):
                        # Position debug texts in the right half with more vertical space
                        im = cv2.putText(im, txt, (w, (i_txt + 1) * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                         (255, 255, 255), 1)
                except:
                    pass
                im_envs.append(im)

            big_im = tile_images(im_envs)
            list_render.append(big_im)

        return list_render


class DictReplayBufferVlm(ReplayBuffer):
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
            use_vlm: str = 'pvp',  # 'awac', ?pvp?, 'r_clip', 'r_sim', 'none'/None
            use_prioritized: bool = False,
            n_steps: int = 1,
            gamma: float = 0.99
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        print('[INFO] Using replay buffer..')
        print(f'[INFO] observation_space: {observation_space}')
        print(f'[INFO] action_space: {action_space}')

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.use_vlm = use_vlm
        # self.use_vlm = 'none'

        # N-step return configuration
        self.n_steps = n_steps
        self.gamma = gamma

        self.reward_debugs = [[] for i in range(self.n_envs)]
        self.terminal_debugs = [[] for i in range(self.n_envs)]

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.use_n_step = n_steps > 1

        if self.use_n_step:
            print(f"[INFO] N-step returns enabled: n_steps={n_steps}, gamma={gamma}")
            if self.optimize_memory_usage:
                raise NotImplementedError("NStepReplayBuffer doesn't support optimize_memory_usage=True")

        self.observations = {
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        if 'pvp' in self.use_vlm or 'awac' in self.use_vlm:
            self.vlm_actions = np.zeros((self.buffer_size, self.n_envs) + self.action_space.shape, dtype=np.float32)
            self.has_vlm_actions = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
        if 'r_clip' in self.use_vlm:
            # why negative 1, the safety score is between 0-1, so why we get -1, we know that this score is invalid
            self.clip_safety_scores = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32) - 10
            # just for recording
            self.clip_raw_bonus = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32) - 10
            self.clip_rms_mean = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32) - 10
            self.clip_rms_std = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32) - 10
        if 'lag' in self.use_vlm:
            self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.id_dict = [{} for _ in range(self.n_envs)]
        self.id_dict_clip = [{} for _ in range(self.n_envs)]

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                next_obs_nbytes = 0
                for _, obs in self.observations.items():
                    next_obs_nbytes += obs.nbytes
                total_memory_usage += next_obs_nbytes

            print(f"replay buffer {total_memory_usage / 1_073_741_824:.2f}GB, {mem_available / 1_073_741_824:.2f}GB")

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage / 1_073_741_824:.2f}GB > {mem_available / 1_073_741_824:.2f}GB"
                )

        self.use_prioritized = use_prioritized
        if self.use_prioritized:
            from agents.rl_vlm.models.tree import SumTree
            # Create separate priority trees for each environment
            self.trees = [SumTree(size=self.buffer_size) for _ in range(self.n_envs)]
            self.eps = 0.01  # minimal priority, prevents zero probabilities
            self.alpha = 0.7  # determines how much prioritization is used, ? = 0 corresponding to the uniform case
            self.beta = 0.4  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
            self.max_priority = 0.01
            self.real_size = 0
            # Track environment-specific positions for each tree
            self.env_positions = [0 for _ in range(self.n_envs)]

            # Adaptive priority management to prevent drift
            self.priority_clip_threshold = 50.0  # Clip priorities above this value
            self.priority_decay_factor = 0.9999  # Decay factor for old priorities
            self.min_priority_threshold = 0.01  # Minimum priority threshold
            self.priority_normalization = True  # Enable priority normalization
            self.step_count = 0  # Track training steps

    def add(
            self,
            obs: Dict[str, np.ndarray],
            next_obs: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:

        # Hard-coded, to make sure the saved obs is unit8
        obs['bev_masks'] = obs['bev_masks'].astype(np.uint8)
        next_obs['bev_masks'] = next_obs['bev_masks'].astype(np.uint8)

        # Same reshape, for actions
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.action_dim))

        if self.use_prioritized:
            # Add priorities for each environment separately
            for env_idx in range(self.n_envs):
                self.trees[env_idx].add(max(self.max_priority, 1e-8), self.env_positions[env_idx])
                self.env_positions[env_idx] = (self.env_positions[env_idx] + 1) % self.buffer_size

        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = np.array(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][self.pos] = np.array(next_obs[key]).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        for i in range(self.n_envs):
            self.id_dict[i][infos[i]['id']] = self.pos
            self.id_dict_clip[i][infos[i]['id']] = self.pos

        if 'pvp' in self.use_vlm or 'awac' in self.use_vlm or 'r_sim' in self.use_vlm:
            for i in range(self.n_envs):
                control_dict_list = infos[i]['vlm_actions']
                if control_dict_list:
                    id_list = [control_dict['id'] for control_dict in control_dict_list]
                    for idx in range(len(control_dict_list)):
                        control_dict = control_dict_list[idx]
                        id = id_list[idx]
                        if id in self.id_dict[i]:
                            pos = self.id_dict[i][id]
                        else:
                            continue
                        vlm_action = np.zeros(self.action_space.shape)  # (2, )
                        if len(vlm_action) == 2:
                            if control_dict['throttle'] > 0 and control_dict['throttle'] > control_dict['brake']:
                                acc = control_dict['throttle']
                            elif control_dict['brake'] > 0:
                                acc = -abs(control_dict['brake'])
                            else:
                                acc = 0.
                            vlm_action[0] = acc
                            vlm_action[1] = control_dict['steer']
                        else:
                            vlm_action[0] = control_dict['throttle']
                            vlm_action[1] = control_dict['steer']
                            vlm_action[2] = control_dict['brake']
                        self.vlm_actions[pos, i] = vlm_action
                        self.has_vlm_actions[pos, i] = 1

        if 'r_clip' in self.use_vlm:
            for i in range(self.n_envs):
                clip_safety_scores_list = infos[i]['clip_safety_scores']
                if clip_safety_scores_list:
                    id_list = [clip_safety_scores['id'] for clip_safety_scores in clip_safety_scores_list]
                    for idx in range(len(clip_safety_scores_list)):
                        clip_safety_scores = clip_safety_scores_list[idx]
                        id = id_list[idx]
                        if id in self.id_dict_clip[i]:
                            pos = self.id_dict_clip[i][id]
                            self.clip_safety_scores[pos, i] = clip_safety_scores['score']
                            self.clip_raw_bonus[pos, i] = clip_safety_scores['raw_bonus']
                            self.clip_rms_mean[pos, i] = clip_safety_scores['clip_rms_mean']
                            self.clip_rms_std[pos, i] = clip_safety_scores['clip_rms_std']
                        else:
                            continue

        if 'lag' in self.use_vlm:
            for i in range(self.n_envs):
                cost = infos[i]['cost']
                self.costs[self.pos, i] = cost

        for i in range(self.n_envs):
            self.reward_debugs[i].append(infos[i]['reward_debug']['debug_texts'])
            self.terminal_debugs[i].append(infos[i]['terminal_debug']['debug_texts'])

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
            self.reward_debugs = [[] for i in range(self.n_envs)]
            self.terminal_debugs = [[] for i in range(self.n_envs)]

        if self.use_prioritized:
            self.real_size = min(self.buffer_size, self.real_size + 1)

    def _manage_priorities(self):
        """Adaptive priority management to prevent priority drift"""
        if not self.use_prioritized:
            return

        self.step_count += 1

        # Don't start priority management until we have enough samples
        if self.step_count < 20000:  # Wait for 100 samples before starting management
            return

        # Apply priority decay to prevent old high-priority transitions from dominating
        if self.step_count % 5000 == 0:  # Every 1000 steps
            for tree in self.trees:
                # Decay all priorities slightly
                for i in range(tree.size):
                    if tree.data[i] is not None:
                        current_priority = tree.nodes[i + tree.size - 1]
                        if current_priority > self.min_priority_threshold * 2:
                            decayed_priority = current_priority * self.priority_decay_factor
                            tree.update(i, max(decayed_priority, self.min_priority_threshold))

        # Priority normalization to prevent extreme values (less aggressive)
        if self.priority_normalization and self.step_count % 5000 == 0:  # Less frequent
            for tree in self.trees:
                if tree.total > 0:
                    # Find current priority range
                    priorities = [tree.nodes[i + tree.size - 1] for i in range(tree.size) if tree.data[i] is not None]
                    if priorities:
                        max_prio = max(priorities)
                        min_prio = min(priorities)
                        if max_prio > min_prio and max_prio > self.priority_clip_threshold * 0.5:  # More lenient threshold
                            # Normalize priorities to prevent extreme values
                            print(
                                '==================================normalize_priorities==================================')
                            print(f"max_prio: {max_prio}, min_prio: {min_prio}")
                            scale_factor = self.priority_clip_threshold / max_prio / 2.1
                            for i in range(tree.size):
                                if tree.data[i] is not None:
                                    current_priority = tree.nodes[i + tree.size - 1]
                                    normalized_priority = current_priority * scale_factor
                                    tree.update(i, max(normalized_priority, self.min_priority_threshold))

        # Check if priorities need reset due to extreme skew (less aggressive)
        if self.step_count % 10000 == 0:  # Less frequent
            print('==================================reset_priorities==================================')
            print(self.get_priority_stats())
            self.reset_priorities_if_needed()

        if self.step_count % 1000 == 0:
            stats = self.get_priority_stats()
            print('==================================priority_stats==================================')
            print(stats)

    def reset_priorities_if_needed(self):
        """Reset priorities if they become too skewed"""
        if not self.use_prioritized:
            return

        for tree in self.trees:
            if tree.total > 0:
                # Check if priorities are too skewed (more lenient)
                priorities = [tree.nodes[i + tree.size - 1] for i in range(tree.size) if tree.data[i] is not None]
                if priorities:
                    max_prio = max(priorities)
                    min_prio = min(priorities)
                    if max_prio > min_prio and max_prio > self.priority_clip_threshold * 0.9:
                        print(
                            f"[WARNING] Priority skew detected (max/min ratio: {max_prio / min_prio:.1f}). Resetting priorities.")
                        # Reset all priorities to a moderate value
                        for i in range(tree.size):
                            if tree.data[i] is not None:
                                tree.update(i, self.min_priority_threshold)

    def get_priority_stats(self):
        """Get priority statistics for monitoring"""
        if not self.use_prioritized:
            return {}

        stats = {}
        for i, tree in enumerate(self.trees):
            if tree.total > 0:
                priorities = [tree.nodes[j + tree.size - 1] for j in range(tree.size) if tree.data[j] is not None]
                if priorities:
                    stats[f'env_{i}'] = {
                        'total': tree.total,
                        'max': max(priorities),
                        'min': min(priorities),
                        'mean': np.mean(priorities),
                        'std': np.std(priorities),
                        'skew_ratio': max(priorities) / min(priorities) if min(priorities) > 0 else float('inf')
                    }
        return stats

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        # Manage priorities adaptively to prevent drift
        self._manage_priorities()

        if self.use_prioritized:
            assert self.real_size >= batch_size, "buffer contains less samples than batch size"
            sample_idxs, tree_idxs, env_indices = [], [], []
            priorities = th.empty(batch_size, 1, dtype=th.float32)

            # Ensure we sample from all environments
            samples_per_env = batch_size // self.n_envs
            remaining_samples = batch_size % self.n_envs

            for env_idx in range(self.n_envs):
                # Calculate how many samples to take from this environment
                env_batch_size = samples_per_env + (1 if env_idx < remaining_samples else 0)
                if env_batch_size == 0:
                    continue

                # Sample from this environment's priority tree
                tree_total = self.trees[env_idx].total
                if tree_total <= 0:
                    # Skip this environment if it has no valid priorities
                    continue

                # Try to get samples from this environment with retries
                samples_from_env = 0
                max_retries = env_batch_size * 3  # Allow more retries
                retry_count = 0

                while samples_from_env < env_batch_size and retry_count < max_retries:
                    segment = tree_total / env_batch_size
                    for i in range(env_batch_size - samples_from_env):
                        a, b = segment * i, segment * (i + 1)
                        cumsum = random.uniform(a, b)

                        result = self.trees[env_idx].get(cumsum)
                        if result[0] is None:  # Check if SumTree returned None (no valid data)
                            retry_count += 1
                            continue  # Skip this sample and try again

                        tree_idx, priority, sample_idx = result
                        if type(priority) == np.ndarray:
                            priority = priority.item()

                        priorities[len(sample_idxs)] = priority
                        tree_idxs.append(tree_idx)
                        sample_idxs.append(sample_idx)
                        env_indices.append(env_idx)
                        samples_from_env += 1

                        if samples_from_env >= env_batch_size:
                            break

                    retry_count += 1

            # Check if we have enough samples
            if len(sample_idxs) < batch_size:
                print(
                    f"[WARNING] Only got {len(sample_idxs)} samples from prioritized sampling, falling back to uniform sampling")
                # Fallback to uniform sampling if prioritized sampling fails
                batch_inds = np.random.randint(0, self.buffer_size if self.full else self.pos, size=batch_size)
                env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
                # Return in the same format as prioritized sampling
                samples = self._get_samples(batch_inds, env, env_indices)
                # Create dummy weights and tree indices for consistency
                weights = th.ones(batch_size, 1, dtype=th.float32)
                tree_idxs = [-1] * batch_size  # -1 indicates uniform sampling

                # Validate fallback weights
                if th.isnan(weights).any() or th.isinf(weights).any():
                    print(f"[CRITICAL] Invalid fallback weights detected!")
                    weights = th.ones(batch_size, 1, dtype=th.float32)

                return samples, weights, tree_idxs

            batch_inds = np.array(sample_idxs, dtype=np.int64)
            env_indices = np.array(env_indices, dtype=np.int64)

            # Calculate weights using the combined priorities
            total_priorities = sum(tree.total for tree in self.trees)
            if type(total_priorities) == np.ndarray:
                total_priorities = total_priorities.item()

            probs = priorities / total_priorities
            # Ensure minimum probability to prevent NaN weights (use PyTorch operations)
            probs = th.maximum(probs, th.tensor(1e-4, device=probs.device, dtype=probs.dtype))
            weights = (self.real_size * probs) ** -self.beta
            weights = weights / weights.max()

            # Validate weights before returning
            if th.isnan(weights).any() or th.isinf(weights).any():
                # Replace invalid weights with uniform weights
                weights = th.ones_like(weights)
                print(f"  - Replaced with uniform weights")

            samples = self._get_samples(batch_inds, env, env_indices)
            # Add environment indices to the samples for priority updates
            if hasattr(samples, '_replace'):
                samples = samples._replace(env_indices=self.to_torch(env_indices))
            return samples, weights, tree_idxs
        else:
            # 1. Determine sampling range
            upper_bound = self.buffer_size if self.full else self.pos

            # 2. Initial random sampling
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
            env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

            if 'r_clip' in self.use_vlm:
                scores = self.clip_safety_scores[batch_inds, env_indices]

                # Invalid if score < -5 (init is -1; only sample valid CLIP scores)
                invalid_mask = (scores < -5).flatten()

                retries = 0
                max_retries = 5

                while invalid_mask.any() and retries < max_retries:
                    num_invalid = invalid_mask.sum()

                    # Resample only invalid positions
                    new_batch_inds = np.random.randint(0, upper_bound, size=num_invalid)
                    new_env_indices = np.random.randint(0, high=self.n_envs, size=num_invalid)

                    batch_inds[invalid_mask] = new_batch_inds
                    env_indices[invalid_mask] = new_env_indices

                    # Recheck validity
                    scores = self.clip_safety_scores[batch_inds, env_indices]
                    invalid_mask = (scores < -5).flatten()

                    retries += 1

                # If still invalid after retries (buffer may be mostly -1 at start)
                if invalid_mask.any():
                    print(f"Warning: ReplayBuffer sample contains {invalid_mask.sum()} invalid CLIP scores.")
            # 3. Fetch samples with cleaned indices
            samples = self._get_samples(batch_inds, env, env_indices)

            # Add environment indices to the samples for consistency
            if hasattr(samples, '_replace'):
                samples = samples._replace(env_indices=self.to_torch(env_indices))

            return samples

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None,
                     env_indices: Optional[np.ndarray] = None):
        # Use provided env_indices if available, otherwise sample randomly
        if env_indices is None:
            env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.use_n_step:
            return self._get_n_step_samples(batch_inds, env, env_indices)
        else:
            return self._get_single_step_samples(batch_inds, env, env_indices)

    def _get_single_step_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None,
                                 env_indices: Optional[np.ndarray] = None):
        """Get single-step samples (original behavior)."""
        # Use provided env_indices if available, otherwise sample randomly
        if env_indices is None:
            env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()})
        next_obs_ = self._normalize_obs(
            {key: obs[batch_inds, env_indices, :]
             for key, obs in self.next_observations.items()}
        )

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        if 'none' in self.use_vlm:
            return DictReplayBufferSamples(
                observations=observations,
                actions=self.to_torch(self.actions[batch_inds, env_indices]),
                next_observations=next_observations,
                dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])
                                    ).reshape(-1, 1),
                rewards=self.to_torch(
                    self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
                env_indices=self.to_torch(env_indices),
            )
        else:
            if 'awac' in self.use_vlm or 'pvp' in self.use_vlm:
                vlm_actions = self.to_torch(self.vlm_actions[batch_inds, env_indices])
                has_vlm_actions = self.to_torch(self.has_vlm_actions[batch_inds, env_indices])
            else:
                vlm_actions, has_vlm_actions = None, None
            if 'r_clip' in self.use_vlm:
                clip_safety_scores = self.to_torch(self.clip_safety_scores[batch_inds, env_indices].reshape(-1, 1))
            else:
                clip_safety_scores = None
            if 'lag' in self.use_vlm:
                costs = self.to_torch(self.costs[batch_inds, env_indices])
            else:
                costs = None
            return DictReplayBufferSamplesVlm(
                observations=observations,
                actions=self.to_torch(self.actions[batch_inds, env_indices]),
                next_observations=next_observations,
                dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])
                                    ).reshape(-1, 1),
                rewards=self.to_torch(
                    self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
                vlm_actions=vlm_actions,
                has_vlm_actions=has_vlm_actions,
                env_indices=self.to_torch(env_indices),
                clip_safety_scores=clip_safety_scores,
                costs=costs
            )

    def _get_n_step_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None,
                            env_indices: Optional[np.ndarray] = None):
        """
        Sample a batch of transitions and compute n-step returns.

        For each sampled transition, the method computes the cumulative discounted reward over
        the next `n_steps`, properly handling episode termination and timeouts.
        The next observation and done flag correspond to the last transition in the computed n-step trajectory.

        :param batch_inds: Indices of samples to retrieve
        :param env: Optional VecNormalize environment for normalizing observations/rewards
        :param env_indices: Environment indices for each sample
        :return: A batch of samples with n-step returns and corresponding observations/actions
        """
        # Use provided env_indices if available, otherwise sample randomly
        if env_indices is None:
            env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Note: the self.pos index is dangerous (will overlap two different episodes when buffer is full)
        # so we set self.pos-1 to truncated=True (temporarily) if done=False and truncated=False
        last_valid_index = self.pos - 1
        original_timeout_values = self.timeouts[last_valid_index].copy()
        self.timeouts[last_valid_index] = np.logical_or(original_timeout_values,
                                                        np.logical_not(self.dones[last_valid_index]))

        # Compute n-step indices with wrap-around
        steps = np.arange(self.n_steps).reshape(1, -1)  # shape: [1, n_steps]
        indices = (batch_inds[:, None] + steps) % self.buffer_size  # shape: [batch, n_steps]

        # Retrieve sequences of transitions
        rewards_seq = self._normalize_reward(self.rewards[indices, env_indices[:, None]], env)  # [batch, n_steps]
        dones_seq = self.dones[indices, env_indices[:, None]]  # [batch, n_steps]
        truncated_seq = self.timeouts[indices, env_indices[:, None]]  # [batch, n_steps]

        # Compute masks: 1 until first done/truncation (inclusive)
        done_or_truncated = np.logical_or(dones_seq, truncated_seq)
        done_idx = done_or_truncated.argmax(axis=1)
        # If no done/truncation, keep full sequence
        has_done_or_truncated = done_or_truncated.any(axis=1)
        done_idx = np.where(has_done_or_truncated, done_idx, self.n_steps - 1)

        mask = np.arange(self.n_steps).reshape(1, -1) <= done_idx[:, None]  # shape: [batch, n_steps]

        # Compute discount factors for bootstrapping (using target Q-Value)
        # It is gamma ** n_steps by default but should be adjusted in case of early termination/truncation.
        target_q_discounts = self.gamma ** mask.sum(axis=1, keepdims=True).astype(np.float32)  # [batch, 1]

        # Apply discount
        discounts = self.gamma ** np.arange(self.n_steps, dtype=np.float32).reshape(1, -1)  # [1, n_steps]
        discounted_rewards = rewards_seq * discounts * mask
        n_step_returns = discounted_rewards.sum(axis=1, keepdims=True)  # [batch, 1]

        # Compute indices of next_obs/done at the final point of the n-step transition
        last_indices = (batch_inds + done_idx) % self.buffer_size
        next_obs_ = self._normalize_obs(
            {key: obs[last_indices, env_indices] for key, obs in self.next_observations.items()}
        )
        next_dones = self.dones[last_indices, env_indices][:, None].astype(np.float32)
        next_timeouts = self.timeouts[last_indices, env_indices][:, None].astype(np.float32)
        final_dones = next_dones * (1.0 - next_timeouts)

        # Revert back tmp changes to avoid sampling across episodes
        self.timeouts[last_valid_index] = original_timeout_values

        # Gather observations and actions
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices] for key, obs in self.observations.items()})
        actions = self.actions[batch_inds, env_indices]

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        if not self.use_vlm:
            return DictReplayBufferSamples(
                observations=observations,
                actions=self.to_torch(actions),
                next_observations=next_observations,
                dones=self.to_torch(final_dones),
                rewards=self.to_torch(n_step_returns),
                discounts=self.to_torch(target_q_discounts),
                env_indices=self.to_torch(env_indices),
            )
        else:
            return DictReplayBufferSamplesVlm(
                observations=observations,
                actions=self.to_torch(actions),
                next_observations=next_observations,
                dones=self.to_torch(final_dones),
                rewards=self.to_torch(n_step_returns),
                vlm_actions=self.to_torch(self.vlm_actions[batch_inds, env_indices]),
                has_vlm_actions=self.to_torch(self.has_vlm_actions[batch_inds, env_indices]),
                discounts=self.to_torch(target_q_discounts),
                env_indices=self.to_torch(env_indices),
            )

    def update_priorities(self, data_idxs, priorities, env_indices=None):
        if isinstance(priorities, th.Tensor):
            priorities = priorities.detach().cpu().numpy()

        # Ensure priorities are valid numbers
        priorities = np.nan_to_num(priorities, nan=0.0, posinf=1.0, neginf=0.0)

        if env_indices is None:
            # Fallback to old behavior if env_indices not provided
            for data_idx, priority in zip(data_idxs, priorities):
                # Validate data_idx
                if not (0 <= data_idx < self.buffer_size):
                    print(f"[WARNING] Invalid data_idx: {data_idx}, skipping priority update")
                    continue

                # Ensure priority is valid before power operation
                if np.isnan(priority) or np.isinf(priority):
                    priority = self.min_priority_threshold
                else:
                    priority = max(priority + self.eps, 1e-8) ** self.alpha
                    # Clip priorities to prevent extreme values
                    priority = np.clip(priority, self.min_priority_threshold, self.priority_clip_threshold)
                # Update all environment trees with the same priority
                for tree in self.trees:
                    tree.update(data_idx, priority)
                self.max_priority = max(self.max_priority, priority)
        else:
            # Update priorities for specific environments
            for data_idx, priority, env_idx in zip(data_idxs, priorities, env_indices):
                # Validate data_idx and env_idx
                if not (0 <= data_idx < self.buffer_size and 0 <= env_idx < self.n_envs):
                    print(
                        f"[WARNING] Invalid indices: data_idx={data_idx}, env_idx={env_idx}, skipping priority update")
                    continue

                # Ensure priority is valid before power operation
                if np.isnan(priority) or np.isinf(priority):
                    priority = self.min_priority_threshold
                else:
                    priority = max(priority + self.eps, 1e-8) ** self.alpha
                    # Clip priorities to prevent extreme values
                    priority = np.clip(priority, self.min_priority_threshold, self.priority_clip_threshold)
                # Update the specific environment's tree
                self.trees[env_idx].update(data_idx, priority)
                self.max_priority = max(self.max_priority, priority)

    def warm_up(self, dir, num_envs):
        raise TypeError('disable warm up')
        print('[INFO] Warming up replay buffer')
        path_list = glob.glob(f'{dir}/*.npz')
        for path in tqdm(path_list):
            self._warm_up_single(path, num_envs)
        print(f'[INFO] Current pos at {self.pos}')

    def _warm_up_single(self, path, num_envs):
        """
        Load offline data from a .npz file and distribute it across num_envs environments.
        Each environment will receive approximately T // num_envs transitions.

        :param path: Path to .npz file
        :param num_envs: Number of parallel environments to distribute the data to
        """
        data = np.load(path)

        bev_masks = data["observations__bev_masks"]  # (T, H, W, 3)
        states = data["observations__state"]  # (T, state_dim)
        rewards = data["rewards"]  # (T,)
        dones = data["dones"]  # (T,)
        actions = data["actions"]  # (T, action_dim)

        T = len(rewards)
        transitions_per_env = T // num_envs

        # Check buffer size
        max_T = min(transitions_per_env, self.buffer_size)

        for env_id in range(num_envs):
            start = env_id * transitions_per_env
            end = start + max_T

            self.observations["bev_masks"][self.pos:self.pos + max_T, env_id] = bev_masks[start:end]
            self.observations["state"][self.pos:self.pos + max_T, env_id] = states[start:end]
            self.rewards[self.pos:self.pos + max_T, env_id] = rewards[start:end]
            self.dones[self.pos:self.pos + max_T, env_id] = dones[start:end]
            self.actions[self.pos:self.pos + max_T, env_id] = actions[start:end]

            if not self.optimize_memory_usage:
                self.next_observations["bev_masks"][self.pos:self.pos + max_T - 1, env_id] = bev_masks[start + 1:end]
                self.next_observations["state"][self.pos:self.pos + max_T - 1, env_id] = states[start + 1:end]

        self.pos += max_T
        if self.use_prioritized:
            # Update environment positions for priority trees
            for i in range(self.n_envs):
                self.env_positions[i] = (self.env_positions[i] + max_T) % self.buffer_size
        self.full = max_T == self.buffer_size

    def render(self):
        list_render = []

        _, _, c, h, w = self.observations['bev_masks'].shape
        if c == 15:
            vis_idx = np.array([0, 1, 2, 6, 10, 14])
        elif c == 12:
            vis_idx = np.array([0, 1, 2, 5, 8, 11])
        elif c == 9:
            vis_idx = np.array([0, 1, 2, 4, 6, 8])
        elif c == 6:
            vis_idx = np.array([0, 1, 2, 3, 4, 5])
        elif c == 16:
            vis_idx = np.array([0, 1, 2, 7, 11, 15, 3])
        else:
            raise NotImplementedError('history_idx [] should has at least one idx, for example, -1')

        lookback = 300
        for i in [(self.pos - lookback + i) % self.buffer_size for i in range(lookback)]:  # circular buffer
            im_envs = []
            for j in range(self.n_envs):

                masks = self.observations['bev_masks'][i, j, vis_idx, :, :] > 100

                im_birdview = np.zeros([h, w, 3], dtype=np.uint8)
                for idx_c in range(len(vis_idx)):
                    im_birdview[masks[idx_c]] = COLORS[idx_c]

                # a special render for ego vehicle
                masks_ego = self.observations['bev_masks'][i, j, 1, :, :] == 120
                im_birdview[masks_ego] = COLORS[-1]

                im = np.zeros([h * 2, w * 2, 3], dtype=np.uint8)  # Double the height to fit all debug texts
                im[:h, :w] = im_birdview

                action_str = np.array2string(self.actions[i, j], precision=1, separator=',', suppress_small=True)
                try:
                    vlm_action_str = np.array2string(self.vlm_actions[i, j], precision=1, separator=',',
                                                     suppress_small=True)
                except:
                    vlm_action_str = ''
                state_str = np.array2string(self.observations['state'][i, j],
                                            precision=1, separator=',', suppress_small=True)

                reward = self.rewards[i, j]
                done = int(self.dones[i, j])

                txt_1 = f'a{action_str}, av{vlm_action_str}'
                im = cv2.putText(im, txt_1, (2, 7), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
                txt_2 = f'{done} {state_str}'
                im = cv2.putText(im, txt_2, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
                txt_3 = f'rw:{reward:5.2f}'
                im = cv2.putText(im, txt_3, (2, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)

                try:
                    for i_txt, txt in enumerate(
                            self.reward_debugs[j][i - self.pos] + self.terminal_debugs[j][i - self.pos]):
                        # Position debug texts in the right half with more vertical space
                        im = cv2.putText(im, txt, (w, (i_txt + 1) * 14), cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                                         (255, 255, 255), 1)
                except:
                    pass
                im_envs.append(im)

            big_im = tile_images(im_envs)
            list_render.append(big_im)

        return list_render

