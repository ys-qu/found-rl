import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, Optional, Union, NamedTuple

import cv2
import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import RolloutBuffer
from agents.rl_vlm.models.torch_util import convert_jump_gap_to_actions
from stable_baselines3.common.vec_env.base_vec_env import tile_images

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


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


class DictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    exploration_suggests: th.Tensor


class DictRolloutBufferSamplesVlm(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    exploration_suggests: th.Tensor
    vlm_actions: th.Tensor
    has_vlm_actions: th.Tensor


class RolloutBufferVlm(RolloutBuffer):
    observation_space: spaces.Dict
    obs_shape: dict[str, tuple[int, ...]]  # type: ignore[assignment]
    observations: dict[str, np.ndarray]  # type: ignore[assignment]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        use_vlm: bool = False,
    ):
        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.use_vlm = use_vlm

        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = np.zeros((self.buffer_size, self.n_envs, *obs_input_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        if self.use_vlm:
            self.vlm_actions = np.zeros((self.buffer_size, self.n_envs) + self.action_space.shape, dtype=np.float32)
            self.has_vlm_actions = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
        self.exploration_suggests = np.zeros((self.buffer_size, self.n_envs), dtype=[('acc', 'U10'), ('steer', 'U10')])
        self.reward_debugs = [[] for i in range(self.n_envs)]
        self.terminal_debugs = [[] for i in range(self.n_envs)]
        self.id_dict = [{} for _ in range(self.n_envs)]
        super(RolloutBuffer, self).reset()

    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        infos: None
    ) -> None:
        obs['bev_masks'] = obs['bev_masks'].astype(np.uint8)
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = np.array(obs[key])
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()

        for i in range(self.n_envs):
            self.reward_debugs[i].append(infos[i]['reward_debug']['debug_texts'])
            self.terminal_debugs[i].append(infos[i]['terminal_debug']['debug_texts'])

            n_steps = infos[i]['terminal_debug']['exploration_suggest']['n_steps']
            if n_steps > 0:
                n_start = max(0, self.pos-n_steps)
                self.exploration_suggests[n_start:self.pos, i] = \
                    infos[i]['terminal_debug']['exploration_suggest']['suggest']

        if self.use_vlm:
            for i in range(self.n_envs):
                control_dict_list = infos[i]['vlm_actions']
                if control_dict_list:
                    id_list = [control_dict['id'] for control_dict in control_dict_list]
                    for idx in range(len(control_dict_list)):
                        control_dict = control_dict_list[idx]
                        id = id_list[idx]
                        pos = self.id_dict[i][id]
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
            for i in range(self.n_envs):
                self.id_dict[i][infos[i]['id']] = self.pos

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(  # type: ignore[override]
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns", "exploration_suggests"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictRolloutBufferSamples | DictRolloutBufferSamplesVlm:
        obs = {key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()}
        if not self.use_vlm:
            return DictRolloutBufferSamples(
                observations=obs,
                actions=self.to_torch(self.actions[batch_inds]),
                old_values=self.to_torch(self.values[batch_inds].flatten()),
                old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
                advantages=self.to_torch(self.advantages[batch_inds].flatten()),
                returns=self.to_torch(self.returns[batch_inds].flatten()),
                exploration_suggests=self.exploration_suggests[batch_inds].flatten()
            )
        else:
            return DictRolloutBufferSamplesVlm(
                observations=obs,
                actions=self.to_torch(self.actions[batch_inds]),
                old_values=self.to_torch(self.values[batch_inds].flatten()),
                old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
                advantages=self.to_torch(self.advantages[batch_inds].flatten()),
                returns=self.to_torch(self.returns[batch_inds].flatten()),
                exploration_suggests=self.exploration_suggests[batch_inds].flatten(),
                vlm_actions=self.to_torch(self.vlm_actions[batch_inds]),
                has_vlm_actions=self.to_torch(self.has_vlm_actions[batch_inds].flatten()),
            )

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

                im = np.zeros([h, w*2, 3], dtype=np.uint8)
                im[:h, :w] = im_birdview

                action_str = np.array2string(self.actions[i, j], precision=1, separator=',', suppress_small=True)
                state_str = np.array2string(self.observations['state'][i, j],
                                            precision=1, separator=',', suppress_small=True)

                reward = self.rewards[i, j]

                txt_1 = f'a{action_str}'
                im = cv2.putText(im, txt_1, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                txt_2 = f'{state_str}'
                im = cv2.putText(im, txt_2, (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                txt_3 = f'rw:{reward:5.2f}'
                im = cv2.putText(im, txt_3, (2, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

                try:
                    for i_txt, txt in enumerate(self.reward_debugs[j][i] + self.terminal_debugs[j][i]):
                        im = cv2.putText(im, txt, (w, (i_txt+1)*15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                except:
                    pass

                im_envs.append(im)

            big_im = tile_images(im_envs)
            list_render.append(big_im)

        return list_render