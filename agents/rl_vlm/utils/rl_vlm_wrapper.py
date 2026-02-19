"""Observation/action processing and rendering for RL agents using birdview + state inputs."""
import gym
import numpy as np
import cv2
import carla
import uuid

eval_num_zombie_vehicles = {
    'Town01': 120,
    'Town02': 70,
    'Town03': 70,
    'Town04': 150,
    'Town05': 120,
    'Town06': 120
}
eval_num_zombie_walkers = {
    'Town01': 120,
    'Town02': 70,
    'Town03': 70,
    'Town04': 80,
    'Town05': 120,
    'Town06': 80
}


def downsample_maxpool(bev_masks, k):
    """
    bev_masks: (C,H,W) in {0,1}
    k: downsample factor (e.g., 2 or 4)
    returns: (C, H//k, W//k) with block-wise OR
    """
    C, H, W = bev_masks.shape
    H2, W2 = (H // k) * k, (W // k) * k
    x = bev_masks[:, :H2, :W2].reshape(C, H2//k, k, W2//k, k)
    y = x.max(axis=(2, 4)).astype(np.uint8)
    return y


class RlVlmWrapper(gym.Wrapper):
    _birdview_key = None
    _compress_size = None

    def __init__(self, env, input_states=[], acc_as_action=False, birdview_key='masks', compress_size=[128, 128]):
        assert len(env.obs_configs) == 1
        self._ev_id = list(env.obs_configs.keys())[0]
        self._input_states = input_states
        self._acc_as_action = acc_as_action
        self._birdview_key = self.set_birdview_key(birdview_key)
        self._compress_size = self.set_compress_size(compress_size)
        self._render_dict = {}

        if len(self._input_states) > 0:
            state_spaces = []
            if 'speed' in self._input_states:
                state_spaces.append(env.observation_space[self._ev_id]['speed']['speed_xy'])
            if 'speed_limit' in self._input_states:
                state_spaces.append(env.observation_space[self._ev_id]['control']['speed_limit'])
            if 'control' in self._input_states:
                state_spaces.append(env.observation_space[self._ev_id]['control']['throttle'])
                state_spaces.append(env.observation_space[self._ev_id]['control']['steer'])
                state_spaces.append(env.observation_space[self._ev_id]['control']['brake'])
                state_spaces.append(env.observation_space[self._ev_id]['control']['gear'])
            if 'acc_xy' in self._input_states:
                state_spaces.append(env.observation_space[self._ev_id]['velocity']['acc_xy'])
            if 'vel_xy' in self._input_states:
                state_spaces.append(env.observation_space[self._ev_id]['velocity']['vel_xy'])
            if 'vel_ang_z' in self._input_states:
                state_spaces.append(env.observation_space[self._ev_id]['velocity']['vel_ang_z'])
            if 'traffic_light' in self._input_states:
                state_spaces.append(env.observation_space[self._ev_id]['traffic_light']['light_state'])
                state_spaces.append(env.observation_space[self._ev_id]['traffic_light']['distance_rl_ratio'])
                state_spaces.append(env.observation_space[self._ev_id]['traffic_light']['ttc_phase_end_ratio'])
            if 'stop_sign' in self._input_states:
                state_spaces.append(env.observation_space[self._ev_id]['stop_sign']['at_stop_sign'])

            state_low = np.concatenate([s.low for s in state_spaces])
            state_high = np.concatenate([s.high for s in state_spaces])

            original_space = env.observation_space[self._ev_id]['birdview'][self._birdview_key]
            new_space = gym.spaces.Box(low=original_space.low.min(), high=original_space.high.max(),
                                       shape=(original_space.shape[0], compress_size[0], compress_size[1]),
                            dtype=original_space.dtype)
            env.observation_space[self._ev_id]['birdview'][self._birdview_key] = new_space
            env.observation_space = gym.spaces.Dict(
                {'state': gym.spaces.Box(low=state_low, high=state_high, dtype=np.float32),
                 'bev_masks': env.observation_space[self._ev_id]['birdview'][self._birdview_key],
                 }
            )
        else:
            env.observation_space = env.observation_space[self._ev_id]['birdview'][self._birdview_key]

        if self._acc_as_action:
            env.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        else:
            env.action_space = gym.spaces.Box(low=np.array([0, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32)

        super(RlVlmWrapper, self).__init__(env)

        self.eval_mode = False

    @classmethod
    def set_birdview_key(cls, key):
        cls._birdview_key = key
        return cls._birdview_key

    @classmethod
    def get_birdview_key(cls):
        return cls._birdview_key

    @classmethod
    def set_compress_size(cls, key):
        cls._compress_size = key
        return cls._compress_size

    @classmethod
    def get_compress_size(cls):
        return cls._compress_size

    def set_eval_mode(self, flag: bool):
        self.eval_mode = bool(flag)

    def reset(self):
        self.env.set_task_idx(np.random.choice(self.env.num_tasks))
        if self.eval_mode:
            self.env.task['num_zombie_vehicles'] = eval_num_zombie_vehicles[self.env.carla_map]
            self.env.task['num_zombie_walkers'] = eval_num_zombie_walkers[self.env.carla_map]
            for ev_id in self.env.ev_handler._terminal_configs:
                self.env.ev_handler._terminal_configs[ev_id]['kwargs']['eval_mode'] = True
        else:
            for ev_id in self.env.ev_handler._terminal_configs:
                self.env.ev_handler._terminal_configs[ev_id]['kwargs']['eval_mode'] = False

        obs_ma = self.env.reset()
        action_ma = {self._ev_id: carla.VehicleControl(manual_gear_shift=True, gear=1)}
        obs_ma, _, _, _ = self.env.step(action_ma)
        action_ma = {self._ev_id: carla.VehicleControl(manual_gear_shift=False)}
        obs_ma, _, _, _ = self.env.step(action_ma)

        snap_shot = self.env.world.get_snapshot()
        self.env.timestamp = {
            'step': 0,
            'frame': 0,
            'relative_wall_time': 0.0,
            'wall_time': snap_shot.timestamp.platform_timestamp,
            'relative_simulation_time': 0.0,
            'simulation_time': snap_shot.timestamp.elapsed_seconds,
            'start_frame': snap_shot.timestamp.frame,
            'start_wall_time': snap_shot.timestamp.platform_timestamp,
            'start_simulation_time': snap_shot.timestamp.elapsed_seconds
        }

        obs = self.process_obs(obs_ma[self._ev_id], self._input_states)

        self._render_dict['prev_obs'] = obs
        self._render_dict['prev_im_render'] = obs_ma[self._ev_id]['birdview']['rendered']
        return obs

    def step(self, action):
        action_ma = {self._ev_id: self.process_act(action, self._acc_as_action)}

        obs_ma, reward_ma, done_ma, info_ma = self.env.step(action_ma)
        obs = self.process_obs(obs_ma[self._ev_id], self._input_states)
        self.process_info(info_ma[self._ev_id], obs_ma[self._ev_id], obs)
        reward = reward_ma[self._ev_id]
        done = done_ma[self._ev_id]
        info = info_ma[self._ev_id]

        self._render_dict = {
            'timestamp': self.env.timestamp,
            'obs': self._render_dict['prev_obs'],
            'prev_obs': obs,
            'im_render': self._render_dict['prev_im_render'],
            'prev_im_render': obs_ma[self._ev_id]['birdview']['rendered'],
            'action': action,
            'reward_debug': info['reward_debug'],
            'terminal_debug': info['terminal_debug']
        }

        return obs, reward, done, info

    def render(self, mode='human'):
        return self.im_render(self._render_dict)

    @staticmethod
    def im_render(render_dict):
        im_birdview = render_dict['im_render']
        h, w, c = im_birdview.shape
        im = np.zeros([h, w*2, c], dtype=np.uint8)
        im[:h, :w] = im_birdview

        action_str = np.array2string(render_dict['action'], precision=2, separator=',', suppress_small=True)
        state_str = np.array2string(render_dict['obs']['state'], precision=2, separator=',', suppress_small=True)

        txt_t = f'step:{render_dict["timestamp"]["step"]:5}, frame:{render_dict["timestamp"]["frame"]:5}'
        im = cv2.putText(im, txt_t, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_1 = f'a{action_str}'
        im = cv2.putText(im, txt_1, (3, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_2 = f's{state_str}'
        im = cv2.putText(im, txt_2, (3, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        for i, txt in enumerate(render_dict['reward_debug']['debug_texts'] +
                                render_dict['terminal_debug']['debug_texts']):
            im = cv2.putText(im, txt, (w, (i+2)*12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return im

    @staticmethod
    def process_obs(obs, input_states, train=True):
        birdview_key = RlVlmWrapper.get_birdview_key()
        compress_size = RlVlmWrapper.get_compress_size()

        state_list = []
        if 'speed' in input_states:
            state_list.append(obs['speed']['speed_xy'])
        if 'speed_limit' in input_states:
            state_list.append(obs['control']['speed_limit'])
        if 'control' in input_states:
            state_list.append(obs['control']['throttle'])
            state_list.append(obs['control']['steer'])
            state_list.append(obs['control']['brake'])
            state_list.append(obs['control']['gear']/5.0)
        if 'acc_xy' in input_states:
            state_list.append(obs['velocity']['acc_xy'])
        if 'vel_xy' in input_states:
            state_list.append(obs['velocity']['vel_xy'])
        if 'vel_ang_z' in input_states:
            state_list.append(obs['velocity']['vel_ang_z'])
        if 'traffic_light' in input_states:
            state_list.append(obs['traffic_light']['light_state'])
            state_list.append(obs['traffic_light']['distance_rl_ratio'])
            state_list.append(obs['traffic_light']['ttc_phase_end_ratio'])
        if 'stop_sign' in input_states:
            state_list.append(obs['stop_sign']['at_stop_sign'])

        bev_masks = obs['birdview'][birdview_key].astype(np.uint8)
        if compress_size[0] == 96 and birdview_key == 'masks':
            assert bev_masks.shape[1] / compress_size[0] == 2
            bev_masks = downsample_maxpool(bev_masks, 2)

        if len(state_list) > 0:
            state = np.concatenate(state_list)

            if not train:
                bev_masks = np.expand_dims(bev_masks, 0)
                state = np.expand_dims(state, 0)

            obs_dict = {
                'state': state.astype(np.float32),
                'bev_masks': bev_masks,
            }
            return obs_dict
        else:
            if not train:
                bev_masks = np.expand_dims(bev_masks, 0)
            return bev_masks

    @staticmethod
    def process_act(action, acc_as_action, train=True):
        if not train:
            action = action[0]
        if acc_as_action:
            acc, steer = action.astype(np.float64)
            if acc >= 0.0:
                throttle = acc
                brake = 0.0
            else:
                throttle = 0.0
                brake = np.abs(acc)
        else:
            throttle, steer, brake = action.astype(np.float64)

        throttle = np.clip(throttle, 0, 1)
        steer = np.clip(steer, -1, 1)
        brake = np.clip(brake, 0, 1)
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        return control

    def process_info(self, info, obs, obs_processed):
        bev_rendered = np.transpose(obs['birdview']['rendered'], (1, 2, 0))
        forward_speed = obs['speed']['forward_speed']
        gnss = obs['gnss']
        info['forward_speed'] = forward_speed
        info['gnss'] = gnss
        info['central_rgb'] = bev_rendered
        uid16 = uuid.uuid4().hex[:16]
        info['id'] = uid16