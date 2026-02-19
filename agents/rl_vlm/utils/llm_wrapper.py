"""Env wrappers for VLM (LLM) and OpenCLIP: send requests to async inference servers, receive actions/scores."""
from gym import Wrapper
import numpy as np
import carla
import carla_gym.utils.transforms as trans_utils
import carla_gym.core.task_actor.common.navigation.route_manipulation as gps_util
from utils.carla_utils import TTCCalculator
from agents.rl_vlm.utils.async_types import InferenceRequest, InferenceRequestOpenClip
import time
import copy


COMMANDS = [
    'turn left at the intersection',
    'turn right at the intersection',
    'go straight at the intersection',
    'follow the current lane',
    'change to the left lane',
    'change to the right lane'
]


class LLMWrapper(Wrapper):
    def __init__(self, env, env_index, request_queue, shared_output_dict):
        super().__init__(env)
        self.env_index = env_index
        self.request_queue = request_queue
        self.shared_output_dict = shared_output_dict
        self.has_requested = False
        self.vlm_slow_interval = 1
        self.step_count = 0
        self.enable_wait_vlm = True
        self.eval_mode = False

    def generate_prompt(self, info):
        speed = info['forward_speed'][0]

        ev_gps = info['gnss']['gnss']
        compass = 0.0 if np.isnan(info['gnss']['imu'][-1]) else info['gnss']['imu'][-1]
        gps_point = info['gnss']['target_gps']
        target_vec_in_global = gps_util.gps_to_location(gps_point) - gps_util.gps_to_location(ev_gps)
        ref_rot_in_global = carla.Rotation(yaw=float(np.rad2deg(compass)) - 90.0)
        loc_in_ev = trans_utils.vec_global_to_ref(target_vec_in_global, ref_rot_in_global)
        forward_vector_x = loc_in_ev.x
        forward_vector_y = loc_in_ev.y

        command = info['gnss']['command'][0]
        if command < 0:
            command = 4
        command -= 1
        assert command in [0, 1, 2, 3, 4, 5]
        command_prompt = COMMANDS[command]

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
        return prompt

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def is_unsafe(self, info):
        ego_vehicles = self.env.ev_handler.ego_vehicles
        for ev_id, ev in ego_vehicles.items():
            ttc = TTCCalculator.get_ttc(ev.vehicle, self.world, self.world.get_map())

        outside_route_lane = info.get('outside_route_lane')
        route_deviation = info.get('route_deviation')
        blocked = info.get('blocked')
        collision = info.get('collision')
        run_red_light = info.get('run_red_light')
        run_stop_sign = info.get('run_stop_sign')
        if ttc < 3 or outside_route_lane or route_deviation or blocked or collision or run_red_light or run_stop_sign:
            return 1
        else:
            return 0

    def set_eval_mode(self, flag: bool):
        self.eval_mode = bool(flag)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.eval_mode:
            return obs, reward, done, info

        prompt = self.generate_prompt(info)
        central_rgb = info['central_rgb']

        if self.step_count % self.vlm_slow_interval == 0:
            self.request_queue.put(InferenceRequest(prompt, central_rgb, self.env_index, info['id']))
            vlm_working = 1
        else:
            vlm_working = self.is_unsafe(info)
            if vlm_working:
                self.request_queue.put(InferenceRequest(prompt, central_rgb, self.env_index, info['id']))
        ###################################################################
        if self.enable_wait_vlm and self.request_queue.qsize() > 64:
            wait_round = 0
            control_dict_list = None
            while control_dict_list is None:
                control_dict_list = self.shared_output_dict.get(self.env_index, None)
                if control_dict_list is None:
                    time.sleep(0.01)
                    wait_round += 1
                else:
                    info['vlm_actions'] = copy.deepcopy(control_dict_list)
                    self.shared_output_dict[self.env_index] = []

                if wait_round > 100:
                    info['vlm_actions'] = None
                    print('[WARNING] llm wrapper wait round is > 100')
                    break

        else:
            control_dict_list = self.shared_output_dict.get(self.env_index, None)
            if control_dict_list is not None:
                info['vlm_actions'] = copy.deepcopy(control_dict_list)
                self.shared_output_dict[self.env_index] = []
            else:
                info['vlm_actions'] = None

        self.step_count += 1
        return obs, reward, done, info


class OpenCLIPWrapper(Wrapper):
    def __init__(self, env, env_index, request_queue, shared_output_dict):
        super().__init__(env)
        self.env_index = env_index
        self.request_queue = request_queue
        self.shared_output_dict = shared_output_dict
        self.has_requested = False
        self.vlm_slow_interval = 1
        self.step_count = 0
        self.jump_gap = 0
        self.enable_wait_vlm = True
        self.eval_mode = False

    def _get_action_description(self, throttle, steer, brake):
        """Map continuous action to semantic description (30 action types)."""
        action_parts = []
        if brake > 0.05:
            if brake > 0.5:
                action_parts.append("braking hard")
            else:
                action_parts.append("braking")
        elif throttle > 0.05:
            if throttle > 0.8:
                action_parts.append("accelerating fast")
            elif throttle > 0.3:
                action_parts.append("accelerating")
            else:
                action_parts.append("accelerating gently")
        else:
            action_parts.append("idling")
        if abs(steer) < 0.05:
            action_parts.append("going straight")
        elif steer > 0:
            if steer > 0.3:
                action_parts.append("turning right sharply")
            else:
                action_parts.append("turning right")
        else:
            if steer < -0.3:
                action_parts.append("turning left sharply")
            else:
                action_parts.append("turning left")

        return " and ".join(action_parts)

    def _get_speed_index(self, speed_val):
        """Speed level classification (0-3)"""
        if speed_val < 0.1:
            return 0
        elif speed_val < 2.0:
            return 1
        elif speed_val < 4.5:
            return 2
        else:
            return 3

    def _get_command_index(self, raw_cmd):
        """Map raw command to index 0-5."""
        if raw_cmd < 0:
            return 3
        cmd_idx = int(raw_cmd - 1)
        if cmd_idx < 0: cmd_idx = 3
        if cmd_idx > 5: cmd_idx = 3
        return cmd_idx

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def set_eval_mode(self, flag: bool):
        self.eval_mode = bool(flag)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.eval_mode:
            return obs, reward, done, info

        steer = float(action[0])
        acc = float(action[1])
        if acc >= 0:
            throttle, brake = acc, 0.0
        else:
            throttle, brake = 0.0, abs(acc)

        action_desc = self._get_action_description(throttle, steer, brake)
        speed_idx = self._get_speed_index(info['forward_speed'][0])
        cmd_idx = self._get_command_index(info['gnss']['command'][0])

        central_rgb = info['central_rgb']

        if self.step_count % self.vlm_slow_interval == 0:
            req = InferenceRequestOpenClip(
                action_desc=action_desc,
                command_idx=cmd_idx,
                speed_idx=speed_idx,
                image=central_rgb,
                env_index=self.env_index,
                id=info['id']
            )
            self.request_queue.put(req)
        if self.enable_wait_vlm and self.request_queue.qsize() > 64:
            wait_round = 0
            clip_safety_scores_list = None
            while clip_safety_scores_list is None:
                clip_safety_scores_list = self.shared_output_dict.get(self.env_index, None)
                if clip_safety_scores_list is None:
                    time.sleep(0.01)
                    wait_round += 1
                else:
                    info['clip_safety_scores'] = copy.deepcopy(clip_safety_scores_list)
                    self.shared_output_dict[self.env_index] = []

                if wait_round > 100:
                    info['clip_safety_scores'] = None
                    print('[WARNING] openclip wrapper wait round is > 100')
                    break

        else:
            clip_safety_scores_list = self.shared_output_dict.get(self.env_index, None)
            if clip_safety_scores_list is not None:
                info['clip_safety_scores'] = copy.deepcopy(clip_safety_scores_list)
                self.shared_output_dict[self.env_index] = []
            else:
                info['clip_safety_scores'] = None

        self.step_count += 1
        return obs, reward, done, info
