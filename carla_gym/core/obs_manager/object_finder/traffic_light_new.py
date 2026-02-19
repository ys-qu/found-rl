import numpy as np
import carla
from gym import spaces
from carla_gym.core.obs_manager.obs_manager import ObsManagerBase
from carla_gym.utils.traffic_light import TrafficLightHandler


class ObsManager(ObsManagerBase):
    # Template config
    # obs_configs = {
    #     "module": "object_finder.traffic_light_new",
    # }
    def __init__(self, obs_configs):
        self._parent_actor = None
        super(ObsManager, self).__init__()

    def _define_obs_space(self):
        self.obs_space = spaces.Dict({
            'light_state': spaces.Box(low=0.0, high=3.0, shape=(1,), dtype=np.float32),
            'distance_rl_ratio': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'ttc_phase_end_ratio': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        })

    def attach_ego_vehicle(self, parent_actor):
        self._parent_actor = parent_actor

    def get_observation(self):
        _tl_offset = -0.8 * self._parent_actor.vehicle.bounding_box.extent.x
        light_state, light_loc, light_id, distance, ttc_phase_end_ratio = TrafficLightHandler.get_light_state_with_timing(self._parent_actor.vehicle, offset=_tl_offset, dist_threshold=18.0)
        if light_state == carla.TrafficLightState.Red:
            light_state_id = 1
            dist_rl = max(0.0, np.linalg.norm(light_loc[0:2])-5.0)
            dist_rl_ratio = np.clip(dist_rl, 0.0, 5.0)/5.0
        elif light_state == carla.TrafficLightState.Yellow:
            light_state_id = 2
            dist_rl = max(0.0, np.linalg.norm(light_loc[0:2])-5.0)
            dist_rl_ratio = np.clip(dist_rl, 0.0, 5.0)/5.0
        elif light_state == carla.TrafficLightState.Green:
            light_state_id = 3
            dist_rl_ratio = -1.0
        else:
            light_state_id = 0
            dist_rl_ratio = -1.0
            ttc_phase_end_ratio = -1.0
        
        obs = {
            'light_state': np.array([light_state_id], dtype=np.float32),
            'distance_rl_ratio': np.array([dist_rl_ratio], dtype=np.float32),
            'ttc_phase_end_ratio': np.array([ttc_phase_end_ratio], dtype=np.float32),
        }
        
        return obs

    def clean(self):
        self._parent_actor = None