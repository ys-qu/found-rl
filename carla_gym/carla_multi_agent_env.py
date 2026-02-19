import gc
import logging
import random
import time

import gym
import numpy as np
import carla

from .core.zombie_walker.zombie_walker_handler import ZombieWalkerHandler
from .core.zombie_vehicle.zombie_vehicle_handler import ZombieVehicleHandler
from .core.obs_manager.obs_manager_handler import ObsManagerHandler
from .core.task_actor.ego_vehicle.ego_vehicle_handler import EgoVehicleHandler
from .core.task_actor.scenario_actor.scenario_actor_handler import ScenarioActorHandler
from .utils.traffic_light import TrafficLightHandler
from .utils.dynamic_weather import WeatherHandler
from stable_baselines3.common.utils import set_random_seed

logger = logging.getLogger(__name__)


class CarlaMultiAgentEnv(gym.Env):
    def __init__(self, carla_map, host, port, seed, no_rendering,
                 obs_configs, reward_configs, terminal_configs, all_tasks, light_always_green=False, disbale_bg_actors=False):
        self._all_tasks = all_tasks
        self.obs_configs = obs_configs
        self.carla_map = carla_map
        self._seed = seed
        self.light_always_green = light_always_green
        self.disbale_bg_actors = disbale_bg_actors
        self.name = self.__class__.__name__

        self._init_client(carla_map, host, port, seed=seed, no_rendering=no_rendering)

        # define observation spaces exposed to agent
        self._om_handler = ObsManagerHandler(obs_configs)
        self.ev_handler = EgoVehicleHandler(self._client, reward_configs, terminal_configs)
        self._zw_handler = ZombieWalkerHandler(self._client)
        self._zv_handler = ZombieVehicleHandler(self._client, tm_port=self._tm.get_port())
        self._sa_handler = ScenarioActorHandler(self._client)
        self._wt_handler = WeatherHandler(self.world)

        # observation spaces
        self.observation_space = self._om_handler.observation_space
        # define action spaces exposed to agent
        # throttle, steer, brake
        self.action_space = gym.spaces.Dict({ego_vehicle_id: gym.spaces.Box(
            low=np.array([0.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32)
            for ego_vehicle_id in obs_configs.keys()})

        self._task_idx = 0
        self._shuffle_task = True
        self._task = self._all_tasks[self._task_idx].copy()

        self.num_timesteps = 0
        self.last_reward = [0]
        self.first_reset = True

        self.disbale_static = True
        self.eval_mode = False

    def set_task_idx(self, task_idx):
        self._task_idx = task_idx
        self._shuffle_task = False
        self._task = self._all_tasks[self._task_idx].copy()

    @property
    def num_tasks(self):
        return len(self._all_tasks)

    @property
    def task(self):
        return self._task

    def soft_reset(self):
        # do not reset zombies, but they might occupancy ev's spawn locations, so make sure ego is reset with max tries
        if self._shuffle_task:
            self._task_idx = np.random.choice(self.num_tasks)
            self._task = self._all_tasks[self._task_idx].copy()
        self.soft_clean()

        self._wt_handler.reset(self.task['weather'])
        logger.debug("_wt_handler reset done!!")

        ev_spawn_locations = self.ev_handler.reset(self.task['ego_vehicles'])
        logger.debug("_ev_handler reset done!!")

        self._sa_handler.reset(self.task['scenario_actors'], self.ev_handler.ego_vehicles)
        logger.debug("_sa_handler reset done!!")

        self._om_handler.reset(self.ev_handler.ego_vehicles)
        logger.debug("_om_handler reset done!!")

        self.world.tick()

        snap_shot = self.world.get_snapshot()
        self._timestamp = {
            'step': 0,
            'frame': snap_shot.timestamp.frame,
            'relative_wall_time': 0.0,
            'wall_time': snap_shot.timestamp.platform_timestamp,
            'relative_simulation_time': 0.0,
            'simulation_time': snap_shot.timestamp.elapsed_seconds,
            'start_frame': snap_shot.timestamp.frame,
            'start_wall_time': snap_shot.timestamp.platform_timestamp,
            'start_simulation_time': snap_shot.timestamp.elapsed_seconds
        }

        _, _, _ = self.ev_handler.tick(self.timestamp)
        # get obeservations
        obs_dict = self._om_handler.get_observation(self.timestamp)

        if self.light_always_green:
            for light in self.world.get_actors().filter('traffic.traffic_light'):
                light.set_state(carla.TrafficLightState.Green)
                light.set_green_time(9999)
                light.set_red_time(0)
                light.set_yellow_time(0)
                light.freeze(True)

        if self.disbale_static:
            labels_to_hide = [
                carla.CityObjectLabel.Poles,
                carla.CityObjectLabel.Fences,
                carla.CityObjectLabel.GuardRail,
                carla.CityObjectLabel.TrafficSigns,
                carla.CityObjectLabel.TrafficLight,
                carla.CityObjectLabel.Static,
                carla.CityObjectLabel.Vegetation,
            ]

            obj_ids = []
            for lab in labels_to_hide:
                obj_ids += [o.id for o in self.world.get_environment_objects(lab)]

            self.world.enable_environment_objects(obj_ids, False)
        return obs_dict

    def hard_reset(self):
        if self._shuffle_task:
            self._task_idx = np.random.choice(self.num_tasks)
            self._task = self._all_tasks[self._task_idx].copy()
        self.clean()

        self._wt_handler.reset(self.task['weather'])
        logger.debug("_wt_handler reset done!!")

        ev_spawn_locations = self.ev_handler.reset(self.task['ego_vehicles'])
        logger.debug("_ev_handler reset done!!")

        self._sa_handler.reset(self.task['scenario_actors'], self.ev_handler.ego_vehicles)
        logger.debug("_sa_handler reset done!!")

        self._zw_handler.reset(self.task['num_zombie_walkers'], ev_spawn_locations)
        logger.debug("_zw_handler reset done!!")

        self._zv_handler.reset(self.task['num_zombie_vehicles'], ev_spawn_locations)
        logger.debug("_zv_handler reset done!!")

        self._om_handler.reset(self.ev_handler.ego_vehicles)
        logger.debug("_om_handler reset done!!")

        self.world.tick()

        snap_shot = self.world.get_snapshot()
        self._timestamp = {
            'step': 0,
            'frame': snap_shot.timestamp.frame,
            'relative_wall_time': 0.0,
            'wall_time': snap_shot.timestamp.platform_timestamp,
            'relative_simulation_time': 0.0,
            'simulation_time': snap_shot.timestamp.elapsed_seconds,
            'start_frame': snap_shot.timestamp.frame,
            'start_wall_time': snap_shot.timestamp.platform_timestamp,
            'start_simulation_time': snap_shot.timestamp.elapsed_seconds
        }

        _, _, _ = self.ev_handler.tick(self.timestamp)
        # get obeservations
        obs_dict = self._om_handler.get_observation(self.timestamp)

        if self.light_always_green:
            for light in self.world.get_actors().filter('traffic.traffic_light'):
                light.set_state(carla.TrafficLightState.Green)
                light.set_green_time(9999)
                light.set_red_time(0)
                light.set_yellow_time(0)
                light.freeze(True)

        if self.disbale_static:
            labels_to_hide = [
                carla.CityObjectLabel.Poles,
                carla.CityObjectLabel.Fences,
                carla.CityObjectLabel.GuardRail,
                carla.CityObjectLabel.TrafficSigns,
                carla.CityObjectLabel.TrafficLight,
                carla.CityObjectLabel.Static,
                carla.CityObjectLabel.Vegetation,
            ]

            obj_ids = []
            for lab in labels_to_hide:
                obj_ids += [o.id for o in self.world.get_environment_objects(lab)]

            self.world.enable_environment_objects(obj_ids, False)

        return obs_dict

    def reset(self):
        if self.disbale_bg_actors:
            return self.soft_reset()
        else:
            if self.first_reset:
                self.first_reset = False
                return self.hard_reset()
            else:
                if sum(self.last_reward)/len(self.last_reward) < 100 or self.num_timesteps < 10000:
                    self.last_reward = [0]
                    reset_prob = random.random()
                    if reset_prob > 0.25:
                        try:
                            return self.soft_reset()
                        except:
                            return self.hard_reset()
                    else:
                        return self.hard_reset()
                else:
                    self.last_reward = [0]
                    return self.hard_reset()

    def update_num_timesteps(self, num_timesteps):
        self.num_timesteps = num_timesteps

    def set_eval_mode(self, flag: bool):
        self.eval_mode = bool(flag)

    def step(self, control_dict):
        self.ev_handler.apply_control(control_dict)
        self._sa_handler.tick()
        # tick world
        t0 = time.time()
        self.world.tick()

        # update timestamp
        snap_shot = self.world.get_snapshot()
        self._timestamp['step'] = snap_shot.timestamp.frame - self._timestamp['start_frame']
        self._timestamp['frame'] = snap_shot.timestamp.frame
        self._timestamp['wall_time'] = snap_shot.timestamp.platform_timestamp
        self._timestamp['relative_wall_time'] = self._timestamp['wall_time'] - self._timestamp['start_wall_time']
        self._timestamp['simulation_time'] = snap_shot.timestamp.elapsed_seconds
        self._timestamp['relative_simulation_time'] = self._timestamp['simulation_time'] \
                                                      - self._timestamp['start_simulation_time']

        reward_dict, done_dict, info_dict = self.ev_handler.tick(self.timestamp)
        info_dict['hero']['world_tick_time'] = time.time() - t0
        self.last_reward.append(reward_dict['hero'])

        # get observations
        obs_dict = self._om_handler.get_observation(self.timestamp)

        # update weather
        self._wt_handler.tick(snap_shot.timestamp.delta_seconds)

        # num_walkers = len(self._world.get_actors().filter("*walker.pedestrian*"))
        # num_vehicles = len(self._world.get_actors().filter("vehicle*"))
        # logger.debug(f"num_walkers: {num_walkers}, num_vehicles: {num_vehicles}, ")

        ego_vehicle = next(iter(self.ev_handler.ego_vehicles.values())).vehicle
        spectator = self.world.get_spectator()

        def _follow_ego():

            if ego_vehicle is None or not ego_vehicle.is_alive:
                return  # Skip camera setup
            try:
                ego_transform = ego_vehicle.get_transform()
            except RuntimeError:
                return  # Avoid crash when actor destroyed after collision

            # Relative offset
            offset_location = carla.Location(x=-6.0, z=2.5)
            world_location = ego_transform.transform(offset_location)

            world_rotation = carla.Rotation(
                pitch=ego_transform.rotation.pitch,
                yaw=ego_transform.rotation.yaw,
                roll=ego_transform.rotation.roll
            )

            spectator.set_transform(carla.Transform(world_location, world_rotation))

        _follow_ego()

        return obs_dict, reward_dict, done_dict, info_dict

    def _init_client(self, carla_map, host, port, seed=2021, no_rendering=False):
        client = None
        while client is None:
            try:
                client = carla.Client(host, port)
                client.set_timeout(100.0)
            except RuntimeError as re:
                if "timeout" not in str(re) and "time-out" not in str(re):
                    print("Could not connect to Carla server because:", re)
                client = None

        self._client = client
        self.world = client.load_world(carla_map)

        self._tm = client.get_trafficmanager(port + 6000)

        self.set_sync_mode(True)
        self.set_no_rendering_mode(self.world, no_rendering)

        # self._tm.set_hybrid_physics_mode(True)

        # self._tm.set_global_distance_to_leading_vehicle(5.0)
        # logger.debug("trafficmanager set_global_distance_to_leading_vehicle")

        set_random_seed(self._seed, using_cuda=True)
        self._tm.set_random_device_seed(self._seed)

        self.world.tick()

        # register traffic lights
        TrafficLightHandler.reset(self.world)

    def set_sync_mode(self, sync):
        settings = self.world.get_settings()
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = 0.1
        settings.deterministic_ragdolls = True
        self.world.apply_settings(settings)
        self._tm.set_synchronous_mode(sync)

    @staticmethod
    def set_no_rendering_mode(world, no_rendering):
        settings = world.get_settings()
        settings.no_rendering_mode = no_rendering
        world.apply_settings(settings)

    @property
    def timestamp(self):
        return self._timestamp.copy()

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
        logger.debug("env __exit__!")

    def close(self):
        self.clean()
        self.set_sync_mode(False)
        self._client = None
        self.world = None
        self._tm = None

    def clean(self):
        self._sa_handler.clean()
        self._zw_handler.clean()
        self._zv_handler.clean()
        self._om_handler.clean()
        self.ev_handler.clean()
        self._wt_handler.clean()
        self.world.tick()
        gc.collect()

    def soft_clean(self):
        self._sa_handler.clean()
        self._om_handler.clean()
        self.ev_handler.clean()
        self._wt_handler.clean()
        self.world.tick()
        gc.collect()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
