from carla_gym.carla_multi_agent_env import CarlaMultiAgentEnv


class EndlessEnv(CarlaMultiAgentEnv):
    def __init__(self, carla_map, host, port, seed, no_rendering, obs_configs, reward_configs, terminal_configs,
                 num_zombie_vehicles, num_zombie_walkers, weather_group, light_always_green, disbale_bg_actors):
        all_tasks = self.build_all_tasks(num_zombie_vehicles, num_zombie_walkers, weather_group)
        super().__init__(carla_map, host, port, seed, no_rendering,
                         obs_configs, reward_configs, terminal_configs, all_tasks, light_always_green, disbale_bg_actors)
        #
        # self.observation_space = self.observation_space['hero']
        # self.action_space = self.action_space['hero']

    @staticmethod
    def build_all_tasks(num_zombie_vehicles, num_zombie_walkers, weather_group):
        if weather_group == 'new':
            weathers = ['SoftRainSunset', 'WetSunset']
        elif weather_group == 'train':
            weathers = ['ClearNoon', 'WetNoon', 'HardRainNoon', 'ClearSunset']
        elif weather_group == 'all':
            weathers = ['ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon', 'SoftRainNoon', 'MidRainyNoon',
                        'HardRainNoon', 'ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset',
                        'SoftRainSunset', 'MidRainSunset', 'HardRainSunset']
        else:
            weathers = [weather_group]

        actor_configs_dict = {
            'ego_vehicles': {
                'hero': {'model': 'vehicle.lincoln.mkz2017'}
            }
        }
        route_descriptions_dict = {
            'ego_vehicles': {
                'hero': []
            }
        }
        endless_dict = {
            'ego_vehicles': {
                'hero': True
            }
        }
        all_tasks = []
        for weather in weathers:
            task = {
                'weather': weather,
                'description_folder': 'None',
                'route_id': 0,
                'num_zombie_vehicles': num_zombie_vehicles,
                'num_zombie_walkers': num_zombie_walkers,
                'ego_vehicles': {
                    'routes': route_descriptions_dict['ego_vehicles'],
                    'actors': actor_configs_dict['ego_vehicles'],
                    'endless': endless_dict['ego_vehicles']
                },
                'scenario_actors': {},
            }
            all_tasks.append(task)

        return all_tasks
    #
    # def step(self, control):
    #     control_dict = {}
    #     control_dict['hero'] = control
    #     obs_dict, reward_dict, done_dict, info_dict = super(EndlessEnv, self).step(control_dict)
    #     return obs_dict['hero'], reward_dict['hero'], done_dict['hero'], info_dict['hero']
    #
    # def reset(self):
    #     obs_dict = super(EndlessEnv, self).reset()
    #     return obs_dict['hero']


