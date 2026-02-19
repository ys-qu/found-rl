"""Evaluate trained RL+VLM agents on CARLA benchmarks."""
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
import os
import sys

from stable_baselines3.common.vec_env.base_vec_env import tile_images
from agents.rl_vlm.utils.wandb_callback import ImageEncoder

from carla_gym.utils import config_utils
from utils import saving_utils, server_utils
from agents.rl_vlm.utils.wandb_callback import WandbCallback

log = logging.getLogger(__name__)


def eval_single(run_name, env, driver_dict, driver_log_dir, log_video):

    list_debug_render = []
    ep_stat_dict = {}
    ep_event_dict = {}
    for actor_id, driver in driver_dict.items():
        log_dir = driver_log_dir / actor_id  # driver_log/hero
        log_dir.mkdir(parents=True, exist_ok=True)
        driver.reset(log_dir / f'{run_name}.log')

    obs = env.reset()
    timestamp = env.timestamp
    done = False
    while not done:
        driver_control = {}
        driver_supervision = {}

        for actor_id, driver in driver_dict.items():
            driver_control[actor_id] = driver.run_step(obs, timestamp)
            driver_supervision[actor_id] = driver.supervision_dict

        new_obs, reward, done, info = env.step(driver_control[actor_id])

        obs = new_obs

        debug_imgs = []
        for actor_id, driver in driver_dict.items():
            if log_video:
                debug_imgs.append(driver.render(info['central_rgb'], info['reward_debug'], info['terminal_debug']))
            if done and (actor_id not in ep_stat_dict):
                episode_stat = info['episode_stat']
                ep_stat_dict[actor_id] = episode_stat
                ep_event_dict[actor_id] = info['episode_event']

                log.info(f'Episode {run_name} done')

        if log_video:
            list_debug_render.append(tile_images(debug_imgs))
        timestamp = env.timestamp

    return list_debug_render, ep_stat_dict, ep_event_dict, timestamp


@ hydra.main(config_path='config', config_name='eval_rl_vlm')
def main(cfg: DictConfig):
    if cfg.host == 'localhost' and cfg.kill_running:
        server_utils.kill_carla()
    log.setLevel(getattr(logging, cfg.log_level.upper()))

    # start carla servers
    server_manager = server_utils.CarlaServerManager(cfg.carla_sh_path, port=cfg.port)
    server_manager.start()

    # single actor, place holder for multi actors
    driver_dict = {}
    obs_configs = {}
    reward_configs = {}
    terminal_configs = {}
    for ev_id, ev_cfg in cfg.actors.items():
        # initiate driver agent
        cfg_driver = cfg.agent[ev_cfg.driver]
        OmegaConf.save(config=cfg_driver, f='config_driver.yaml')
        DriverAgentClass = config_utils.load_entry_point(cfg_driver.entry_point)
        driver_dict[ev_id] = DriverAgentClass('config_driver.yaml')
        obs_configs[ev_id] = driver_dict[ev_id].obs_configs

        # get obs_configs from agent
        reward_configs[ev_id] = OmegaConf.to_container(ev_cfg.reward)
        terminal_configs[ev_id] = OmegaConf.to_container(ev_cfg.terminal)

    # check h5 birdview maps have been generated
    config_utils.check_h5_maps(cfg.test_suites, obs_configs, cfg.carla_sh_path)

    # resume env_idx from checkpoint.txt
    last_checkpoint_path = f'{hydra.utils.get_original_cwd()}/outputs/checkpoint_eval.txt'
    if cfg.resume and os.path.isfile(last_checkpoint_path):
        with open(last_checkpoint_path, 'r') as f:
            env_idx = int(f.read())
    else:
        env_idx = 0

    # resume task_idx from ep_stat_buffer_{env_idx}.json
    ep_state_buffer_json = f'{hydra.utils.get_original_cwd()}/outputs/ep_stat_buffer_{env_idx}.json'
    if cfg.resume and os.path.isfile(ep_state_buffer_json):
        ep_stat_buffer = json.load(open(ep_state_buffer_json, 'r'))
        ckpt_task_idx = len(ep_stat_buffer['hero'])
    else:
        ckpt_task_idx = 0
        ep_stat_buffer = {}
        for actor_id in driver_dict.keys():
            ep_stat_buffer[actor_id] = []

    # resume wandb run
    wb_checkpoint_path = f'{hydra.utils.get_original_cwd()}/outputs/wb_run_id_eval.txt'
    if cfg.resume and os.path.isfile(wb_checkpoint_path):
        with open(wb_checkpoint_path, 'r') as f:
            wb_run_id = f.read()
    else:
        wb_run_id = None

    log.info(f'Start from env_idx: {env_idx}, task_idx {ckpt_task_idx}')

    wb_run_name = cfg.wb_name

    diags_dir = Path('diagnostics')
    driver_log_dir = Path('driver_log')
    video_dir = Path('videos')
    diags_dir.mkdir(parents=True, exist_ok=True)
    driver_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    # init wandb
    wandb.init(project=cfg.wb_project, name=wb_run_name, group=cfg.wb_group, notes=cfg.wb_notes, tags=cfg.wb_tags,
               id=wb_run_id, resume="allow")
    wandb.config.update(OmegaConf.to_container(cfg), allow_val_change=True)
    wandb.save('./config_agent.yaml')
    with open(wb_checkpoint_path, 'w') as f:
        f.write(wandb.run.id)

    if env_idx >= len(cfg.test_suites):
        log.info(f'Finished! env_idx: {env_idx}, resave to wandb')
        return

    # make env
    env_setup = OmegaConf.to_container(cfg.test_suites[env_idx])
    EnvWrapper = config_utils.load_entry_point(cfg_driver.env_wrapper.entry_point)

    wrapper_kargs = cfg_driver.env_wrapper.kwargs
    EnvWrapper.set_birdview_key(wrapper_kargs['birdview_key'])  # must call outside, or else will be None
    EnvWrapper.set_compress_size(wrapper_kargs['compress_size'])
    env = gym.make(env_setup['env_id'], obs_configs=obs_configs, reward_configs=reward_configs,
                   terminal_configs=terminal_configs, host=cfg.host, port=cfg.port,
                   seed=cfg.seed, no_rendering=True, light_always_green=cfg.light_always_green,
                    disbale_bg_actors=cfg.disbale_bg_actors, **env_setup['env_configs'])
    env = EnvWrapper(env, **wrapper_kargs)

    # init eval policy
    for actor_id, driver in driver_dict.items():
        driver.init_eval_policy(env, cfg.local_ckpt_path)

    # main loop
    n_episodes_per_env = math.ceil(cfg.n_episodes/len(cfg.test_suites))

    for task_idx in range(ckpt_task_idx, n_episodes_per_env):
        idx_episode = task_idx + n_episodes_per_env * env_idx
        run_name = f'{idx_episode:04}'

        env.set_task_idx(task_idx % env.num_tasks)

        log.info(f'Start episode {run_name}, {env_setup}')

        list_debug_render, ep_stat_dict, ep_event_dict, timestamp = eval_single(
                run_name, env, driver_dict, driver_log_dir, cfg.log_video)

        # log videos
        if cfg.log_video:
            debug_video_path = (video_dir / f'debug_{run_name}.mp4').as_posix()
            encoder = ImageEncoder(debug_video_path, list_debug_render[0].shape, 30, 30)
            for im in list_debug_render:
                encoder.capture_frame(im)
            encoder.close()
            wandb.log({f'video/debug_{run_name}': wandb.Video(debug_video_path)}, step=idx_episode)
            encoder = None

        # dump events
        diags_json_path = (diags_dir / f'{run_name}.json').as_posix()
        with open(diags_json_path, 'w') as fd:
            json.dump(ep_event_dict, fd, indent=4, sort_keys=False)

        # save diags and agents_log
        wandb.save(diags_json_path)
        wandb.save(f'{driver_log_dir.as_posix()}/*/*')

        # save time
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
        # clean up
        list_debug_render.clear()
        ep_stat_dict = None
        ep_event_dict = None

    # close env
    env.close()
    env = None
    server_manager.stop()

    # log after all episodes are completed
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

    log.info(f"Finished eval env_idx {env_idx}, {env_setup['env_id']}.")
    if env_idx+1 == len(cfg.test_suites):
        log.info(f"Finished, {env_idx+1}/{len(cfg.test_suites)}")
        return
    else:
        log.info(f"Not finished, {env_idx+1}/{len(cfg.test_suites)}")
        sys.exit(1)

    return


if __name__ == '__main__':
    main()
    log.info("eval_rl_vlm.py DONE!")