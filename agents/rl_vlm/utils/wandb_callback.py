"""WandB logging callbacks: checkpointing, evaluation, video capture."""
import os.path

import numpy as np
import time
from pathlib import Path
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from omegaconf import OmegaConf
from utils.os_utils import find_project_root
import imageio
import torch as th


class ImageEncoder:
    def __init__(self, output_path, frame_shape, fps=30):
        self.writer = imageio.get_writer(output_path, fps=fps, codec='libx264')
        self.height, self.width = frame_shape[:2]

    def capture_frame(self, frame):
        if frame.dtype != np.uint8:
            frame_min = frame.min()
            frame_max = frame.max()
            frame = (frame - frame_min) / (frame_max - frame_min)
            frame = (255 * np.clip(frame, 0, 1)).astype(np.uint8)
        self.writer.append_data(frame)

    def close(self):
        self.writer.close()


class WandbCallback(BaseCallback):
    def __init__(self, cfg, vec_env, local_output_dir=None):
        super(WandbCallback, self).__init__(verbose=1)
        self.local_output_dir = local_output_dir
        # save_dir = Path.cwd()
        # self._save_dir = save_dir
        self._video_path = Path('video')
        self._video_path.mkdir(parents=True, exist_ok=True)
        self._ckpt_dir = Path('ckpt')
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

        # wandb.init(project=cfg.wb_project, dir=save_dir, name=cfg.wb_runname)
        try:
            run_id = cfg.agent.rl_vlm.wb_run_path.split('/')[-1]
            print(f'[INFO] Resume wandb from {run_id}')
        except:
            run_id = None
        wandb.init(project=cfg.wb_project, name=cfg.wb_name, notes=cfg.wb_notes, tags=cfg.wb_tags,
                   id=run_id, resume="allow")
        wandb.config.update(OmegaConf.to_container(cfg), allow_val_change=True)
        wandb.save('./config_agent.yaml')
        wandb.save('.hydra/*')

        self.vec_env = vec_env

        self._eval_step = int(5e4)  # 1e5
        self._save_step = int(5e4)  # carla can easily crash
        self._buffer_step = int(5e4)  # 1e5
        self._init_save = False

    def _init_callback(self):
        self.n_epoch = 0
        self._last_time_buffer = self.model.num_timesteps
        self._last_time_eval = self.model.num_timesteps
        self._last_time_save = self.model.num_timesteps

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self):
        pass

    def _on_training_end(self) -> None:
        time_elapsed = time.time() - self.model.start_time
        wandb.log({
            'time/n_epoch': self.n_epoch,
            'time/sec_per_epoch': time_elapsed / (self.n_epoch+1),
            'time/fps': (self.model.num_timesteps-self.model.start_num_timesteps) / time_elapsed,
            'time/train': self.model.t_train,
            'time/rollout': self.model.t_rollout,
            # 'time/t_world_tick': self.model.t_world_tick
        }, step=self.model.num_timesteps)
        wandb.log(self.model.train_debug, step=self.model.num_timesteps)

        if (self.model.num_timesteps - self._last_time_save) >= self._save_step or self._init_save:
            self._last_time_save = self.model.num_timesteps
            ckpt_path = (self._ckpt_dir / f'ckpt_{self.model.num_timesteps}').as_posix()
            self.model.save(ckpt_path)
            wandb.save(f'./{ckpt_path}')
            if hasattr(self.model, "save_replay_buffer"):
                self.model.save_replay_buffer(self.local_output_dir / 'replay_buffer.pkl')
            with open(self.local_output_dir / "num_timesteps.txt", "w") as f:
                f.write(str(self.model.num_timesteps))

        # evaluate and save checkpoint
        if (self.model.num_timesteps - self._last_time_eval) >= self._eval_step or self._init_save:
            self._last_time_eval = self.model.num_timesteps
            # evaluate
            eval_video_path = (self._video_path / f'eval_{self.model.num_timesteps}.mp4').as_posix()
            # avg_ep_stat, ep_events = self.evaluate_policy(self.vec_env, self.model.policy, eval_video_path)
            # # log to wandb
            # try:
            #     wandb.log({f'video/{self.model.num_timesteps}': wandb.Video(eval_video_path)},
            #               step=self.model.num_timesteps)
            # except:
            #     print('[INFO] RGB CAMERA is not enabled, front view video will not be logged to WANDB!')
            # wandb.log(avg_ep_stat, step=self.model.num_timesteps)
            #
            ckpt_path = (self._ckpt_dir / f'ckpt_{self.model.num_timesteps}').as_posix()
            self.model.save(ckpt_path)
            wandb.save(f'./{ckpt_path}.zip')
        self.n_epoch += 1
        self._init_save = False

    def _on_rollout_end(self):
        # save rollout statistics
        avg_ep_stat = self.get_avg_ep_stat(self.model.ep_stat_buffer, prefix='rollout/')
        wandb.log(avg_ep_stat, step=self.model.num_timesteps)

        action_statistics = np.array(self.model.action_statistics)
        n_action = action_statistics.shape[-1]
        action_statistics = action_statistics.reshape(-1, n_action)
        
        # Filter out NaN values before creating histograms
        for i in range(n_action):
            action_data = action_statistics[:, i]
            # Remove NaN values
            valid_actions = action_data[~np.isnan(action_data)]
            
            if len(valid_actions) > 0:
                # Only log histogram if we have valid data
                wandb.log({f'action[{i}]': wandb.Histogram(valid_actions)}, step=self.model.num_timesteps)
            else:
                # Log a warning if no valid actions
                wandb.log({f'action[{i}]_warning': 'No valid actions (all NaN)'}, step=self.model.num_timesteps)

        # render buffer
        # if (self.model.num_timesteps - self._last_time_buffer) >= self._buffer_step:
        #     self._last_time_buffer = self.model.num_timesteps
        #     buffer_video_path = (self._video_path / f'buffer_{self.model.num_timesteps}.mp4').as_posix()
        #
        #     if hasattr(self.model, 'replay_buffer'):
        #         list_buffer_im = self.model.replay_buffer.render()
        #     else:
        #         list_buffer_im = self.model.rollout_buffer.render()
        #     self._video_path.mkdir(parents=True, exist_ok=True)
        #     encoder = ImageEncoder(buffer_video_path, list_buffer_im[0].shape, 30)
        #     for im in list_buffer_im:
        #         encoder.capture_frame(im)
        #     encoder.close()
        #     encoder = None
        #
        #     wandb.log({f'buffer/{self.model.num_timesteps}': wandb.Video(buffer_video_path)},
        #               step=self.model.num_timesteps)

    @staticmethod
    def evaluate_policy(env, policy, video_path, min_eval_steps=3000):
        policy = policy.eval()
        t0 = time.time()
        for i in range(env.num_envs):
            env.set_attr('eval_mode', True, indices=i)
        env.env_method('set_eval_mode', True)
        obs = env.reset()

        list_render = []
        ep_stat_buffer = []
        ep_events = {}
        for i in range(env.num_envs):
            ep_events[f'venv_{i}'] = []

        n_step = 0
        n_timeout = 0
        env_done = np.array([False]*env.num_envs)
        # while n_step < min_eval_steps:
        while n_step < min_eval_steps or not np.all(env_done):
            actions, _states = policy.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(actions)

            list_render.append(env.render(mode='rgb_array'))

            n_step += 1
            env_done |= done

            for i in np.where(done)[0]:
                ep_stat_buffer.append(info[i]['episode_stat'])
                ep_events[f'venv_{i}'].append(info[i]['episode_event'])
                n_timeout += int(info[i]['timeout'])

        # conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
        try:
            encoder = ImageEncoder(video_path, list_render[0].shape, 30)
            for im in list_render:
                encoder.capture_frame(im)
            encoder.close()
        except:
            print('[INFO] RGB CAMERA is not enabled, front view video will not be saved!')

        avg_ep_stat = WandbCallback.get_avg_ep_stat(ep_stat_buffer, prefix='eval/')
        avg_ep_stat['eval/eval_timeout'] = n_timeout

        duration = time.time() - t0
        avg_ep_stat['time/t_eval'] = duration
        avg_ep_stat['time/fps_eval'] = n_step * env.num_envs / duration

        for i in range(env.num_envs):
            env.set_attr('eval_mode', False, indices=i)
        env.env_method('set_eval_mode', False)
        obs = env.reset()
        return avg_ep_stat, ep_events

    @staticmethod
    def get_avg_ep_stat(ep_stat_buffer, prefix=''):
        avg_ep_stat = {}
        if len(ep_stat_buffer) > 0:
            for ep_info in ep_stat_buffer:
                for k, v in ep_info.items():
                    k_avg = f'{prefix}{k}'
                    if k_avg in avg_ep_stat:
                        avg_ep_stat[k_avg] += v
                    else:
                        avg_ep_stat[k_avg] = v

            n_episodes = float(len(ep_stat_buffer))
            for k in avg_ep_stat.keys():
                avg_ep_stat[k] /= n_episodes
            avg_ep_stat[f'{prefix}n_episodes'] = n_episodes

        return avg_ep_stat
