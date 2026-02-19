import time
from collections import deque

from stable_baselines3.sac.sac import SAC
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TrainFreq, RolloutReturn, TrainFrequencyUnit
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
import math


def delayed_lin_anneal(start, end, t, t0, T_total):
    if t <= t0:
        return start
    if t > T_total:
        return end
    w = min(max((t - t0) / max(T_total - t0, 1), 0.0), 1.0)
    return (1 - w) * start + w * end


# def il_coef_schedule(step, start=0.03, end=0., total_steps=200000, gamma=0.7):
#     p = min(step/total_steps, 1.0)
#     p = p**gamma
#     decay = 0.5*(1+math.cos(math.pi*p))
#     return end + (start-end)*decay

def il_coef_schedule(step,
                     total_steps=300_000,
                     delay=20_000,
                     peak_pos=0.45,
                     pre=0.001,
                     peak=0.03,
                     after=0.0):
    peak_step = int(total_steps * peak_pos)
    step = int(step)
    if step <= delay:
        return float(pre)
    if step <= peak_step:
        t = (step - delay) / max(1, (peak_step - delay))
        t = min(max(t, 0.0), 1.0)
        s = 0.5 * (1 - math.cos(math.pi * t))  # 0->1
        return float(pre + (peak - pre) * s)
    t2 = (step - peak_step) / max(1, (total_steps - peak_step))
    t2 = min(max(t2, 0.0), 1.0)
    s2 = 0.5 * (1 + math.cos(math.pi * t2))   # 1->0
    return float(after + (peak - after) * s2)

# Don't set default device globally - let Stable-Baselines3 handle device placement
# This prevents meta tensor issues

class SACVlm(SAC):
    def __init__(
            self,
            policy: Union[str, type[SACPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            buffer_size: int = 150_000,  # 1e6
            learning_starts: int = 100,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, tuple[int, str]] = 1,
            gradient_steps: int = 1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[type[ReplayBuffer]] = None,
            replay_buffer_kwargs: Optional[dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            ent_coef: Union[str, float] = "auto",
            target_update_interval: int = 1,
            target_entropy: Union[str, float] = "auto",
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            use_sde_at_warmup: bool = False,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            use_vlm: str = '',
            imitation_coef: float = 0.1
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None

        if _init_setup_model:
            self._setup_model()

        self.use_vlm = use_vlm
        self.imitation_coef = imitation_coef

        self.ep_stat_buffer = None
        self.start_num_timesteps = self.num_timesteps

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        if self.use_vlm:
            imitation_losses = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            if 'r_clip' in self.use_vlm:
                bonus = replay_data.clip_safety_scores  # (N, n_envs)
                replay_data.rewards.add_(bonus)
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            if 'pvp' in self.use_vlm:
                lambda_pvp = delayed_lin_anneal(start=1.0, end=0.0, t=self.num_timesteps, t0=50_000,
                                                T_total=150_000)
                current_q_values_vlm = self.critic(replay_data.observations, replay_data.vlm_actions)
                critic_loss = []
                for current_q, current_q_vlm in zip(current_q_values, current_q_values_vlm):
                    l_ = 0.5 * th.mean((current_q - target_q_values)**2)
                    pvp_loss = (replay_data.has_vlm_actions * F.mse_loss(current_q_vlm, current_q.detach() + 2)).mean()
                    l = l_ + lambda_pvp * pvp_loss
                    critic_loss.append(l)
                critic_loss = sum(critic_loss)
            else:
                critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            base_actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(base_actor_loss.item())

            # AWAC loss - following official implementation
            awac_loss = th.zeros_like(base_actor_loss)

            if 'awac' in self.use_vlm:
                has = replay_data.has_vlm_actions
                if isinstance(has, th.Tensor):
                    mask = has > 0
                else:
                    mask = th.tensor(has, dtype=th.bool, device=replay_data.actions.device)

                # Convert to 1D bool vector
                mask = mask.bool()
                if mask.ndim > 1:
                    mask = mask.squeeze(-1)

                if mask.any():
                    # 1) Extract masked observations and actions
                    obs = replay_data.observations
                    if isinstance(obs, dict):
                        s_b = {k: v[mask] for k, v in obs.items()}
                    else:
                        s_b = obs[mask]
                    eps = 1e-6
                    vlm_actions = replay_data.vlm_actions[mask].clamp(-1 + eps, 1 - eps)  # Expert actions

                    with th.no_grad():
                        pi_act_det = self.actor.forward(s_b, deterministic=True)

                        q1_pi, q2_pi = self.critic(s_b, pi_act_det)
                        v_pi = th.min(q1_pi, q2_pi)  # [B,1]

                        q1_exp, q2_exp = self.critic(s_b, vlm_actions)
                        q_exp = th.min(q1_exp, q2_exp)  # [B,1]

                        adv = (q_exp - v_pi).squeeze(-1)
                        gate = adv > 0.0  # Q-filter
                        weights = th.exp(adv / 2.0)
                        weights = th.clamp(weights, max=20.0)

                    mean, log_std, kwargs = self.actor.get_action_dist_params(s_b)
                    dist = self.actor.action_dist.proba_distribution(mean, log_std, **kwargs)
                    vlm_actions = vlm_actions.clamp(-0.999, 0.999)
                    logp_exp = dist.log_prob(vlm_actions).sum(-1)
                    awac_loss = - (weights.detach() * logp_exp)[gate].mean() if gate.any() else th.zeros_like(
                        base_actor_loss)

            # Combine actor loss
            # il = il_coef_schedule(self.num_timesteps)
            # il = awac_coef(self.num_timesteps)
            actor_loss = base_actor_loss + 0.01 * awac_loss  # il *
            if 'awac' in self.use_vlm:
                imitation_losses.append(awac_loss.item())
            if 'pvp' in self.use_vlm:
                imitation_losses.append(pvp_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        # self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # self.logger.record("train/ent_coef", np.mean(ent_coefs))
        # self.logger.record("train/actor_loss", np.mean(actor_losses))
        # self.logger.record("train/critic_loss", np.mean(critic_losses))
        # if len(ent_coef_losses) > 0:
        #     self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        self.train_debug = {
            "train/actor_loss": np.mean(actor_losses),
            "train/critic_loss": np.mean(critic_losses),
            "train/ent_coef": np.mean(ent_coefs),
        }
        if 'r_clip' in self.use_vlm:
            self.train_debug.update({"train/clip_socres_percentage>0":
                                         (self.replay_buffer.clip_safety_scores[:self.replay_buffer.pos if not self.replay_buffer.full else self.buffer_size]>0).mean()})
            # self.train_debug.update({"train/clip_socres_percentage>0.5":
            #                              (self.replay_buffer.clip_safety_scores[
            #                               :self.replay_buffer.pos if not self.replay_buffer.full else self.buffer_size] > 0.5).mean()})
            # self.train_debug.update({"train/clip_socres_percentage>0.6":
            #                              (self.replay_buffer.clip_safety_scores[:self.replay_buffer.pos if not self.replay_buffer.full else self.buffer_size]>0.6).mean()})
            # self.train_debug.update({"train/clip_socres_percentage>0.7":
            #                              (self.replay_buffer.clip_safety_scores[:self.replay_buffer.pos if not self.replay_buffer.full else self.buffer_size]>0.7).mean()})
            # self.train_debug.update({"train/clip_socres_percentage>0.8":
            #                              (self.replay_buffer.clip_safety_scores[:self.replay_buffer.pos if not self.replay_buffer.full else self.buffer_size]>0.8).mean()})
            # self.train_debug.update({"train/clip_socres_percentage>0.9":
            #                              (self.replay_buffer.clip_safety_scores[:self.replay_buffer.pos if not self.replay_buffer.full else self.buffer_size]>0.9).mean()})
            self.train_debug.update({"train/clip_socres_average":
                                         (self.replay_buffer.clip_safety_scores[:self.replay_buffer.pos if not self.replay_buffer.full else self.buffer_size]).mean()})
        if len(ent_coef_losses) > 0:
            self.train_debug.update({"train/ent_coef_loss": np.mean(ent_coef_losses)})
        if self.use_vlm:
            self.train_debug.update({"train/imitation_loss": np.mean(imitation_losses)})

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        self.ep_stat_buffer = deque(maxlen=100)
        self.t_train = 0.
        self.t_rollout = 0.
        self.t_world_tick = 0.

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"
        assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()

        while self.num_timesteps < total_timesteps:
            t0 = time.time()
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )
            self.t_rollout = time.time() - t0

            if not rollout.continue_training:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    t0 = time.time()
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
                    self.t_train = time.time() - t0

                    callback.on_training_end()

        return self

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: ReplayBuffer,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        self.action_statistics = []

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)  # type: ignore[operator]

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)  # type: ignore[operator]

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            self.action_statistics.append(actions)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)
            self.t_world_tick = sum([info['world_tick_time'] for info in infos]) / len([info['world_tick_time'] for info in infos])

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes,
                                     continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # update_info_buffer
            for i in np.where(dones)[0]:
                self.ep_stat_buffer.append(infos[i]['episode_stat'])

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones,
                                   infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self.dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)