import time
from collections import deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, NormalActionNoise
from stable_baselines3.common.utils import polyak_update, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.td3.td3 import TD3
from typing import Any, ClassVar, Optional, TypeVar, Union
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TrainFreq, RolloutReturn, TrainFrequencyUnit
from stable_baselines3.common.buffers import ReplayBuffer
import torch as th
from torch.nn import functional as F
import random
import math
from torch.distributions import Normal

# Don't set default device globally - let Stable-Baselines3 handle device placement
# This prevents meta tensor issues
if th.cuda.is_available():
    th.set_default_device("cuda")
else:
    th.set_default_device("cpu")


def il_coef_schedule(step, start=0.02, end=0., total_steps=250_000, gamma=0.6):
    p = min(step/total_steps, 1.0)
    p = p**gamma
    decay = 0.5*(1+math.cos(math.pi*p))
    return end + (start-end)*decay


def delayed_lin_anneal(start, end, t, t0, T_total):
    if t <= t0:
        return start
    if t > T_total:
        return end
    w = min(max((t - t0) / max(T_total - t0, 1), 0.0), 1.0)
    return (1 - w) * start + w * end


class TD3Vlm(TD3):
    def __init__(
        self,
        policy: Union[str, type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 150_000,  # 1e6
        learning_starts: int = 1024,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        use_vlm: str = '',
        imitation_coef: float = 0.1,
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
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        if _init_setup_model:
            self._setup_model()

        self.use_vlm = use_vlm
        self.imitation_coef = imitation_coef
        self.cql_coefficient = 1
        self.q_value_bound = 1

        self.ep_stat_buffer = None
        self.start_num_timesteps = self.num_timesteps

        self.awac_lambda = getattr(self, "awac_lambda", 1.0)
        self.awac_wmax = getattr(self, "awac_wmax", 20.0)
        self.awac_beta = getattr(self, "awac_beta", 0.5)
        self.awac_trust_dims = getattr(self, "awac_trust_dims", None)

        self.alpha = 2.5

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        
        # Simple linear noise decay
        if hasattr(self, 'action_noise') and self.action_noise is not None:
            progress = min(self.num_timesteps / self.noise_decay_steps, 1.0)
            current_sigma = self.initial_noise_sigma - progress * (self.initial_noise_sigma - self.final_noise_sigma)
            self.action_noise.sigma = np.array([current_sigma, current_sigma], dtype=np.float32)
            if self.num_timesteps % 1000 == 0:
                print(f"Action noise sigma: {self.action_noise.sigma}")

        # Linear annealing

        if getattr(self.replay_buffer, "use_prioritized", False):
            t_start = max(self.learning_starts, int(0.5 * self.replay_buffer.buffer_size))
            t_end = int(0.8 * self._total_timesteps)
            if self.num_timesteps >= t_end:
                self.replay_buffer.beta = 1.0
            elif self.num_timesteps >= t_start:
                frac = (self.num_timesteps - t_start) / (t_end - t_start)
                self.replay_buffer.beta = 0.4 + 0.6 * frac
            else:
                self.replay_buffer.beta = 0.4  # Explicit initial value

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        td_errors = []
        if self.use_vlm:
            imitation_losses = []
        
        # Add counters to track skipped updates
        skipped_updates = 0
        total_updates = 0
        
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            if getattr(self.replay_buffer, "use_prioritized", False):
                replay_data, weights, tree_idxs = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            else:
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                weights = None
                tree_idxs = []  # Initialize empty list for non-prioritized replay

            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
                
            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss, td_error = [], []
            if 'pvp' in self.use_vlm:
                lambda_pvp = delayed_lin_anneal(start=1.0, end=0.0, t=self.num_timesteps, t0=50_000,
                                                T_total=150_000)
                current_q_values_vlm = self.critic(replay_data.observations, replay_data.vlm_actions)
                for current_q, current_q_vlm in zip(current_q_values, current_q_values_vlm):
                    l_ = 0.5 * th.mean((current_q - target_q_values)**2)
                    pvp_loss = (replay_data.has_vlm_actions * F.mse_loss(current_q_vlm, current_q.detach() + 2)).mean()
                    l = l_ + lambda_pvp * pvp_loss
                    td = 0.5 * th.abs(current_q - target_q_values).detach()
                    critic_loss.append(l)
                    td_error.append(td)
            else:
                for current_q_novice in current_q_values:
                    l = 0.5 * F.mse_loss(current_q_novice, target_q_values)
                    td = 0.5 * th.abs(current_q_novice - target_q_values).detach()
                    critic_loss.append(l)
                    td_error.append(td)

            critic_loss = sum(critic_loss)
            assert isinstance(critic_loss, th.Tensor)
                
            critic_losses.append(critic_loss.item())
            td_errors.append(sum(td_error))
            
            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            
            total_updates += 1

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actions_pi = self.actor(replay_data.observations)
                if 'awac' in self.use_vlm:
                    Q1 = self.critic.q1_forward(replay_data.observations, actions_pi)
                    lmbda = self.alpha / Q1.abs().mean().detach()
                    base_actor_loss = - lmbda * Q1.mean()
                else:
                    base_actor_loss = -self.critic.q1_forward(replay_data.observations, actions_pi).mean()
                actor_losses.append(base_actor_loss.item())

                # awac_loss = th.zeros_like(base_actor_loss)
                # if 'awac' in self.use_vlm:
                #     has = replay_data.has_vlm_actions
                #     if isinstance(has, th.Tensor):
                #         mask = has > 0
                #     else:
                #         mask = th.tensor(has, dtype=th.bool, device=replay_data.actions.device)
                #
                #     mask = mask.bool()
                #     if mask.ndim > 1:
                #         mask = mask.squeeze(-1)
                #
                #     if mask.any():
                #         obs = replay_data.observations
                #         if isinstance(obs, dict):
                #             s_b = {k: v[mask] for k, v in obs.items()}
                #         else:
                #             s_b = obs[mask]
                #
                #         a_exp = replay_data.vlm_actions[mask]
                #
                #         # 1. Advantage-based weighting (AWAC)
                #         with th.no_grad():
                #             q_exp1, q_exp2 = self.critic(s_b, a_exp)
                #             q_exp = 0.5 * (q_exp1 + q_exp2)
                #
                #             a_pi_b = self.actor(s_b)
                #             q_pi1_b, q_pi2_b = self.critic(s_b, a_pi_b)
                #             q_pi_b = 0.5 * (q_pi1_b + q_pi2_b)
                #
                #             adv = (q_exp - q_pi_b).squeeze(-1)
                #             gate = adv > 0.0
                #             weights = th.exp(adv / 2.0)  # β≈2
                #             weights = th.clamp(weights, max=20.0)
                #
                #         # 2. Fixed std imitation loss
                #         mu = self.actor(s_b)
                #         std = th.ones_like(mu) * 0.5
                #         eps = 1e-6
                #
                #         # Inverse tanh: map [-1,1] back to uncompressed space
                #         pre_tanh_mu = 0.5 * th.log((1 + mu + eps) / (1 - mu + eps))
                #         pre_tanh_a_exp = 0.5 * th.log((1 + a_exp.clamp(-0.999, 0.999) + eps) /
                #                                       (1 - a_exp.clamp(-0.999, 0.999) + eps))
                #
                #         dist = Normal(pre_tanh_mu, std)
                #         # log_prob + tanh Jacobian
                #         log_prob = dist.log_prob(pre_tanh_a_exp).sum(-1) - th.log(1 - a_exp.pow(2) + eps).sum(-1)
                #         imitation_loss = -log_prob
                #
                #         # 3. Combine imitation loss with AWAC weights
                #         if gate.any():
                #             awac_loss = (weights.detach() * imitation_loss)[gate].mean()
                #         else:
                #             awac_loss = th.zeros_like(base_actor_loss)
                #
                #     else:
                #         awac_loss = th.zeros_like(base_actor_loss)

                awac_loss = th.zeros_like(base_actor_loss)
                if 'awac' in self.use_vlm:
                    has = replay_data.has_vlm_actions
                    if isinstance(has, th.Tensor):
                        mask = has > 0
                    else:
                        mask = th.tensor(has, dtype=th.bool, device=replay_data.actions.device)

                    # Flatten to 1D bool
                    mask = mask.bool()
                    if mask.ndim > 1:
                        mask = mask.squeeze(-1)

                    if mask.any():
                        # 1) Extract masked obs
                        obs = replay_data.observations
                        if isinstance(obs, dict):
                            s_b = {k: v[mask] for k, v in obs.items()}
                        else:
                            s_b = obs[mask]

                        # 2) Mask expert actions too
                        a_exp = replay_data.vlm_actions[mask]

                        # next_obs = replay_data.next_observations
                        # next_s_b = {k: v[mask] for k, v in next_obs.items()} if isinstance(next_obs, dict) else next_obs[mask]

                        with th.no_grad():
                            q_exp1, q_exp2 = self.critic(s_b, a_exp)
                            q_exp = 0.5 * (q_exp1 + q_exp2)

                            a_pi_b = self.actor(s_b)
                            q_pi1_b, q_pi2_b = self.critic(s_b, a_pi_b)
                            q_pi_b = 0.5 * (q_pi1_b + q_pi2_b)

                            adv = (q_exp - q_pi_b).squeeze(-1)
                            gate = adv > 0.0  # Q-filter
                            weights = th.exp(adv / 2.0)
                            weights = th.clamp(weights, max=20.0)

                        pi_b = self.actor(s_b)
                        diff = pi_b - a_exp
                        awac_loss = (weights.view(-1, 1).detach() * (diff ** 2))[gate].mean() if gate.any() else th.zeros_like(
                            base_actor_loss)
                    else:
                        awac_loss = th.zeros_like(base_actor_loss)

                # 4) Combine actor loss
                il = il_coef_schedule(self.num_timesteps)
                actor_loss = base_actor_loss  + il * awac_loss

                if 'pvp' in self.use_vlm:
                    imitation_losses.append(pvp_loss.item())
                if 'awac' in self.use_vlm:
                    imitation_losses.append(awac_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)
        
        # Ensure we have valid losses before computing means
        self.train_debug = {
            "train/actor_loss": np.mean(actor_losses),
            "train/critic_loss": np.mean(critic_losses),
        }
        if 'pvp' in self.use_vlm or 'awac' in self.use_vlm:
            self.train_debug.update({"train/imitation_loss": np.mean(imitation_losses)})

        if getattr(self.replay_buffer, "use_prioritized", False):
            # Check if we have valid TD errors and tree indices before updating priorities
            if len(td_errors) > 0 and len(tree_idxs) > 0:
                avg_td_error = sum(td_errors) / len(td_errors)
                # Use environment indices if available for better multi-env priority updates
                if hasattr(replay_data, 'env_indices') and replay_data.env_indices is not None:
                    self.replay_buffer.update_priorities(tree_idxs, avg_td_error, replay_data.env_indices)
                else:
                    self.replay_buffer.update_priorities(tree_idxs, avg_td_error)
            else:
                print(f"[WARNING] Cannot update priorities: td_errors={len(td_errors)}, tree_idxs={len(tree_idxs)}")

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
