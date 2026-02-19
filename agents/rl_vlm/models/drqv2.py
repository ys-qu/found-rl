import threading
import time
import torch as th
import numpy as np
from collections import deque

from gymnasium import spaces
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import ContinuousCritic, BasePolicy
from stable_baselines3.common.type_aliases import Schedule, GymEnv, MaybeCallback, TrainFreq, TrainFrequencyUnit, \
    RolloutReturn
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update, should_collect_more_steps
from sympy.physics.control.control_plots import plt
from torch.nn import functional as F

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from .replay_buffer import ReplayBuffer
from typing import Union, Optional, Any, ClassVar

import pathlib
import io
import re
import gc
import math
import random

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from agents.rl_vlm.models.drqv2_policy import DrQv2Policy, Actor, MultiInputPolicy, CnnPolicy, MlpPolicy
from agents.rl_vlm.models.augs import RandomRotateAug, RandomShiftsAug
from dataclasses import dataclass

if th.cuda.is_available():
    th.set_default_device("cuda")
else:
    th.set_default_device("cpu")


def atanh(x, eps=1e-6):
    x = x.clamp(-1+eps, 1-eps)
    return 0.5 * (th.log1p(x) - th.log1p(-x))

def delayed_lin_anneal(start, end, t, t0, T_total):
    if t <= t0:
        return start
    if t > T_total:
        return end
    w = min(max((t - t0) / max(T_total - t0, 1), 0.0), 1.0)
    return (1 - w) * start + w * end

# def rbf(x, y, sigma=0.5):
#     x2 = (x**2).sum(-1, keepdim=True)
#     y2 = (y**2).sum(-1, keepdim=True)
#     d  = x2 - 2*x@y.transpose(0,1) + y2.transpose(0,1)
#     return th.exp(-d/(2*sigma**2))

def rbf(x, y, sigma):
    d2 = th.cdist(x, y, p=2).pow(2)
    return th.exp(-d2 / (2 * sigma**2))

def mmd2_conditional(z, x, y, sigma_s=None, sigma_a=None):
    """MMD with state conditioning. z: state embed [B,D]; x=atanh(pi(s)) [B,A]; y=atanh(a_exp) [B,A]"""
    with th.no_grad():
        if sigma_s is None:
            Ds = th.cdist(z, z, p=2);  sigma_s = Ds.median().clamp_min(1e-3)
        if sigma_a is None:
            Da = th.cdist(y, y, p=2);  sigma_a = Da.median().clamp_min(1e-3)

    Ks = rbf(z, z, sigma_s)
    Kxx, Kyy, Kxy = rbf(x,x,sigma_a), rbf(y,y,sigma_a), rbf(x,y,sigma_a)

    # Unbiased: exclude diagonal
    n = x.size(0); eye = th.eye(n, device=x.device, dtype=x.dtype)
    Ks_off = Ks * (1 - eye)
    denom = (Ks_off.sum() + 1e-8)

    term_xx = (Ks_off * Kxx).sum() / denom
    term_yy = (Ks_off * Kyy).sum() / denom
    term_xy = (Ks * Kxy).sum()   / (Ks.sum() + 1e-8)

    return term_xx + term_yy - 2.0 * term_xy

# step, start=1, end=0.1, total_steps=10000, gamma=0.5
def il_coef_schedule(step, start=0.03, end=0., total_steps=300000, gamma=0.7):
    p = min(step/total_steps, 1.0)
    p = p**gamma
    decay = 0.5*(1+math.cos(math.pi*p))
    return end + (start-end)*decay


def awac_coef(step, start=1.0, end=0.01, total_steps=100_000, start_step=50_000):
    """
    Exponential decay: alpha_t = start * kappa**t.
    Anneals from start_step to start_step+total_steps, reaching end.
    """
    if step <= start_step:
        return 0.
    if step >= start_step + total_steps:
        return float(end)
    t = step - start_step
    # Avoid log(0)
    kappa = math.exp(math.log((end + 1e-12) / (start + 1e-12)) / float(total_steps))
    return float(start * (kappa ** t))


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


def expectile_loss(diff, expectile=0.8):
    weight = th.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class CostEma:
    def __init__(self, initial=0):
        self.value = initial


import torch
import torch.distributed as dist

class AugLagrange(torch.nn.Module):
    """
    Augmented Lagrangian (AL) multiplier updater.
    λ: Lagrange multiplier (dual, >=0); c: penalty multiplier (non-decreasing).
    Ref: Nocedal & Wright, Numerical Optimization, Eq. 17.65
    """
    def __init__(self, lagrange_multiplier_init=1e-6, penalty_multiplier_init=5e-9, cost_limit=25.0, device=None, dtype=torch.float32):
        super().__init__()
        self.cost_limit = float(cost_limit)
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Parameter for save/load; updates are manual, no backprop
        self.lagrange_multiplier = torch.nn.Parameter(torch.full((), lagrange_multiplier_init, dtype=dtype, device=device), requires_grad=False)
        self.penalty_multiplier  = torch.nn.Parameter(torch.full((), penalty_multiplier_init,  dtype=dtype, device=device), requires_grad=False)

    @torch.no_grad()
    def update(self, cost_ret: torch.Tensor):
        """
        Input: cost_ret [B] or [T,B], mean taken. Output: psi, lambda_, c.
        g = E[cost] - d; cond = λ + c*g; λ <- clip(cond, 0, +inf);
        psi = λ*g + 0.5*c*g^2 if cond>0 else -0.5*λ^2/c;
        c <- clip(c*(1e-5+1), c, 1.0)  # non-decreasing
        """
        g = cost_ret.mean()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
            g /= dist.get_world_size()
        g = g - self.cost_limit

        lambda_ = self.lagrange_multiplier.detach()
        c = self.penalty_multiplier.detach()

        cond = lambda_ + c * g
        new_lambda = torch.clamp(cond, min=0.0)

        # Piecewise psi (match JAX impl)
        psi_pos  = lambda_ * g + 0.5 * c * g * g
        psi_neg  = -0.5 * (lambda_ * lambda_) / (c + 1e-12)
        psi = torch.where(cond > 0.0, psi_pos, psi_neg)

        self.lagrange_multiplier.copy_(new_lambda)
        new_c = torch.clamp(c * (1e-5 + 1.0), min=c, max=1.0)
        self.penalty_multiplier.copy_(new_c)

        return psi.detach(), self.lagrange_multiplier.detach(), self.penalty_multiplier.detach()

    @property
    def lambda_(self):
        return float(self.lagrange_multiplier.item())

    @property
    def c(self):
        return float(self.penalty_multiplier.item())

    @torch.no_grad()
    def effective_penalty(self, g: torch.Tensor):
        """Effective penalty for actor: λ_eff = ReLU(λ + c*g). g is constraint residual."""
        val = self.lagrange_multiplier + self.penalty_multiplier * g
        return torch.clamp(val, min=0.0)


@dataclass
class PIDConfig:
    kp: float = 0.0
    ki: float = 0.1
    kd: float = 0.0
    init_penalty: float = 0.0
    d_delay: int = 10
    delta_p_ema_alpha: float = 0.95
    delta_d_ema_alpha: float = 0.95
    sum_norm: bool = True
    diff_norm: bool = False
    penalty_max: float = 100.0
    lagrangian_multiplier_init: float = 0.001
    use_cost_decay: bool = False
    init_cost_limit: float = 10.0
    decay_time_step: int = 20_000
    decay_num: int = 7
    decay_limit_step: float = 2.0

@dataclass
class Config:
    cost_limit: float = 0.1
    pid: PIDConfig = PIDConfig()


class PIDLagrangian():  # noqa: B024
    """PID version of Lagrangian.

    Similar to the :class:`Lagrange` module, this module implements the PID version of the
    lagrangian method.

    .. note::
        The PID-Lagrange is more general than the Lagrange, and can be used in any policy gradient
        algorithm. As PID_Lagrange use the PID controller to control the lagrangian multiplier, it
        is more stable than the naive Lagrange.

    Args:
        pid_kp (float): The proportional gain of the PID controller.
        pid_ki (float): The integral gain of the PID controller.
        pid_kd (float): The derivative gain of the PID controller.
        pid_d_delay (int): The delay of the derivative term.
        pid_delta_p_ema_alpha (float): The exponential moving average alpha of the delta_p.
        pid_delta_d_ema_alpha (float): The exponential moving average alpha of the delta_d.
        sum_norm (bool): Whether to use the sum norm.
        diff_norm (bool): Whether to use the diff norm.
        penalty_max (int): The maximum penalty.
        lagrangian_multiplier_init (float): The initial value of the lagrangian multiplier.
        cost_limit (float): The cost limit.

    References:
        - Title: Responsive Safety in Reinforcement Learning by PID Lagrangian Methods
        - Authors: Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel.
        - URL: `PID Lagrange <https://arxiv.org/abs/2007.03964>`_
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
            self,
            config,
    ) -> None:
        """Initialize an instance of :class:`PIDLagrangian`."""
        self._pid_kp: float = config.pid.kp
        self._pid_ki: float = config.pid.ki
        self._pid_kd: float = config.pid.kd
        self._pid_d_delay = config.pid.d_delay
        self._pid_delta_p_ema_alpha: float = config.pid.delta_p_ema_alpha
        self._pid_delta_d_ema_alpha: float = config.pid.delta_d_ema_alpha
        self._penalty_max: int = config.pid.penalty_max
        self._sum_norm: bool = config.pid.sum_norm
        self._diff_norm: bool = config.pid.diff_norm
        self._pid_i: float = config.pid.lagrangian_multiplier_init
        self._cost_ds: deque[float] = deque(maxlen=self._pid_d_delay)
        self._cost_ds.append(0.0)
        self._delta_p: float = 0.0
        self._cost_d: float = 0.0
        self._pid_d: float = 0.0
        self._cost_limit: float = config.cost_limit
        self._cost_penalty: float = config.pid.init_penalty
        self._use_cost_decay: bool = config.pid.use_cost_decay
        self._current_cost_limit: float = config.pid.init_cost_limit
        if self._use_cost_decay:
            self._steps = [config.pid.decay_time_step * (i + 1) for i in range(config.pid.decay_num)]
            self._limits = [max(config.pid.init_cost_limit - i * config.pid.decay_limit_step, config.cost_limit) for i
                            in range(config.pid.decay_num)]

    @property
    def lagrange_penalty(self) -> float:
        """The lagrangian multiplier."""
        return self._cost_penalty

    @property
    def delta_p(self) -> float:
        """The lagrangian multiplier p."""
        return self._delta_p

    @property
    def pid_i(self) -> float:
        """The lagrangian multiplier i."""
        return self._pid_i

    @property
    def pid_d(self) -> float:
        """The lagrangian multiplier d."""
        return self._pid_d

    def pid_update(self, epcost, step) -> None:
        r"""Update the PID controller.

        PID controller update the lagrangian multiplier following the next equation:

        .. math::

            \lambda_{t+1} = \lambda_t + (K_p e_p + K_i \int e_p dt + K_d \frac{d e_p}{d t}) \eta

        where :math:`e_p` is the error between the current episode cost and the cost limit,
        :math:`K_p`, :math:`K_i`, :math:`K_d` are the PID parameters, and :math:`\eta` is the
        learning rate.

        Args:
            ep_cost_avg (float): The average cost of the current episode.
        """
        ep_cost_avg = epcost
        if self._use_cost_decay:
            for i, threshold in enumerate(self._steps):
                if step < threshold:
                    self._current_cost_limit = self._limits[i]
                    break
            else:
                self._current_cost_limit = self._cost_limit
        else:
            self._current_cost_limit = self._cost_limit

        delta = float(ep_cost_avg - self._current_cost_limit)
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        if self._diff_norm:
            self._pid_i = max(0.0, min(1.0, self._pid_i))
        a_p = self._pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self._pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        self._pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * self._pid_d
        self._cost_penalty = max(0.0, pid_o)
        if self._diff_norm:
            self._cost_penalty = min(1.0, self._cost_penalty)
        if not (self._diff_norm or self._sum_norm):
            self._cost_penalty = min(self._cost_penalty, self._penalty_max)
        self._cost_ds.append(self._cost_d)
        self._cost_penalty = np.clip(self._cost_penalty, 0.0, self._penalty_max)
        return self._cost_penalty, self._pid_d, self._pid_i, self._delta_p


class DrQv2Vlm(OffPolicyAlgorithm):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: DrQv2Policy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(self,
                 policy: Union[str, type[DrQv2Policy]],
                 env: Union[GymEnv, str],
                 learning_rate: Union[float, Schedule] = 1e-3,
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
                 stats_window_size: int = 100,
                 tensorboard_log: Optional[str] = None,
                 policy_kwargs: Optional[dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = "auto",
                 _init_setup_model: bool = True,
                 use_vlm: str = 'pvp',
                 imitation_coef: float = 0.01,
                 stddev_schedule: str = 'linear(1.0,0.1,100000)',
                 stddev_clip: float = 0.3,
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
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.use_vlm = use_vlm
        self.imitation_coef = imitation_coef
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.aug = [RandomShiftsAug(pad=4), RandomRotateAug(max_deg=10)]
        
        # AWAC parameters (following official implementation)
        self.awac_beta = 2.0  # Temperature parameter for softmax weighting

        if _init_setup_model:
            self._setup_model()

        self.ep_stat_buffer = None
        self.start_num_timesteps = self.num_timesteps

        # Inspired by https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py
        self.alpha = 2.5
        self.task = ''  # pvp, imitation loss

        ################ lag ################
        if 'lag' in self.use_vlm:
            cfg = Config(cost_limit=10.)
            # self.pid_lagrange = PIDLagrangian(cfg)
            # self.cost_ema = CostEma(0.0)

            self.aug_lagrange = AugLagrange()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        if 'lag' in self.use_vlm:
            self.critic_cost_batch_norm_stats = get_parameters_by_name(self.critic_cost, ["running_"])
            self.critic_cost_batch_norm_stats_target = get_parameters_by_name(self.critic_cost_target, ["running_"])

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        if 'lag' in self.use_vlm:
            self.critic_cost = self.policy.critic_cost
            self.critic_cost_target = self.policy.critic_cost_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        if 'lag' in self.use_vlm:
            self._update_learning_rate([self.critic_cost.optimizer])

        if 'pvp' in self.use_vlm or 'awac' in self.use_vlm:
            imitation_losses = []
        actor_losses, critic_losses, critic_cost_losses = [], [], []
        # train for gradient_steps epochs
        for gradient_step in range(self.gradient_steps):
            self._n_updates += 1

            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            # discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            # reward shaping
            if 'r_clip' in self.use_vlm:
                bonus = replay_data.clip_safety_scores  # (N, n_envs)
                replay_data.rewards.add_(bonus)
            if 'r_sim' in self.use_vlm:
                delta = 0.1
                mask = replay_data.has_vlm_actions > 0
                if mask.any():
                    sim = F.cosine_similarity(replay_data.actions[mask], replay_data.vlm_actions[mask],
                                                                    dim=-1, eps=1e-8).add(1).div(2).clamp(0, 1)
                    replay_data.rewards[mask] = replay_data.rewards[mask] + delta * sim
                    replay_data.rewards[~mask] = replay_data.rewards[~mask] + delta * sim.mean() / 2.

            with th.no_grad():
                aug = random.choice(self.aug)
                # aug
                if type(replay_data.observations) == dict:
                    replay_data.observations['bev_masks'] = aug(replay_data.observations['bev_masks'].float())
                    replay_data.next_observations['bev_masks'] = aug(replay_data.next_observations['bev_masks'].float())
                else:
                    replay_data = replay_data._replace(observations=aug(replay_data.observations.float()))
                    replay_data = replay_data._replace(next_observations=aug(replay_data.next_observations.float()))

            # Critic update
            with th.no_grad():
                stddev = schedule(self.stddev_schedule, self._n_updates)
                dist = self.actor.forward_dist(replay_data.next_observations, stddev)
                next_action = dist.sample(clip=self.stddev_clip)
                target_Q = th.cat(self.critic_target(replay_data.next_observations, next_action), dim=1)
                target_Q, _ = th.min(target_Q, dim=1, keepdim=True)
                if 'pvp' in self.use_vlm and 'org' in self.use_vlm:
                    target_Q = (1 - replay_data.dones) * (self.gamma * target_Q)
                else:
                    target_Q = replay_data.rewards + (1 - replay_data.dones) * (self.gamma * target_Q)

            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            critic_loss = []
            self.q_value_bound = 1
            self.cql_coefficient = 1

            if 'pvp' in self.use_vlm:
                if 'org' in self.use_vlm:
                    current_q_values_vlm = self.critic(replay_data.observations, replay_data.vlm_actions)
                    for current_q_novice, current_q_vlm in zip(current_q_values, current_q_values_vlm):
                        l_ = 0.5 * F.mse_loss(current_q_novice, target_Q)
                        pvp_loss = 0.
                        pvp_loss += th.mean(
                            replay_data.has_vlm_actions *
                            (F.mse_loss(current_q_vlm, th.ones_like(current_q_vlm)))
                        )
                        pvp_loss += th.mean(
                            replay_data.has_vlm_actions *
                            (F.mse_loss(current_q_novice, -th.ones_like(current_q_novice)))
                        )
                        l = l_ + pvp_loss
                        critic_loss.append(l)
                else:
                    lambda_pvp = delayed_lin_anneal(start=1.0, end=0.0, t=self.num_timesteps, t0=50_000,
                                                    T_total=150_000)
                    current_q_values_vlm = self.critic(replay_data.observations, replay_data.vlm_actions)
                    for current_q_novice, current_q_vlm in zip(current_q_values, current_q_values_vlm):
                        l_ = 0.5 * F.mse_loss(current_q_novice, target_Q)
                        # delta = current_q_vlm - current_q_novice.detach()
                        # pvp_loss = (replay_data.has_vlm_actions * F.softplus(margin - delta)).mean()
                        pvp_loss = (replay_data.has_vlm_actions * F.mse_loss(current_q_vlm, current_q_novice.detach() + 2)).mean()
                        l = l_ + lambda_pvp * pvp_loss
                        critic_loss.append(l)
            else:
                for current_q_novice in current_q_values:
                    l = 0.5 * F.mse_loss(current_q_novice, target_Q)
                    critic_loss.append(l)

            critic_loss = sum(critic_loss)

            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            # Clip gradients to prevent explosion
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic.optimizer.step()

            if 'lag' in self.use_vlm:
                with th.no_grad():
                    stddev = schedule(self.stddev_schedule, self._n_updates)
                    dist = self.actor.forward_dist(replay_data.next_observations, stddev)
                    next_action = dist.sample(clip=self.stddev_clip)
                    target_Q = th.cat(self.critic_cost_target(replay_data.next_observations, next_action), dim=1)
                    target_Q, _ = th.min(target_Q, dim=1, keepdim=True)
                    target_Q = replay_data.costs + (1 - replay_data.dones) * (self.gamma * target_Q)
                current_q_values = self.critic_cost(replay_data.observations, replay_data.actions)
                critic_cost_loss = []
                for current_q_novice in current_q_values:
                    l = 0.5 * F.mse_loss(current_q_novice, target_Q)
                    critic_cost_loss.append(l)
                critic_cost_loss = sum(critic_cost_loss)
                self.critic_cost.optimizer.zero_grad()
                critic_cost_loss.backward()
                # Clip gradients to prevent explosion
                th.nn.utils.clip_grad_norm_(self.critic_cost.parameters(), max_norm=1.0)
                self.critic_cost.optimizer.step()
                critic_cost_losses.append(critic_cost_loss.item())

            # Policy update
            # In original implement of DrQv2, actor and critic share the same feature extractor,
            # the feature extractor (FE) will only be updated by critic loss,
            # before actor update, the FEATURE should be detached before feeding into actor to prevent gradient error.
            # the natural implement of sb3 actor's inputs are raw images, we do not wanna to break it.
            # so for simplicity, we just disable the share_feature_extractor
            stddev = schedule(self.stddev_schedule, self._n_updates)
            dist = self.actor.forward_dist(replay_data.observations, stddev)
            action = dist.sample(clip=self.stddev_clip)
            q_values = th.cat(self.critic(replay_data.observations, action), dim=1)
            Q, _ = th.min(q_values, dim=1, keepdim=True)
            if 'awac' in self.use_vlm:
                Q1 = self.critic.q1_forward(replay_data.observations, action)
                lmbda = self.alpha / Q1.abs().mean().detach()
                base_actor_loss = - lmbda * Q.mean()
                actor_losses.append(base_actor_loss.item())
            elif 'lag' in self.use_vlm:
                Q1 = self.critic.q1_forward(replay_data.observations, action)
                Q1_cost = self.critic_cost.q1_forward(replay_data.observations, action)
                # base_actor_loss = (-Q1 + self.pid_lagrange.lagrange_penalty * Q1_cost).mean()
                base_actor_loss = (-Q1 + self.aug_lagrange.lambda_ * Q1_cost).mean()  # self.aug_lagrange.lambda_
                actor_losses.append(base_actor_loss.item())
            else:
                base_actor_loss = - Q.mean()
                actor_losses.append(base_actor_loss.item())

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
                    vlm_actions = replay_data.vlm_actions[mask].clamp(-1+eps, 1-eps)  # Expert actions

                    # Z = self.actor.extract_features(s_b, self.actor.features_extractor).detach()
                    # X = atanh(self.actor.forward(s_b))
                    # Y = atanh(vlm_actions)
                    # awac_loss = mmd2_conditional(Z, X, Y)

                    # X = atanh(self.actor.forward(s_b))  # [B,A]
                    # Y = atanh(vlm_actions)  # [B,A]
                    # kxx = rbf(X, X).mean()
                    # kyy = rbf(Y, Y).mean()
                    # kxy = rbf(X, Y).mean()
                    # awac_loss = kxx + kyy - 2 * kxy

                    # pi = self.actor.forward(vlm_obs)
                    # with th.no_grad():
                    #     gap = (pi - vlm_actions).abs().mean(dim=-1)  # policy-expert gap
                    #     w_gap = (gap - 0.02).clamp(min=0.0)
                    # awac_loss = w_gap.detach() * F.mse_loss(pi, vlm_actions, reduction='none').mean(dim=-1)
                    # awac_loss = awac_loss.mean()

                    # with th.no_grad():
                    #     pi_act_det = self.actor.forward(s_b)
                    #
                    #     q1_pi, q2_pi = self.critic(s_b, pi_act_det)
                    #     v_pi = th.min(q1_pi, q2_pi)  # [B,1]
                    #
                    #     q1_exp, q2_exp = self.critic(s_b, vlm_actions)
                    #     q_exp = th.min(q1_exp, q2_exp)  # [B,1]
                    #
                    #     adv = (q_exp - v_pi).squeeze(-1)
                    #     gate = adv > 0.0  # Q-filter, learn only positive advantage
                    #     weights = th.exp(adv / 2.0)  # beta~2, per-sample
                    #     weights = th.clamp(weights, max=20.0)

                    with th.no_grad():
                        pi_act_det = self.actor.forward(s_b)

                        q1_pi, q2_pi = self.critic(s_b, pi_act_det)
                        q_pi = th.min(q1_pi, q2_pi)  # [B,1]

                        q1_exp, q2_exp = self.critic(s_b, vlm_actions)
                        q_exp = th.min(q1_exp, q2_exp)  # [B,1]

                        adv = (q_exp - q_pi).squeeze(-1)
                        gate = adv > 0.0  # Q-filter
                        weights = th.exp(adv / 2.0)
                        weights = th.clamp(weights, max=20.0)

                    stddev = schedule(self.stddev_schedule, self._n_updates)
                    pi_dist = self.actor.forward_dist(s_b, std=stddev)
                    logp_exp = pi_dist.log_prob(vlm_actions).sum(-1)
                    awac_loss = - (weights.detach() * logp_exp)[gate].mean() if gate.any() else th.zeros_like(base_actor_loss)

            # Combine actor loss
            il = il_coef_schedule(self.num_timesteps)
            # il = awac_coef(self.num_timesteps)
            actor_loss = base_actor_loss + il * awac_loss  #il *
            if 'awac' in self.use_vlm:
                imitation_losses.append(awac_loss.item())
            if 'pvp' in self.use_vlm:
                imitation_losses.append(pvp_loss.item())

            # actor_loss = base_actor_loss
            # if self.use_vlm:
            #     imitation_losses.append(pvp_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            # Clip gradients to prevent explosion
            th.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor.optimizer.step()

            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
            if 'lag' in self.use_vlm:
                polyak_update(self.critic_cost.parameters(), self.critic_cost_target.parameters(), self.tau)
                polyak_update(self.critic_cost_batch_norm_stats, self.critic_cost_batch_norm_stats_target, 1.0)

        # Logs
        self.train_debug = {
            "train/actor_loss": np.mean(actor_losses),
            "train/critic_loss": np.mean(critic_losses),
        }
        if 'pvp' in self.use_vlm or 'awac' in self.use_vlm:
            self.train_debug.update({"train/vlm_actions_percentage":
                                         (self.replay_buffer.has_vlm_actions[:self.replay_buffer.pos if not
                                         self.replay_buffer.full else self.buffer_size] == 1).mean()})
        if 'r_clip' in self.use_vlm:
            self.train_debug.update({"train/clip_socres_percentage":
                                         (self.replay_buffer.clip_safety_scores[:self.replay_buffer.pos if not
                                         self.replay_buffer.full else self.buffer_size]>-1).mean()})
            self.train_debug.update({"train/clip_socres_average":
                                         (self.replay_buffer.clip_safety_scores[:self.replay_buffer.pos if not
                                         self.replay_buffer.full else self.buffer_size]).mean()})
            self.train_debug.update({"train/clip_raw_bonus_average":
                                         (self.replay_buffer.clip_raw_bonus[:self.replay_buffer.pos if not
                                         self.replay_buffer.full else self.buffer_size]).mean()})
            self.train_debug.update({"train/clip_rms_mean":
                                         (self.replay_buffer.clip_rms_mean[self.replay_buffer.pos])})
            self.train_debug.update({"train/clip_rms_std":
                                         (self.replay_buffer.clip_rms_std[self.replay_buffer.pos])})
        if 'pvp' in self.use_vlm or 'awac' in self.use_vlm:
            self.train_debug.update({"train/imitation_loss": np.mean(imitation_losses)})
        if 'lag' in self.use_vlm:
            self.train_debug.update({"train/critic_cost_loss": np.mean(critic_cost_losses)})
            self.train_debug.update({"train/aug_lagrange_multiplier": np.mean(self.aug_lagrange.lambda_)})
            # self.train_debug.update({"train/lagrange_penalty": self.pid_lagrange.lagrange_penalty})
            # self.train_debug.update({"train/delta_p": self.pid_lagrange.delta_p})
            # self.train_debug.update({"train/pid_i": self.pid_lagrange.pid_i})
            # self.train_debug.update({"train/pid_d": self.pid_lagrange.pid_d})
            # self.train_debug.update({"train/cost_ema": self.cost_ema.value})

    def learn(self,
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

            if self.num_timesteps % 1000 == 0:
                gc.collect()
                th.cuda.empty_cache()

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
                if 'lag' in self.use_vlm:
                    # self.cost_ema.value = self.cost_ema.value * 0.99 + infos[i]['episode_stat']['cost'] * 0.01
                    if self.num_timesteps > 5000:
                        # self.pid_lagrange.pid_update(self.cost_ema.value, self.num_timesteps)
                        self.aug_lagrange.update(infos[i]['episode_stat']['cost'])


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