"""Policies: abstract base class and concrete implementations."""

import torch as th
import torch.nn as nn
import numpy as np
import math

from . import torch_util as tu
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)

class DSConv(nn.Sequential):
    """
    Depthwise(3x3, stride, pad=1) + Pointwise(1x1) + ReLU
    """
    def __init__(self, in_c, out_c, stride=1):
        super().__init__(
            nn.Conv2d(in_c, in_c, kernel_size=3, stride=stride, padding=1,
                      groups=in_c, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

class InvertedBlock(nn.Module):
    """
    MobileNetV2-style Inverted Residual (no BN). PW->DW->PW.
    Residual when stride=1 and ch_in==ch_out.
    """
    def __init__(self, ch_in, ch_out, expand_ratio=2.0, stride=1):
        super().__init__()
        assert stride in [1, 2]
        hidden = int(round(ch_in * expand_ratio))
        self.use_res = (stride == 1 and ch_in == ch_out)

        layers = []
        # 1) Expand
        if expand_ratio != 1.0:
            layers += [
                nn.Conv2d(ch_in, hidden, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
            ]
        else:
            hidden = ch_in
        # 2) Depthwise (spatial conv, optional downsample)
        layers += [
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1,
                      groups=hidden, bias=True),
            nn.ReLU(inplace=True),
        ]
        # 3) Project to ch_out
        layers += [
            nn.Conv2d(hidden, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class MobileXtMaCNN(BaseFeaturesExtractor):
    """BEV(C,H,W) + state -> features_dim. Stem -> InvertedBlock x4 -> bottleneck -> Flatten."""
    def __init__(self, observation_space, features_dim=256, states_neurons=[256, 256]):
        super().__init__(observation_space, features_dim)

        # Parse obs space
        if hasattr(observation_space, 'spaces'):
            bev_shape = observation_space.spaces['bev_masks'].shape  # (C,H,W)
            state_dim = observation_space.spaces['state'].shape[0]
        else:
            bev_shape = observation_space['bev_masks'].shape
            state_dim = observation_space['state'].shape[0]

        n_in, H, W = bev_shape

        C1, C2, C3, C4 = 32, 64, 128, 256
        EXP = 2.0
        BOTTLENECK_C = 64

        # stem：92/96 -> 46/48
        stem = nn.Sequential(
            nn.Conv2d(n_in, C1, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # 4 inverted blocks, 3x s2 downsample -> H/16
        stages = nn.Sequential(
            InvertedBlock(C1, C1, expand_ratio=EXP, stride=1),  # H/2
            InvertedBlock(C1, C2, expand_ratio=EXP, stride=2),  # H/4
            InvertedBlock(C2, C3, expand_ratio=EXP, stride=2),  # H/8
            InvertedBlock(C3, C4, expand_ratio=EXP, stride=2),  # H/16
        )

        # 1x1 bottleneck -> Flatten
        tail = nn.Sequential(
            nn.Conv2d(C4, BOTTLENECK_C, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

        self.cnn = nn.Sequential(stem, stages, tail)

        # Dynamic flatten dim
        with th.no_grad():
            dummy = th.zeros(1, n_in, H, W)
            img_feat_dim = self.cnn(dummy).view(1, -1).shape[1]  # BOTTLENECK_C * H' * W'

        # State branch
        dims = [state_dim] + list(states_neurons)
        mlp = []
        for i in range(len(dims) - 1):
            mlp += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True)]
        self.state_linear = nn.Sequential(*mlp)

        # Fuse and project to features_dim
        self.linear = nn.Sequential(
            nn.Linear(img_feat_dim + dims[-1], 256), nn.ReLU(inplace=True),
            nn.Linear(256, self.features_dim), nn.ReLU(inplace=True),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # Kaiming/Xavier init
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # SB3 compat
    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        return self.forward(obs)

    def forward(self, obs, state=None):
        if isinstance(obs, dict):
            if 'bev_masks' not in obs or 'state' not in obs:
                raise ValueError(f"Expected dict with 'bev_masks' and 'state', got {list(obs.keys())}")
            bev = obs['bev_masks']; st = obs['state']
        else:
            bev = obs; st = state
            if st is None:
                raise ValueError("State data is required for IRLiteXtMaCNN")

        z_img = self.cnn(bev).flatten(1)
        z_st  = self.state_linear(st)
        x = th.cat([z_img, z_st], dim=1)
        return self.linear(x)


class GhostModule(nn.Module):
    """
    GhostConv: 1x1 for main features, DW 3x3 for ghost channels. No BN, bias=True.
    """
    def __init__(self, in_c, out_c, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        assert ratio >= 2
        assert stride in [1], "GhostModule does not downsample; use stride=1, downsample in Bottleneck."
        self.out_c = out_c

        init_channels = math.ceil(out_c / ratio)
        new_channels  = init_channels * (ratio - 1)
        pad_k = kernel_size // 2
        pad_dw = dw_size // 2

        # Main branch: 1x1 (or larger kernel) for main features
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_c, init_channels, kernel_size, stride, pad_k, bias=True),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        # Ghost branch: DW 3x3 synthesizes remaining channels
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, pad_dw,
                      groups=init_channels, bias=True),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = th.cat([x1, x2], dim=1)
        return out[:, :self.out_c, :, :]


class SELayerLite(nn.Module):
    """Optional SE (no BN), disabled by default."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.avg(x).view(b, c)
        s = self.fc(s).clamp_(0, 1).view(b, c, 1, 1)
        return x * s


class GhostBottleneckLite(nn.Module):
    """
    GhostBottleneck (no BN): GhostModule -> DW (stride 2) -> SE -> GhostModule.
    Shortcut: identity if stride=1 and channels match, else DW+PW.
    """
    def __init__(self, inp, hidden_dim, oup, kernel_size=3, stride=1, use_se=False, ratio=2):
        super().__init__()
        assert stride in [1, 2]
        pad = kernel_size // 2

        # Main path
        layers = []
        # 1) ghost expand
        layers.append(GhostModule(inp, hidden_dim, kernel_size=1, ratio=ratio, dw_size=3, stride=1, relu=True))
        # 2) depthwise (only when downsample)
        if stride == 2:
            layers += [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, pad,
                          groups=hidden_dim, bias=True),
                # No extra activation (linear dw)
            ]
        # 3) SE (optional)
        if use_se:
            layers.append(SELayerLite(hidden_dim))
        # 4) ghost project (linear, no ReLU)
        layers.append(GhostModule(hidden_dim, oup, kernel_size=1, ratio=ratio, dw_size=3, stride=1, relu=False))
        self.conv = nn.Sequential(*layers)

        # Shortcut
        if stride == 1 and inp == oup:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=True),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

# -----------------------------
# Feature Extractor (SB3 style)
# -----------------------------
class GhostLiteXtMaCNN(BaseFeaturesExtractor):
    '''
    Lite (GhostNet-style, no BN): stem -> GhostBottleneckLite x4 -> bottleneck -> Flatten, fuse with state.
    '''
    def __init__(self, observation_space, features_dim=256, states_neurons=[256, 256],
                 chans=(32, 64, 128, 256), expand_ratio=2.0, ghost_ratio=2):
        super().__init__(observation_space, features_dim)

        # Parse obs space
        if hasattr(observation_space, 'spaces'):
            n_in, H, W = observation_space.spaces['bev_masks'].shape
            state_dim = observation_space.spaces['state'].shape[0]
        else:
            n_in, H, W = observation_space['bev_masks'].shape
            state_dim = observation_space['state'].shape[0]

        C1, C2, C3, C4 = chans
        BOTTLENECK_C = 64

        # —— stem：3x3 s2（92/96 -> 46/48）
        stem = nn.Sequential(
            nn.Conv2d(n_in, C1, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # Ghost bottlenecks
        def hid(c):  # hidden_dim expansion
            return int(round(c * expand_ratio))

        stages = nn.Sequential(
            GhostBottleneckLite(C1, hid(C1), C1, kernel_size=3, stride=1, use_se=False, ratio=ghost_ratio),  # H/2
            GhostBottleneckLite(C1, hid(C1), C2, kernel_size=3, stride=2, use_se=False, ratio=ghost_ratio),  # H/4
            GhostBottleneckLite(C2, hid(C2), C3, kernel_size=3, stride=2, use_se=True, ratio=ghost_ratio),  # H/8
            GhostBottleneckLite(C3, hid(C3), C4, kernel_size=3, stride=2, use_se=True, ratio=ghost_ratio),  # H/16
        )

        tail = nn.Sequential(
            nn.Conv2d(C4, BOTTLENECK_C, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

        self.cnn = nn.Sequential(stem, stages, tail)

        # Dynamic flatten dim
        with th.no_grad():
            img_feat_dim = self.cnn(th.zeros(1, n_in, H, W)).view(1, -1).shape[1]

        # State branch
        dims = [state_dim] + list(states_neurons)
        mlp = []
        for i in range(len(dims) - 1):
            mlp += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True)]
        self.state_linear = nn.Sequential(*mlp)

        # Fuse and project to features_dim
        self.linear = nn.Sequential(
            nn.Linear(img_feat_dim + dims[-1], 256), nn.ReLU(inplace=True),
            nn.Linear(256, self.features_dim), nn.ReLU(inplace=True),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    # SB3 compat
    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        return self.forward(obs)

    def forward(self, obs, state=None):
        if isinstance(obs, dict):
            if 'bev_masks' not in obs or 'state' not in obs:
                raise ValueError(f"Expected dict with 'bev_masks' and 'state', got {list(obs.keys())}")
            bev = obs['bev_masks']; st = obs['state']
        else:
            bev = obs; st = state
            if st is None:
                raise ValueError("State data is required for GhostLiteXtMaCNN")

        z_img = self.cnn(bev).flatten(1)
        z_st  = self.state_linear(st)
        x = th.cat([z_img, z_st], dim=1)
        return self.linear(x)


class LiteXtMaCNN(BaseFeaturesExtractor):
    '''
    Inspired by https://github.com/xtma/pytorch_car_caring
    Lite: stride=2 downsample + 1x1 bottleneck -> Flatten, fuse state -> features_dim
    '''
    def __init__(self, observation_space, features_dim=256, states_neurons=[256, 256]):
        super().__init__(observation_space, features_dim)

        # Parse obs space
        if hasattr(observation_space, 'spaces'):
            bev_shape  = observation_space.spaces['bev_masks'].shape  # (C,H,W)
            state_dim  = observation_space.spaces['state'].shape[0]
        else:
            bev_shape  = observation_space['bev_masks'].shape
            state_dim  = observation_space['state'].shape[0]

        n_input_channels, H, W = bev_shape
        C1, C2, C3, C4 = 32, 64, 128, 256
        BOTTLENECK_C = 64

        # 4x stride=2 downsample, then 1x1 channel reduction
        # , gn(C1)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, C1, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(C1, C2, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(C2, C3, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(C3, C4, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(C4, BOTTLENECK_C, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True),
        )

        # Dynamic CNN flatten dim
        with th.no_grad():
            dummy = th.zeros(1, n_input_channels, H, W)
            feat  = self.cnn(dummy)                   # [1, BOTTLENECK_C, H', W']
            img_feat_dim = feat.view(1, -1).shape[1]  # BOTTLENECK_C * H' * W'

        # State branch
        layers = []
        dims = [state_dim] + list(states_neurons)
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True)]
        self.state_linear = nn.Sequential(*layers)

        # Fuse and project to features_dim
        self.linear = nn.Sequential(
            nn.Linear(img_feat_dim + dims[-1], 256), nn.ReLU(inplace=True),
            nn.Linear(256, self.features_dim), nn.ReLU(inplace=True),
        )

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    # SB3 compat
    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        return self.forward(obs)

    def forward(self, obs, state=None):
        # SB3(dict) or direct call(tensor, state)
        if isinstance(obs, dict):
            if 'bev_masks' not in obs or 'state' not in obs:
                raise ValueError(f"Expected dict with 'bev_masks' and 'state', got {list(obs.keys())}")
            birdview = obs['bev_masks']
            state_data = obs['state']
        else:
            birdview = obs
            state_data = state
            if state_data is None:
                raise ValueError("State data is required for XtMaCNN")

        z_img = self.cnn(birdview)          # [B, BOTTLENECK_C, H', W']
        z_img = th.flatten(z_img, 1)        # [B, BOTTLENECK_C*H'*W']
        z_state = self.state_linear(state_data)
        x = th.cat([z_img, z_state], dim=1)
        x = self.linear(x)
        return x


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def load_matching_weights_(target: nn.Module, source: nn.Module, strict: bool = False, prefix_filter=None):
    """Copy params by matching keys+shapes from source.state_dict() into target."""
    with th.no_grad():
        src_sd = source.state_dict()
        tgt_sd = target.state_dict()
        copied = 0
        for k, v in src_sd.items():
            if (k in tgt_sd) and (tgt_sd[k].shape == v.shape):
                if prefix_filter is None or any(k.startswith(p) for p in prefix_filter):
                    tgt_sd[k].copy_(v)    # in-place copy
                    copied += 1
        # strict=False: allow unmatched keys to keep original values
        target.load_state_dict(tgt_sd, strict=strict)
    return copied


class LiteXtMaCNNCURL(BaseFeaturesExtractor):
    '''
    Inspired by https://github.com/xtma/pytorch_car_caring
    Lite: stride=2 downsample + 1x1 bottleneck -> Flatten, fuse state -> features_dim
    '''
    def __init__(self, observation_space, features_dim=256, states_neurons=[256, 256], output_logits=False):
        super().__init__(observation_space, features_dim)

        # Parse obs space
        if hasattr(observation_space, 'spaces'):
            bev_shape  = observation_space.spaces['bev_masks'].shape  # (C,H,W)
            state_dim  = observation_space.spaces['state'].shape[0]
        else:
            bev_shape  = observation_space['bev_masks'].shape
            state_dim  = observation_space['state'].shape[0]

        n_input_channels, H, W = bev_shape
        C1, C2, C3, C4 = 32, 64, 128, 256
        BOTTLENECK_C = 64

        # 4x stride=2 downsample, then 1x1 channel reduction
        # , gn(C1)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, C1, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(C1, C2, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(C2, C3, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(C3, C4, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(C4, BOTTLENECK_C, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True),
        )

        # Dynamic CNN flatten dim
        with th.no_grad():
            dummy = th.zeros(1, n_input_channels, H, W)
            feat  = self.cnn(dummy)                   # [1, BOTTLENECK_C, H', W']
            img_feat_dim = feat.view(1, -1).shape[1]  # BOTTLENECK_C * H' * W'

        # State branch
        layers = []
        dims = [state_dim] + list(states_neurons)
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True)]
        self.state_linear = nn.Sequential(*layers)

        # Fuse and project to features_dim
        self.linear = nn.Sequential(
            nn.Linear(img_feat_dim + dims[-1], 256), nn.ReLU(inplace=True),
            nn.Linear(256, self.features_dim), nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(self.features_dim, self.features_dim)
        self.ln = nn.LayerNorm(self.features_dim)
        self.output_logits = output_logits
        self.outputs = dict()

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    # SB3 compat
    def extract_features(self, obs: th.Tensor, detach=False) -> th.Tensor:
        return self.forward(obs, detach=detach)

    def forward(self, obs, state=None, detach=False):
        # SB3(dict) or direct call(tensor, state)
        if isinstance(obs, dict):
            if 'bev_masks' not in obs or 'state' not in obs:
                raise ValueError(f"Expected dict with 'bev_masks' and 'state', got {list(obs.keys())}")
            birdview = obs['bev_masks']
            state_data = obs['state']
        else:
            birdview = obs
            state_data = state
            if state_data is None:
                raise ValueError("State data is required for XtMaCNN")

        z_img = self.cnn(birdview)          # [B, BOTTLENECK_C, H', W']
        z_img = th.flatten(z_img, 1)        # [B, BOTTLENECK_C*H'*W']
        z_state = self.state_linear(state_data)
        x = th.cat([z_img, z_state], dim=1)
        x = self.linear(x)

        if detach:
            x = x.detach()

        h_fc = self.fc(x)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = th.tanh(h_norm)
            self.outputs['tanh'] = out
        return out

    def copy_conv_ln_weights_from(self, source):
        # the original paper used tie wieght to make the actir and critic share the same encoder.
        # But since we also have Relu, we use load_matching_weights_ to share the same weights
        load_matching_weights_(self, source.cnn, strict=False)
        load_matching_weights_(self, source.state_linear, strict=False)
        load_matching_weights_(self, source.linear, strict=False)
        # for i in range(len(self.cnn)):
        #     tie_weights(src=source.cnn[i], trg=self.cnn[i])
        # for i in range(len(self.state_linear)):
        #     tie_weights(src=source.state_linear[i], trg=self.state_linear[i])
        # tie_weights(src=source.linear, trg=self.linear)

class FineXtMaCNN(BaseFeaturesExtractor):
    '''
    Tiny: first layer stride=2, rest 3x3 stride=1; split projection heads to avoid state being overwhelmed.
    '''
    def __init__(self, observation_space, features_dim=256, states_neurons=[256, 256]):
        super().__init__(observation_space, features_dim)

        # Parse obs space
        if hasattr(observation_space, 'spaces'):
            bev_shape  = observation_space.spaces['bev_masks'].shape  # (C,H,W)
            state_dim  = observation_space.spaces['state'].shape[0]
        else:
            bev_shape  = observation_space['bev_masks'].shape
            state_dim  = observation_space['state'].shape[0]

        n_input_channels, H, W = bev_shape

        # DrQ-v2 style: first s=2, rest s=1, no padding
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 3, stride=2), nn.ReLU(inplace=True),  # H/2-0.5
            nn.Conv2d(32, 32, 3, stride=1),              nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),              nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),              nn.ReLU(inplace=True),
        )

        # Dynamic CNN flatten dim
        with th.no_grad():
            dummy = th.zeros(1, n_input_channels, H, W)
            feat  = self.cnn(dummy)
            img_feat_dim = feat.view(1, -1).shape[1]      # 32 * H' * W'

        # State branch
        layers = []
        dims = [state_dim] + list(states_neurons)
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True)]
        self.state_linear = nn.Sequential(*layers)

        # Project img and state to same small dim to balance
        PROJ_DIM = 128
        self.img_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_feat_dim, PROJ_DIM),
            nn.LayerNorm(PROJ_DIM),
            nn.ReLU(inplace=True),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(dims[-1], PROJ_DIM),
            nn.LayerNorm(PROJ_DIM),
            nn.ReLU(inplace=True),
        )

        # Fuse and project to features_dim
        self.linear = nn.Sequential(
            nn.Linear(PROJ_DIM * 2, 256), nn.ReLU(inplace=True),
            nn.Linear(256, self.features_dim), nn.ReLU(inplace=True),
        )

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    # SB3 compat
    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        return self.forward(obs)

    def forward(self, obs, state=None):
        # SB3(dict) or direct call(tensor, state)
        if isinstance(obs, dict):
            if 'bev_masks' not in obs or 'state' not in obs:
                raise ValueError(f"Expected dict with 'bev_masks' and 'state', got {list(obs.keys())}")
            birdview = obs['bev_masks']
            state_data = obs['state']
        else:
            birdview = obs
            state_data = state
            if state_data is None:
                raise ValueError("State data is required for XtMaCNN")

        z_img   = self.cnn(birdview)            # [B, 32, H', W']
        z_img   = self.img_proj(z_img)          # [B, PROJ_DIM]
        z_state = self.state_linear(state_data) # [B, ...]
        z_state = self.state_proj(z_state)      # [B, PROJ_DIM]

        x = th.cat([z_img, z_state], dim=1)     # [B, 2*PROJ_DIM]
        x = self.linear(x)                      # [B, features_dim]
        return x


class XtMaCNN(BaseFeaturesExtractor):
    '''
    Inspired by https://github.com/xtma/pytorch_car_caring
    '''

    def __init__(self, observation_space, features_dim=256, states_neurons=[256, 256]):
        super().__init__(observation_space, features_dim)
        # features_dim is already set by super().__init__

        # Extract dimensions from observation space
        if hasattr(observation_space, 'spaces'):
            # Dict observation space
            n_input_channels = observation_space.spaces['bev_masks'].shape[0]
            state_dim = observation_space.spaces['state'].shape[0]
        else:
            # Fallback for older style
            n_input_channels = observation_space['bev_masks'].shape[0]
            state_dim = observation_space['state'].shape[0]

        # Fixed architecture for (15, 96, 96) input
        # Less aggressive downsampling to preserve spatial information
        self.cnn = nn.Sequential(
            # 96x96 -> 96x96 (preserve spatial info)
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            
            # 96x96 -> 48x48
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 48x48 -> 24x24
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # 24x24 -> 12x12
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # 12x12 -> 6x6
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with th.no_grad():
            # Handle Dict observation space properly
            if hasattr(observation_space, 'spaces'):
                sample_obs = observation_space.sample()
                birdview_sample = sample_obs['bev_masks']
            else:
                birdview_sample = observation_space['bev_masks'].sample()
            
            n_flatten = self.cnn(th.as_tensor(birdview_sample[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + states_neurons[-1], 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.features_dim),  # Use self.features_dim from parent
            nn.ReLU()
        )

        states_neurons = [state_dim] + states_neurons
        self.state_linear = []
        for i in range(len(states_neurons)-1):
            self.state_linear.append(nn.Linear(states_neurons[i], states_neurons[i+1]))
            self.state_linear.append(nn.ReLU())
            if i < len(states_neurons)-2:  # Add dropout to intermediate layers
                self.state_linear.append(nn.Dropout(0.1))
        self.state_linear = nn.Sequential(*self.state_linear)

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """Extract features from observations for SB3 compatibility."""
        return self.forward(obs)

    def forward(self, obs, state=None):
        # Handle both SB3 feature extractor calls and direct calls
        if isinstance(obs, dict):
            # SB3 feature extractor call: obs is a dict with 'bev_masks' and 'state'
            if 'bev_masks' not in obs or 'state' not in obs:
                raise ValueError(f"Expected dict with 'bev_masks' and 'state' keys, got {list(obs.keys())}")
            birdview = obs['bev_masks']
            state_data = obs['state']
        else:
            # Direct call: obs is birdview, state is separate
            birdview = obs
            state_data = state
            if state_data is None:
                raise ValueError("State data is required for XtMaCNN")
            
        x = self.cnn(birdview)
        latent_state = self.state_linear(state_data)

        x = th.cat((x, latent_state), dim=1)
        x = self.linear(x)
        return x


class ImpalaCNN(nn.Module):
    def __init__(self, observation_space, chans=(16, 32, 32, 64, 64), states_neurons=[256],
                 features_dim=256, nblock=2, batch_norm=False, final_relu=True):
        # (16, 32, 32)
        super().__init__()
        self.features_dim = features_dim
        self.final_relu = final_relu

        # image encoder
        curshape = observation_space['bev_masks'].shape
        s = 1 / np.sqrt(len(chans))  # per stack scale
        self.stacks = nn.ModuleList()
        for outchan in chans:
            stack = tu.CnnDownStack(curshape[0], nblock=nblock, outchan=outchan, scale=s, batch_norm=batch_norm)
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)

        # dense after concatenate
        n_image_latent = tu.intprod(curshape)
        self.dense = tu.NormedLinear(n_image_latent+states_neurons[-1], features_dim, scale=1.4)

        # state encoder
        states_neurons = [observation_space['state'].shape[0]] + states_neurons
        self.state_linear = []
        for i in range(len(states_neurons)-1):
            self.state_linear.append(tu.NormedLinear(states_neurons[i], states_neurons[i+1]))
            self.state_linear.append(nn.ReLU())
        self.state_linear = nn.Sequential(*self.state_linear)

    def forward(self, birdview, state):
        # birdview: [b, c, h, w]
        # x = x.to(dtype=th.float32) / self.scale_ob

        for layer in self.stacks:
            birdview = layer(birdview)

        x = th.flatten(birdview, 1)
        x = th.relu(x)

        latent_state = self.state_linear(state)

        x = th.cat((x, latent_state), dim=1)
        x = self.dense(x)
        if self.final_relu:
            x = th.relu(x)
        return x
