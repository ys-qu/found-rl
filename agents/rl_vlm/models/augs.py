import torch as th
from torch.nn import functional as F
import math

class RandomShiftsAug(th.nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'constant', 0)
        eps = 1.0 / (h + 2 * self.pad)
        arange = th.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = th.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = th.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             mode='nearest',
                             padding_mode='zeros',
                             align_corners=False)


class RandomRotateAug(th.nn.Module):
    def __init__(self, max_deg: float = 10.0):
        """Random rotation in [-max_deg, +max_deg] degrees."""
        super().__init__()
        self.max_rad = math.radians(max_deg)

    def forward(self, x: th.Tensor) -> th.Tensor:
        n, c, h, w = x.size()
        assert h == w, "Only square BEV masks supported"

        # Random angle (radians)
        angles = (th.rand(n, device=x.device) * 2 - 1) * self.max_rad  # [-max_rad, max_rad]

        # Rotation matrix (batch, 2, 3)
        cos = th.cos(angles)
        sin = th.sin(angles)
        zeros = th.zeros_like(cos)
        ones = th.ones_like(cos)
        rot_mats = th.stack([
            cos, -sin, zeros,
            sin,  cos, zeros
        ], dim=1).view(n, 2, 3)

        # affine_grid generates sampling grid
        grid = F.affine_grid(rot_mats, x.size(), align_corners=False)

        # Use nearest to keep mask clean
        return F.grid_sample(x, grid, mode='nearest', padding_mode='zeros', align_corners=False)