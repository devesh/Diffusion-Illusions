import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fast GPU SSIM / MS-SSIM & DTL regression loss
# ---------------------------------------------------------------------------

class FastSSIM(nn.Module):
    """Cached-window SSIM on GPU with zero per-call CPU->GPU allocations."""

    def __init__(self, window_size: int = 11, channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        w2d = (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)
        window = w2d.expand(channels, 1, window_size, window_size).contiguous()
        self.register_buffer("window", window)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        w = self.window.to(device=img1.device, dtype=img1.dtype)
        c = img1.shape[1]
        mu1 = F.conv2d(img1, w, padding=0, groups=c)
        mu2 = F.conv2d(img2, w, padding=0, groups=c)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, w, padding=0, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, w, padding=0, groups=c) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, w, padding=0, groups=c) - mu1_mu2
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return ssim_map.mean()


def dtl_loss(pred: torch.Tensor, target: torch.Tensor, ssim_fn: FastSSIM, ssim_weight: float = 0.25) -> torch.Tensor:
    """Dream Target Loss: L2 + SSIM regression in [0, 1] pixel space."""
    mse = F.mse_loss(pred, target)
    ssim_val = ssim_fn(pred, target)
    return mse + ssim_weight * (1.0 - ssim_val)


# ---------------------------------------------------------------------------
# Fast Prime Image Parameterizations
# ---------------------------------------------------------------------------

class FastFourierImage(nn.Module):
    """
    Faster drop-in replacement for LearnableImageFourier:
    - Uses precomputed Fourier features + Linear-style 1x1 conv MLP
    - GroupNorm/LayerNorm-free or InstanceNorm/no-BN option for clean AdamW optimization
    - Supports bfloat16/float32 seamlessly
    """

    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        num_channels: int = 3,
        hidden_dim: int = 256,
        num_features: int = 128,
        scale: float = 10.0,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.num_channels = num_channels

        y = torch.linspace(0, 1, height + 1)[:-1]
        x = torch.linspace(0, 1, width + 1)[:-1]
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        uv = torch.stack([gx, gy], dim=-1).reshape(-1, 2)  # [H*W, 2]
        freqs = torch.randn(2, num_features) * scale
        proj = 2.0 * math.pi * (uv @ freqs)
        feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [H*W, 2*F]
        feats = feats.t().reshape(1, 2 * num_features, height, width).contiguous()
        self.register_buffer("features", feats)

        m = 2 * num_features
        self.net = nn.Sequential(
            nn.Conv2d(m, hidden_dim, 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, num_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self) -> torch.Tensor:
        return self.net(self.features).squeeze(0)


class MultiResRasterImage(nn.Module):
    """
    Ultra-fast multi-scale Laplacian-like raster parameterization (coarser + finer grids).
    Converges much faster than an MLP under AdamW while preserving smooth structure.
    """

    def __init__(self, height: int = 512, width: int = 512, num_channels: int = 3):
        super().__init__()
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.coarse = nn.Parameter(torch.zeros(1, num_channels, height // 8, width // 8))
        self.mid = nn.Parameter(torch.zeros(1, num_channels, height // 2, width // 2))
        self.fine = nn.Parameter(torch.zeros(1, num_channels, height, width))

    def forward(self) -> torch.Tensor:
        c = F.interpolate(self.coarse, size=(self.height, self.width), mode="bilinear", align_corners=False)
        m = F.interpolate(self.mid, size=(self.height, self.width), mode="bilinear", align_corners=False)
        return torch.sigmoid(c + 0.5 * m + 0.25 * self.fine).squeeze(0)


# ---------------------------------------------------------------------------
# Differentiable Illusion Operators + Fast Prime Reconciliation
# ---------------------------------------------------------------------------

class FlipOperator:
    """1 prime image -> [upright, 180-deg rotated]."""

    num_primes = 1
    num_views = 2
    is_permutation = True
    equivariant_to_latent = True

    def derive(self, primes: list[torch.Tensor]) -> torch.Tensor:
        p = primes[0]  # [C, H, W]
        return torch.stack([p, p.rot90(2, dims=(1, 2))], dim=0)

    def invert_and_average(self, views: torch.Tensor) -> torch.Tensor:
        """views: [2, C, H, W] -> reconciled views [2, C, H, W]."""
        p = 0.5 * (views[0] + views[1].rot90(-2, dims=(1, 2)))
        return self.derive([p])

    def invert_velocities(self, v: torch.Tensor) -> torch.Tensor:
        """views velocity [2, C, H, W] -> averaged velocity [2, C, H, W]."""
        v0 = 0.5 * (v[0] + v[1].rot90(-2, dims=(1, 2)))
        return torch.stack([v0, v0.rot90(2, dims=(1, 2))], dim=0)


class TwistingSquaresOperator:
    """1 prime image -> [upright, checkerboard +/-90 tile-rotated]."""

    num_primes = 1
    num_views = 2
    is_permutation = True
    equivariant_to_latent = False

    def __init__(self, num_divisions: int = 4):
        self.num_divisions = num_divisions

    def _twist(self, img: torch.Tensor, direction: int = 1) -> torch.Tensor:
        c, h, w = img.shape
        n = self.num_divisions
        th, tw = h // n, w // n
        # Reshape into [c, n, th, n, tw] -> [n, n, c, th, tw]
        tiles = img[:, : n * th, : n * tw].view(c, n, th, n, tw).permute(1, 3, 0, 2, 4).contiguous()
        out = torch.empty_like(tiles)
        for y in range(n):
            for x in range(n):
                k = direction if (x + y) % 2 == 0 else -direction
                out[y, x] = tiles[y, x].rot90(k, dims=(1, 2))
        return out.permute(2, 0, 3, 1, 4).reshape(c, n * th, n * tw).contiguous()

    def derive(self, primes: list[torch.Tensor]) -> torch.Tensor:
        p = primes[0]
        return torch.stack([p, self._twist(p, direction=1)], dim=0)

    def invert_and_average(self, views: torch.Tensor) -> torch.Tensor:
        p = 0.5 * (views[0] + self._twist(views[1], direction=-1))
        return self.derive([p])

    def invert_velocities(self, v: torch.Tensor) -> torch.Tensor:
        v0 = 0.5 * (v[0] + self._twist(v[1], direction=-1))
        return torch.stack([v0, self._twist(v0, direction=1)], dim=0)


class RubiksCubeOperator:
    """
    6 solved cube faces (U, L, F, R, B, D) -> 12 views: [6 before (solved), 6 after (scrambled)].
    Default scramble ("R' L' F2 U' D' R' B2 R B2 L' B2 L2 U2 L' U2 L U2 L'", 18 moves) achieves the
    exact theoretical minimum dispersion across faces AND strictly non-zero orientation change for every square:
      - max squares from any single solved face on any scrambled face = 2 (row counts: {1,1,1,2,2,2})
      - total same-face sticker pairs shared between solved and scrambled states = 18 (vs 216 solved)
      - 0 of 48 edge/corner stickers stay on their solved face (only the 6 fixed centers stay: diag sum = 6)
      - 0 of 54 stickers keep their original "up" orientation (all 54 rotated by 90/180/270 deg: k in {1,2,3})
    """

    num_primes = 6
    num_views = 12
    is_permutation = True
    equivariant_to_latent = False

    FACE_NAMES = ["U", "L", "F", "R", "B", "D"]
    FACE_FRAMES = {
        "U": ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
        "L": ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
        "F": ((0, 0, 1), (1, 0, 0), (0, -1, 0)),
        "R": ((1, 0, 0), (0, 0, -1), (0, -1, 0)),
        "B": ((0, 0, -1), (-1, 0, 0), (0, -1, 0)),
        "D": ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
    }
    DEFAULT_SCRAMBLE = "R' L' F2 U' D' R' B2 R B2 L' B2 L2 U2 L' U2 L U2 L'"

    def __init__(self, scramble: str | None = None):
        self.scramble = scramble or self.DEFAULT_SCRAMBLE
        self.moves = self.scramble.split()
        self.mapping = self._simulate_scramble(self.moves)
        # Single-index vectorized gather indices across 4 rotation variants (4 * 54 = 216)
        fwd_idx = [0] * 54
        inv_idx = [0] * 54
        for dest_idx, (src_idx, k) in enumerate(self.mapping):
            fwd_idx[dest_idx] = k * 54 + src_idx
            inv_idx[src_idx] = ((4 - k) % 4) * 54 + dest_idx
        self._fwd_idx = fwd_idx
        self._inv_idx = inv_idx
        self._idx_cache: dict[tuple[torch.device, bool], torch.Tensor] = {}

    @classmethod
    def _dot(cls, a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @classmethod
    def _cross(cls, a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    @classmethod
    def _rot_cw(cls, v, ax):
        c = cls._cross(ax, v)
        d = cls._dot(ax, v)
        return (ax[0] * d - c[0], ax[1] * d - c[1], ax[2] * d - c[2])

    @classmethod
    def _simulate_scramble(cls, moves: list[str]) -> list[tuple[int, int]]:
        stickers = []
        for fi, fn in enumerate(cls.FACE_NAMES):
            n, u, v = cls.FACE_FRAMES[fn]
            for r in range(3):
                for c in range(3):
                    co, ro = c - 1, r - 1
                    pos = (n[0] + co * u[0] + ro * v[0], n[1] + co * u[1] + ro * v[1], n[2] + co * u[2] + ro * v[2])
                    stickers.append({"orig": fi * 9 + r * 3 + c, "pos": pos, "n": n, "u": u, "v": v})
        for mv in moves:
            ax = cls.FACE_FRAMES[mv[0]][0]
            times = 1 if len(mv) == 1 else (2 if mv[1] == "2" else 3)
            for s in stickers:
                if cls._dot(s["pos"], ax) == 1:
                    for _ in range(times):
                        s["pos"] = cls._rot_cw(s["pos"], ax)
                        s["n"] = cls._rot_cw(s["n"], ax)
                        s["u"] = cls._rot_cw(s["u"], ax)
                        s["v"] = cls._rot_cw(s["v"], ax)
        norm_to_f = {cls.FACE_FRAMES[fn][0]: fi for fi, fn in enumerate(cls.FACE_NAMES)}
        mapping = [None] * 54
        for s in stickers:
            fd = norm_to_f[s["n"]]
            _, Ud, Vd = cls.FACE_FRAMES[cls.FACE_NAMES[fd]]
            cd = cls._dot(s["pos"], Ud) + 1
            rd = cls._dot(s["pos"], Vd) + 1
            dest_idx = fd * 9 + rd * 3 + cd
            du, dv = cls._dot(s["u"], Ud), cls._dot(s["u"], Vd)
            k = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}[(du, dv)]
            mapping[dest_idx] = (s["orig"], k)
        return mapping

    def confusion_matrix(self) -> list[list[int]]:
        N = [[0] * 6 for _ in range(6)]
        for dest_idx, (src_idx, _) in enumerate(self.mapping):
            N[dest_idx // 9][src_idx // 9] += 1
        return N

    def _get_idx(self, device: torch.device, inverse: bool = False) -> torch.Tensor:
        key = (device, inverse)
        if key not in self._idx_cache:
            arr = self._inv_idx if inverse else self._fwd_idx
            self._idx_cache[key] = torch.tensor(arr, dtype=torch.long, device=device)
        return self._idx_cache[key]

    def _permute_faces(self, faces: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """faces: [6, C, H, W] -> [6, C, H, W]. Exact 3x3 oriented sticker permutation."""
        _, c, h, w = faces.shape
        th, tw = h // 3, w // 3
        hc, wc = 3 * th, 3 * tw
        sub = faces[:, :, :hc, :wc]
        tiles = sub.view(6, c, 3, th, 3, tw).permute(0, 2, 4, 1, 3, 5).reshape(54, c, th, tw)
        variants = torch.cat(
            [
                tiles,
                tiles.rot90(1, dims=(2, 3)),
                tiles.rot90(2, dims=(2, 3)),
                tiles.rot90(3, dims=(2, 3)),
            ],
            dim=0,
        )
        out_tiles = variants[self._get_idx(faces.device, inverse=inverse)]
        out_sub = out_tiles.view(6, 3, 3, c, th, tw).permute(0, 3, 1, 4, 2, 5).reshape(6, c, hc, wc)
        if hc == h and wc == w:
            return out_sub
        out = faces.clone()
        out[:, :, :hc, :wc] = out_sub
        return out

    def derive(self, primes: list[torch.Tensor]) -> torch.Tensor:
        """primes: list of 6 tensors [C, H, W] -> [12, C, H, W] ([6 before, 6 after])."""
        before = torch.stack(primes, dim=0)
        after = self._permute_faces(before, inverse=False)
        return torch.cat([before, after], dim=0)

    def invert_and_average(self, views: torch.Tensor) -> torch.Tensor:
        """views: [12, C, H, W] -> reconciled views [12, C, H, W]."""
        before_in = views[:6]
        after_in = views[6:]
        before_rec = 0.5 * (before_in + self._permute_faces(after_in, inverse=True))
        return self.derive([before_rec[i] for i in range(6)])

    def invert_velocities(self, v: torch.Tensor) -> torch.Tensor:
        v_before = 0.5 * (v[:6] + self._permute_faces(v[6:], inverse=True))
        v_after = self._permute_faces(v_before, inverse=False)
        return torch.cat([v_before, v_after], dim=0)

    def match_scrambled_faces(self, views: torch.Tensor) -> tuple[list[int], float, float]:
        """Find optimal 1-to-1 bijection perm[face_j] = src_p minimizing MSE between
        scrambled(views[:6])[j] and views[6 + perm[j]].

        Returns (perm, identity_mse, matched_mse).
        """
        solved_scrambled = self._permute_faces(views[:6].detach().float(), inverse=False)
        cand = views[6:12].detach().float()
        a = solved_scrambled.reshape(6, -1)
        b = cand.reshape(6, -1)
        n = float(a.shape[1])
        a2 = (a * a).sum(dim=1, keepdim=True)
        b2 = (b * b).sum(dim=1, keepdim=True).t()
        cost = ((a2 + b2 - 2.0 * (a @ b.t())) / n).cpu().tolist()

        identity_cost = sum(cost[j][j] for j in range(6)) / 6.0
        best_cost = float("inf")
        best_perm = list(range(6))
        for p in itertools.permutations(range(6)):
            c = cost[0][p[0]] + cost[1][p[1]] + cost[2][p[2]] + cost[3][p[3]] + cost[4][p[4]] + cost[5][p[5]]
            if c < best_cost:
                best_cost = c
                best_perm = list(p)
        return best_perm, float(identity_cost), float(best_cost / 6.0)


class RotationOverlayOperator:
    """2 prime images (bottom, top) -> 4 rotated multiplicative overlay views in [0, 1]."""

    num_primes = 2
    num_views = 4
    is_permutation = False
    equivariant_to_latent = False

    def __init__(self, brightness: float = 3.0):
        self.brightness = brightness

    def derive(self, primes: list[torch.Tensor]) -> torch.Tensor:
        bottom, top = primes
        views = [
            torch.tanh((bottom * top.rot90(k, dims=(1, 2)) * self.brightness).clamp(0.0, 99.0))
            for k in range(4)
        ]
        return torch.stack(views, dim=0)

    @torch.enable_grad()
    def reconcile(
        self,
        targets: torch.Tensor,
        init_logits: tuple[torch.Tensor, torch.Tensor] | None = None,
        steps: int = 25,
        lr: float = 0.35,
    ) -> tuple[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """
        Fast GPU-vectorized prime solver for zero-shot overlay reconciliation.
        targets: [4, C, H, W] in [0, 1].
        Returns: (derived_views [4, C, H, W], primes [bottom, top], (b_logit, t_logit))
        """
        targets = targets.detach().float().clamp(1e-4, 1.0 - 1e-4)
        if init_logits is None:
            b_logit = torch.zeros_like(targets[0], requires_grad=True)
            t_logit = torch.zeros_like(targets[0], requires_grad=True)
        else:
            b_logit = init_logits[0].detach().clone().requires_grad_(True)
            t_logit = init_logits[1].detach().clone().requires_grad_(True)

        opt = torch.optim.Adam([b_logit, t_logit], lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            b = torch.sigmoid(b_logit)
            t = torch.sigmoid(t_logit)
            preds = self.derive([b, t])
            loss = F.mse_loss(preds, targets)
            loss.backward()
            opt.step()

        with torch.no_grad():
            b = torch.sigmoid(b_logit)
            t = torch.sigmoid(t_logit)
            derived = self.derive([b, t])
        return derived, [b, t], (b_logit.detach(), t_logit.detach())


class HiddenOverlayOperator:
    """4 prime images -> 5 views: [p0, p1, p2, p3, tanh(brightness * p0 * p1 * p2 * p3)]."""

    num_primes = 4
    num_views = 5
    is_permutation = False
    equivariant_to_latent = False

    def __init__(self, brightness: float = 3.0, hidden_weight: float = 3.0):
        self.brightness = brightness
        self.hidden_weight = hidden_weight

    def derive(self, primes: list[torch.Tensor]) -> torch.Tensor:
        a, b, c, d = primes
        z = torch.tanh((self.brightness * a * b * c * d).clamp(0.0, 99.0))
        return torch.stack([a, b, c, d, z], dim=0)

    def reconcile(
        self,
        targets: torch.Tensor,
        steps: int = 30,
        step_size: float = 0.015,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        GPU-vectorized version of Ryan's closed-form + projected gradient reconciler
        from notes/multigpu_hiddens.py. Runs in <1 ms on GPU.
        targets: [5, C, H, W] in [0, 1].
        """
        t = targets.detach().float().clamp(1e-3, 1.0 - 1e-3)
        ta, tb, tc, td, tz = t[0], t[1], t[2], t[3], t[4]
        bl = self.brightness
        lz = self.hidden_weight

        # Closed-form linearized initialization
        p = ta * tb * tc * td
        eps = tz - bl * p
        v_dot = (1.0 / (ta * ta)) + (1.0 / (tb * tb)) + (1.0 / (tc * tc)) + (1.0 / (td * td))
        scale = (bl * lz * p * eps) / (1.0 + (bl * bl) * lz * (p * p) * v_dot)
        a = (ta + scale / ta).clamp(0.0, 1.0)
        b = (tb + scale / tb).clamp(0.0, 1.0)
        c = (tc + scale / tc).clamp(0.0, 1.0)
        d = (td + scale / td).clamp(0.0, 1.0)

        for _ in range(steps):
            prod = a * b * c * d
            z = torch.tanh(bl * prod)
            err = tz - z
            sech2 = 1.0 - z * z
            common = -2.0 * bl * lz * err * sech2
            ga = -2.0 * (ta - a) + common * (b * c * d)
            gb = -2.0 * (tb - b) + common * (a * c * d)
            gc = -2.0 * (tc - c) + common * (a * b * d)
            gd = -2.0 * (td - d) + common * (a * b * c)
            a = (a - step_size * ga).clamp(0.0, 1.0)
            b = (b - step_size * gb).clamp(0.0, 1.0)
            c = (c - step_size * gc).clamp(0.0, 1.0)
            d = (d - step_size * gd).clamp(0.0, 1.0)

        derived = self.derive([a, b, c, d])
        return derived, [a, b, c, d]
