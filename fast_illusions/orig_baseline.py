"""Faithful wrapper around the original RyannDaGreat/Diffusion-Illusions pipeline for benchmarking."""

import sys
import time
from pathlib import Path
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr)


def _make_grid(pils: list[Image.Image]) -> Image.Image:
    w, h = pils[0].size
    n = len(pils)
    cols = 6 if n == 12 else n
    rows = (n + cols - 1) // cols
    grid = Image.new("RGB", (w * cols, h * rows))
    for i, im in enumerate(pils):
        r, c = divmod(i, cols)
        grid.paste(im, (c * w, r * h))
    return grid


def run_orig_baseline(
    task: str,
    prompts: list[str],
    negative_prompt: str = "",
    num_iters: int = 2500,
    size: int = 256,
    checkpoint: str = "CompVis/stable-diffusion-v1-4",
    snapshot_iters: tuple[int, ...] = (500, 1000, 2500),
    out_dir: str = "/tmp/out_orig",
    seed: int = 42,
) -> dict:
    """
    Runs the exact original fp32 + SGD(lr=1e-4) + batch_size=1 SDS loop from the repo.
    """
    import rp
    import source.stable_diffusion as sd
    from source.learnable_textures import LearnableImageFourier
    from source.stable_diffusion_labels import NegativeLabel

    torch.manual_seed(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t_load0 = time.perf_counter()
    # Reset singleton if needed across calls
    sd._stable_diffusion_singleton = None
    s = sd.StableDiffusion("cuda:0", checkpoint)
    s.max_step = 990
    s.min_step = 10
    load_s = time.perf_counter() - t_load0

    labels = [NegativeLabel(p, negative_prompt) for p in prompts]

    if task == "flip":
        assert len(prompts) == 2
        img = LearnableImageFourier(height=size, width=size, hidden_dim=256, num_features=128).to(s.device)
        learnable_images = [
            lambda: img(),
            lambda: img().rot90(k=2, dims=[1, 2]),
        ]
        primes = [img]
        optim = torch.optim.SGD(img.parameters(), lr=1e-4)
    elif task == "twisting_squares":
        assert len(prompts) == 2
        from fast_illusions.operators import TwistingSquaresOperator
        tw_op = TwistingSquaresOperator(num_divisions=4)
        img = LearnableImageFourier(height=size, width=size, hidden_dim=256, num_features=128).to(s.device)
        learnable_images = [
            lambda: img(),
            lambda: tw_op._twist(img(), direction=1),
        ]
        primes = [img]
        optim = torch.optim.SGD(img.parameters(), lr=1e-4)
    elif task == "rubiks_cube":
        assert len(prompts) == 12
        from fast_illusions.operators import RubiksCubeOperator
        rb_op = RubiksCubeOperator()
        primes = [LearnableImageFourier(height=size, width=size, hidden_dim=256, num_features=128).to(s.device) for _ in range(6)]
        learnable_images = [
            (lambda i=i: (lambda: primes[i]()))() for i in range(6)
        ] + [
            (lambda j=j: (lambda: rb_op._permute_faces(torch.stack([p() for p in primes], dim=0))[j]))()
            for j in range(6)
        ]
        params = [p for im in primes for p in im.parameters()]
        optim = torch.optim.SGD(params, lr=1e-4)
    elif task == "rotation_overlay":
        assert len(prompts) == 4
        bottom = LearnableImageFourier(height=size, width=size, hidden_dim=256, num_features=128).to(s.device)
        top = LearnableImageFourier(height=size, width=size, hidden_dim=256, num_features=128).to(s.device)

        def sim_overlay(b, t):
            return (b * t * 3).clamp(0, 99).tanh()

        learnable_images = [
            (lambda k=k: (lambda: sim_overlay(bottom(), top().rot90(k=k, dims=[1, 2]))))()
            for k in range(4)
        ]
        primes = [bottom, top]
        optim = torch.optim.SGD(list(bottom.parameters()) + list(top.parameters()), lr=1e-4)
    elif task == "hidden_overlay":
        assert len(prompts) == 5
        imgs = [LearnableImageFourier(height=size, width=size, hidden_dim=256, num_features=128).to(s.device) for _ in range(4)]
        learnable_images = [
            lambda: imgs[0](),
            lambda: imgs[1](),
            lambda: imgs[2](),
            lambda: imgs[3](),
            lambda: (imgs[0]() * imgs[1]() * imgs[2]() * imgs[3]() * 3).clamp(0, 99).tanh(),
        ]
        primes = imgs
        params = [p for im in imgs for p in im.parameters()]
        optim = torch.optim.SGD(params, lr=1e-4)
    else:
        raise ValueError(f"Unsupported task for baseline: {task}")

    weights = [1.0] * len(learnable_images)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    timings = {}

    for it in range(1, num_iters + 1):
        for label, learnable_image, weight in rp.random_batch(
            list(zip(labels, learnable_images, weights)), batch_size=1
        ):
            s.train_step(
                label.embedding,
                learnable_image()[None],
                noise_coef=0.1 * weight,
                guidance_scale=60,
            )
        optim.step()
        optim.zero_grad()

        if it in snapshot_iters or it == num_iters:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            timings[it] = elapsed
            with torch.no_grad():
                views = [_tensor_to_pil(fn()) for fn in learnable_images]
                for idx, v in enumerate(views):
                    v.save(out_path / f"iter_{it:05d}_view_{idx}.png")
                _make_grid(views).save(out_path / f"iter_{it:05d}_grid.png")

    torch.cuda.synchronize()
    total_s = time.perf_counter() - t0
    ms_per_iter = (total_s / num_iters) * 1000.0
    with torch.no_grad():
        for idx, p in enumerate(primes):
            _tensor_to_pil(p()).save(out_path / f"prime_{idx}.png")
        views = [_tensor_to_pil(fn()) for fn in learnable_images]
        _make_grid(views).save(out_path / "final_grid.png")

    return {
        "method": "orig_sd14_sds",
        "task": task,
        "load_s": round(load_s, 3),
        "num_iters": num_iters,
        "total_s": round(total_s, 3),
        "ms_per_iter": round(ms_per_iter, 2),
        "est_10k_iters_s": round(ms_per_iter * 10.0, 2),
        "snapshots_s": {str(k): round(v, 3) for k, v in timings.items()},
        "out_dir": str(out_path),
    }
