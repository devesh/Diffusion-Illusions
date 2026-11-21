"""Accelerated SDS + TAESD + Few-Step Modern DTL (SANA-Sprint / FLUX.2-klein / SDXL-Turbo)."""

from contextlib import nullcontext
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fast_illusions.apple_npu import get_best_torch_device, sync_torch_device
from fast_illusions.operators import (
    FastFourierImage,
    MultiResRasterImage,
    FastSSIM,
    dtl_loss,
    FlipOperator,
    TwistingSquaresOperator,
    RubiksCubeOperator,
    RotationOverlayOperator,
    HiddenOverlayOperator,
)


def _empty_device_cache(device: torch.device | None = None) -> None:
    d_type = device.type if device is not None else get_best_torch_device().type
    if d_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif d_type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


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


def get_operator(task: str):
    if task == "flip":
        return FlipOperator()
    if task == "twisting_squares":
        return TwistingSquaresOperator(num_divisions=4)
    if task == "rubiks_cube":
        return RubiksCubeOperator()
    if task == "rotation_overlay":
        return RotationOverlayOperator(brightness=3.0)
    if task == "hidden_overlay":
        return HiddenOverlayOperator(brightness=3.0, hidden_weight=3.0)
    raise ValueError(f"Unknown task: {task}")


def run_fast_sds_dtl(
    task: str,
    prompts: list[str],
    negative_prompt: str = "",
    sds_iters: int = 400,
    dtl_rounds: int = 2,
    dtl_steps_per_round: int = 120,
    dtl_model: str = "sana_sprint",  # "sana_sprint" | "sd_turbo" | "sdxl_turbo" | "none"
    sds_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    size: int = 512,
    use_taesd: bool = False,
    use_compile: bool = True,
    param_type: str = "fourier",  # "fourier" | "multires"
    init_primes: list[torch.Tensor] | None = None,
    t_start_hi: int = 960,
    t_end_hi: int = 500,
    out_dir: str = "/tmp/out_fast_sds_dtl",
    seed: int = 42,
) -> dict:
    """
    High-speed optimization pipeline:
    1. Batched BF16 SDS with torch.compile, AdamW, and coarse-to-fine timestep annealing.
    2. Few-step Dream Target Loss (DTL) using modern 1-4 step open-weight models (SANA-Sprint / SDXL-Turbo).
    """
    from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderTiny

    torch.manual_seed(seed)
    device = get_best_torch_device()
    dtype = torch.bfloat16 if device.type == "cuda" else (torch.float16 if device.type == "mps" else torch.float32)
    use_compile = bool(use_compile and device.type == "cuda")
    autocast_ctx = torch.autocast(device.type, dtype=dtype) if device.type in ("cuda", "mps") else nullcontext()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    op = get_operator(task)
    assert len(prompts) == op.num_views, f"Expected {op.num_views} prompts for {task}, got {len(prompts)}"

    t_load0 = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(
        sds_model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)

    if use_taesd:
        taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dtype).to(device)
        taesd.eval()
        for p in taesd.parameters():
            p.requires_grad_(False)
        encode_fn = lambda imgs: taesd.encode(2.0 * imgs - 1.0).latents
    else:
        pipe.vae.eval()
        for p in pipe.vae.parameters():
            p.requires_grad_(False)
        encode_fn = lambda imgs: pipe.vae.encode(2.0 * imgs - 1.0).latent_dist.mean * 0.18215

    unet = pipe.unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)

    # Precompute all prompt embeddings once (batched CFG: [uncond_0..N-1, cond_0..N-1])
    with torch.no_grad():
        cond_list = []
        uncond_list = []
        for p_txt in prompts:
            c_emb, u_emb = pipe.encode_prompt(
                prompt=p_txt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            cond_list.append(c_emb)
            uncond_list.append(u_emb)
        cond_embs = torch.cat(cond_list, dim=0)
        uncond_embs = torch.cat(uncond_list, dim=0)
        cfg_embs = torch.cat([uncond_embs, cond_embs], dim=0)  # [2*V, 77, 768]

    # Free text encoder VRAM
    del pipe.text_encoder
    _empty_device_cache(device)

    if use_compile:
        unet = torch.compile(unet, mode="reduce-overhead")

    # Initialize prime images
    if param_type == "fourier" and init_primes is None:
        primes = nn.ModuleList([
            FastFourierImage(height=size, width=size, hidden_dim=256, num_features=128).to(device)
            for _ in range(op.num_primes)
        ])
        lr = 5e-4
    else:
        primes = nn.ModuleList([
            MultiResRasterImage(height=size, width=size).to(device)
            for _ in range(op.num_primes)
        ])
        if init_primes is not None:
            with torch.no_grad():
                for idx, mod in enumerate(primes):
                    target = init_primes[idx].to(device=device, dtype=torch.float32)
                    if target.shape[-2:] != (size, size):
                        target = F.interpolate(target.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
                    target = target.clamp(1e-3, 1.0 - 1e-3)
                    logits = torch.log(target / (1.0 - target)).unsqueeze(0)
                    c_init = F.interpolate(logits, size=(size // 8, size // 8), mode="bilinear", align_corners=False)
                    c_up = F.interpolate(c_init, size=(size, size), mode="bilinear", align_corners=False)
                    mod.coarse.copy_(c_init)
                    mod.mid.zero_()
                    mod.fine.copy_(4.0 * (logits - c_up))
        lr = 1.5e-2 if init_primes is not None else 2e-2

    optim = torch.optim.AdamW(primes.parameters(), lr=lr, weight_decay=0.0)

    # Warmup compile on 2 steps before timing if compiled
    if use_compile:
        with torch.no_grad():
            dummy_lat = torch.randn(op.num_views, 4, 64, 64, device=device, dtype=dtype)
            dummy_in = torch.cat([dummy_lat, dummy_lat], dim=0)
            dummy_t = torch.tensor([500], device=device, dtype=torch.long)
            _ = unet(dummy_in, dummy_t, encoder_hidden_states=cfg_embs).sample
        sync_torch_device(device)

    load_s = time.perf_counter() - t_load0

    # ------------------ Phase 1: Fast Batched SDS ------------------
    sync_torch_device(device)
    t_sds0 = time.perf_counter()
    num_views = op.num_views
    guidance_scale = 50.0

    for it in range(sds_iters):
        optim.zero_grad(set_to_none=True)
        frac = it / max(1, sds_iters - 1)
        # Coarse-to-fine timestep annealing (DreamTime-inspired)
        t_hi = int(t_start_hi - (t_start_hi - t_end_hi) * frac)
        t_lo = int(200 - 180 * frac) if init_primes is None else int(120 - 100 * frac)
        t = torch.randint(t_lo, max(t_lo + 1, t_hi), (1,), device=device, dtype=torch.long)

        views = op.derive([p() for p in primes])  # [V, 3, H, W] in [0, 1]
        if size != 512:
            views_512 = F.interpolate(views, size=(512, 512), mode="bilinear", align_corners=False)
        else:
            views_512 = views

        with autocast_ctx:
            latents = encode_fn(views_512)  # [V, 4, 64, 64]

        with torch.no_grad():
            noise = torch.randn_like(latents, dtype=torch.float32)
            a_t = alphas_cumprod[t]
            sqrt_a = a_t.sqrt().view(1, 1, 1, 1)
            sqrt_one_minus_a = (1.0 - a_t).sqrt().view(1, 1, 1, 1)
            latents_noisy = (sqrt_a * latents.float() + sqrt_one_minus_a * noise).to(dtype)

            latent_model_input = torch.cat([latents_noisy, latents_noisy], dim=0)
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=cfg_embs).sample.float()
            n_uncond, n_cond = noise_pred.chunk(2, dim=0)
            n_cfg = n_uncond + guidance_scale * (n_cond - n_uncond)

            w = (1.0 - a_t).view(1, 1, 1, 1)
            grad = w * (n_cfg - noise) * (0.1 / num_views)

        latents.backward(gradient=grad.to(latents.dtype))
        optim.step()

    sync_torch_device(device)
    sds_s = time.perf_counter() - t_sds0

    with torch.no_grad():
        views_after_sds = op.derive([p() for p in primes])
        _make_grid([_tensor_to_pil(v) for v in views_after_sds]).save(out_path / "after_sds_grid.png")

    # ------------------ Phase 2: Few-Step Dream Target Loss (DTL) ------------------
    dtl_load_s = 0.0
    dtl_s = 0.0
    if dtl_model != "none" and dtl_rounds > 0:
        # Free SDS UNet first
        del unet, pipe
        if use_taesd:
            del taesd
        _empty_device_cache(device)

        t_dtl_load0 = time.perf_counter()
        if dtl_model == "sana_sprint":
            from diffusers import SanaSprintImg2ImgPipeline

            img2img = SanaSprintImg2ImgPipeline.from_pretrained(
                "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
                torch_dtype=dtype,
            ).to(device)
            img2img.vae.to(dtype=torch.float32)
            dtl_steps = 2
            dtl_size = 512
        elif dtl_model == "sdxl_turbo":
            from diffusers import AutoPipelineForImage2Image

            img2img = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=dtype,
                variant="fp16" if dtype in (torch.float16, torch.bfloat16) else None,
            ).to(device)
            img2img.vae.to(dtype=torch.float32)
            dtl_steps = 4
            dtl_size = 512
        elif dtl_model == "sd_turbo":
            from diffusers import AutoPipelineForImage2Image

            sd_dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
            img2img = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/sd-turbo",
                torch_dtype=sd_dtype,
            ).to(device)
            dtl_steps = 4
            dtl_size = 512
        else:
            raise ValueError(f"Unknown dtl_model: {dtl_model}")
        sync_torch_device(device)
        dtl_load_s = time.perf_counter() - t_dtl_load0

        ssim_fn = FastSSIM().to(device)
        dtl_optim = torch.optim.AdamW(primes.parameters(), lr=1.5e-3 if param_type == "fourier" else 2e-2)

        sync_torch_device(device)
        t_dtl0 = time.perf_counter()
        for r in range(dtl_rounds):
            strength = 0.35 - 0.15 * (r / max(1, dtl_rounds - 1)) if dtl_rounds > 1 else 0.30
            # Ensure strength * steps >= 1 for turbo/sprint
            if dtl_model in ("sd_turbo", "sdxl_turbo"):
                steps_for_round = max(2, int(round(dtl_steps * strength)))
                eff_steps = max(2, int(round(steps_for_round / strength)))
            else:
                eff_steps = dtl_steps

            with torch.no_grad():
                cur_views = op.derive([p() for p in primes])
                pils_in = [_tensor_to_pil(v).resize((dtl_size, dtl_size), Image.BICUBIC) for v in cur_views]
                if dtl_model == "sana_sprint":
                    out_pils = img2img(
                        prompt=prompts,
                        image=pils_in,
                        strength=strength,
                        num_inference_steps=eff_steps,
                        height=dtl_size,
                        width=dtl_size,
                    ).images
                else:
                    out_pils = img2img(
                        prompt=prompts,
                        image=pils_in,
                        strength=strength,
                        num_inference_steps=eff_steps,
                        guidance_scale=0.0,
                    ).images
                import numpy as np
                targets = torch.stack([
                    torch.from_numpy(
                        np.array(out_pil.resize((size, size), Image.BICUBIC))
                    ).permute(2, 0, 1).float().to(device) / 255.0
                    for out_pil in out_pils
                ], dim=0)

            # Fast GPU regression toward DTL targets
            for _ in range(dtl_steps_per_round):
                dtl_optim.zero_grad(set_to_none=True)
                preds = op.derive([p() for p in primes])
                loss = dtl_loss(preds, targets, ssim_fn, ssim_weight=0.25)
                loss.backward()
                dtl_optim.step()

        sync_torch_device(device)
        dtl_s = time.perf_counter() - t_dtl0

    with torch.no_grad():
        for idx, p in enumerate(primes):
            _tensor_to_pil(p()).save(out_path / f"prime_{idx}.png")
        final_views = op.derive([p() for p in primes])
        final_pils = [_tensor_to_pil(v) for v in final_views]
        for idx, v in enumerate(final_pils):
            v.save(out_path / f"view_{idx}.png")
        _make_grid(final_pils).save(out_path / "final_grid.png")

    return {
        "method": f"fast_sds_{'taesd_' if use_taesd else 'vae_'}dtl_{dtl_model}",
        "task": task,
        "sds_load_s": round(load_s, 3),
        "sds_iters": sds_iters,
        "sds_s": round(sds_s, 3),
        "dtl_load_s": round(dtl_load_s, 3),
        "dtl_s": round(dtl_s, 3),
        "total_opt_s": round(sds_s + dtl_s, 3),
        "out_dir": str(out_path),
    }


def run_hybrid_zeroshot_sds(
    task: str,
    prompts: list[str],
    negative_prompt: str = "",
    init_method: str = "sana_sprint",
    init_steps: int = 4,
    sds_iters: int = 250,
    size: int = 512,
    out_dir: str = "/tmp/out_hybrid_sds",
    seed: int = 42,
) -> dict:
    """
    Two-stage hybrid:
    1) Zero-shot pixel-reconciled layout initialization (SANA-Sprint or FLUX.2-klein) in ~2-3s.
    2) Short batched BF16 + torch.compile SDS polish in ~5s starting from the reconciled primes.
    """
    import gc
    import numpy as np
    from fast_illusions.zeroshot_modern import (
        run_sana_sprint_zeroshot,
        run_flux2_klein_zeroshot,
        _get_operator,
    )

    out_path = Path(out_dir)
    init_dir = out_path / "_init"
    init_dir.mkdir(parents=True, exist_ok=True)

    if init_method == "sana_sprint":
        init_metrics = run_sana_sprint_zeroshot(
            task=task,
            prompts=prompts,
            num_steps=init_steps,
            size=768 if task == "rubiks_cube" else 1024,
            out_dir=str(init_dir),
            seed=seed,
        )
    elif init_method == "flux2_klein":
        init_metrics = run_flux2_klein_zeroshot(
            task=task,
            prompts=prompts,
            num_steps=init_steps,
            size=size,
            out_dir=str(init_dir),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown init_method: {init_method}")

    op = _get_operator(task)
    init_primes = []
    for idx in range(op.num_primes):
        arr = np.array(Image.open(init_dir / f"prime_{idx}.png").convert("RGB")).astype(np.float32) / 255.0
        init_primes.append(torch.from_numpy(arr).permute(2, 0, 1))

    gc.collect()
    _empty_device_cache()

    sds_prompts = prompts
    if init_metrics.get("scrambled_perm") and len(prompts) == 12:
        perm = init_metrics["scrambled_perm"]
        sds_prompts = prompts[:6] + [prompts[6 + p] for p in perm]

    sds_metrics = run_fast_sds_dtl(
        task=task,
        prompts=sds_prompts,
        negative_prompt=negative_prompt,
        sds_iters=sds_iters,
        dtl_rounds=0,
        dtl_model="none",
        size=size,
        use_taesd=True,
        use_compile=True,
        param_type="multires",
        init_primes=init_primes,
        t_start_hi=650,
        t_end_hi=250,
        out_dir=str(out_path),
        seed=seed,
    )

    res = {
        "method": f"hybrid_{init_method}_fast_sds",
        "task": task,
        "init_gen_s": init_metrics["gen_s"],
        "init_steps": init_steps,
        "sds_iters": sds_iters,
        "sds_s": sds_metrics["sds_s"],
        "total_opt_s": round(init_metrics["gen_s"] + sds_metrics["sds_s"], 3),
        "out_dir": str(out_path),
    }
    for key in ("scrambled_perm", "scrambled_face_prompts", "match_mse_before", "match_mse_after"):
        if key in init_metrics:
            res[key] = init_metrics[key]
    return res
