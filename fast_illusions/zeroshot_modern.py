"""Zero-shot few-step illusion generation using modern open-weight models."""

from contextlib import nullcontext
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fast_illusions.apple_npu import get_best_torch_device, sync_torch_device
from fast_illusions.operators import (
    FlipOperator,
    TwistingSquaresOperator,
    RubiksCubeOperator,
    RotationOverlayOperator,
    HiddenOverlayOperator,
)


def _preferred_dtype(device: torch.device, prefer_bf16: bool = True) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if prefer_bf16 else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _reorder_rubiks_scrambled(
    op,
    imgs: torch.Tensor,
    prompts: list[str],
    *extra_tensors: torch.Tensor | None,
) -> dict | None:
    if not isinstance(op, RubiksCubeOperator):
        return None
    perm, mse_before, mse_after = op.match_scrambled_faces(imgs)
    idx_cpu = torch.tensor(perm, dtype=torch.long)
    imgs[6:12] = imgs[6:12][idx_cpu.to(imgs.device)]
    for t in extra_tensors:
        if t is not None:
            t[6:12] = t[6:12][idx_cpu.to(t.device)]
    return {
        "scrambled_perm": perm,
        "scrambled_face_prompts": {fn: prompts[6 + p] for fn, p in zip(op.FACE_NAMES, perm)},
        "match_mse_before": round(mse_before, 5),
        "match_mse_after": round(mse_after, 5),
    }


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


def _get_operator(task: str):
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


# ---------------------------------------------------------------------------
# 1. Iris-3B Turbo (4 steps, pure pixel-space DiT, zero VAE)
# ---------------------------------------------------------------------------

def run_iris3b_turbo_zeroshot(
    task: str,
    prompts: list[str],
    size: int = 1024,
    out_dir: str = "/tmp/out_iris3b",
    seed: int = 42,
    reconcile_steps: int = 60,
) -> dict:
    """
    4-step pixel-space multi-view / reconciled generation using speridlabs/iris-3b (Turbo).
    No VAE encode/decode, no CFG, no SDS backprop.
    """
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    from iris3b.config import inference_config
    from iris3b.models.dit import IrisDiT
    from iris3b.registry import TEXT_ENCODERS
    from iris3b.sampling import load_for_inference
    import iris3b.text  # noqa: F401

    op = _get_operator(task)
    assert len(prompts) == op.num_views

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = get_best_torch_device()
    model_dtype = _preferred_dtype(device, prefer_bf16=True)
    autocast_ctx = torch.autocast(device.type, dtype=model_dtype) if device.type in ("cuda", "mps") else nullcontext()

    t_load0 = time.perf_counter()
    ckpt_dir = Path(
        snapshot_download("speridlabs/iris-3b", allow_patterns=["turbo/*"])
    ) / "turbo"
    raw, weights = load_for_inference(ckpt_dir)
    cfg = inference_config(raw, [])
    sigmas = list(OmegaConf.load(ckpt_dir / "distill.yaml").dmd.sigmas)

    with torch.device("meta"):
        model = IrisDiT(cfg.model)
    model.load_state_dict(weights, strict=True, assign=True)
    model = model.eval().to(device=device, dtype=model_dtype)

    text_encoder = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device=device)
    sync_torch_device(device)
    load_s = time.perf_counter() - t_load0

    sync_torch_device(device)
    t0 = time.perf_counter()
    gen = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad(), autocast_ctx:
        encoding = text_encoder.encode(prompts)
        cond = encoding.embeddings.to(device=device, dtype=model_dtype)
        mask = encoding.mask.to(device=device)

        if op.is_permutation:
            base_noise = torch.randn(op.num_primes, model.cfg.in_channels, size, size, generator=gen, device=device)
            x = op.derive([base_noise[i] for i in range(op.num_primes)])
        else:
            x = torch.randn(op.num_views, model.cfg.in_channels, size, size, generator=gen, device=device)

        warm_logits = None
        primes = None
        match_info = None
        num_train_timesteps = cfg.flow.num_train_timesteps
        prediction = cfg.flow.prediction
        chunk = 6 if (x.shape[0] > 6 and size >= 576) else x.shape[0]

        sigma = torch.full((x.shape[0], 1, 1, 1), sigmas[0], device=device, dtype=torch.float32)
        for i in range(len(sigmas)):
            if chunk < x.shape[0]:
                outs = []
                t_all = sigma.flatten() * num_train_timesteps
                for s0 in range(0, x.shape[0], chunk):
                    s1 = s0 + chunk
                    outs.append(model(x[s0:s1].to(model_dtype), t_all[s0:s1], cond[s0:s1], y_mask=mask[s0:s1]).x.float())
                out = torch.cat(outs, dim=0)
            else:
                out = model(
                    x.to(model_dtype),
                    sigma.flatten() * num_train_timesteps,
                    cond,
                    y_mask=mask,
                ).x.float()
            clean = out if prediction == "x" else x - sigma * out

            if i == 0 and isinstance(op, RubiksCubeOperator):
                clean_01_match = ((clean + 1.0) * 0.5).clamp(0.0, 1.0)
                match_info = _reorder_rubiks_scrambled(op, clean_01_match, prompts, clean, x, cond, mask)

            if op.is_permutation:
                clean_rec = op.invert_and_average(clean)
            else:
                clean_01 = ((clean + 1.0) * 0.5).clamp(0.0, 1.0)
                if isinstance(op, RotationOverlayOperator):
                    clean_rec_01, primes, warm_logits = op.reconcile(
                        clean_01, init_logits=warm_logits, steps=reconcile_steps
                    )
                else:
                    clean_rec_01, primes = op.reconcile(clean_01)
                clean_rec = 2.0 * clean_rec_01 - 1.0

            if i + 1 < len(sigmas):
                implied_noise = (x - (1.0 - sigma) * clean_rec) / sigma
                sigma = torch.full_like(sigma, sigmas[i + 1])
                x = (1.0 - sigma) * clean_rec + sigma * implied_noise

    sync_torch_device(device)
    gen_s = time.perf_counter() - t0

    if op.is_permutation:
        raw_01 = ((clean_rec + 1.0) * 0.5).clamp(0.0, 1.0)
        primes = [raw_01[i] for i in range(op.num_primes)]
        views_01 = op.derive(primes)
    else:
        views_01 = op.derive(primes)

    for idx, p in enumerate(primes):
        _tensor_to_pil(p).save(out_path / f"prime_{idx}.png")
    view_pils = [_tensor_to_pil(v) for v in views_01]
    for idx, v in enumerate(view_pils):
        v.save(out_path / f"view_{idx}.png")
    _make_grid(view_pils).save(out_path / "final_grid.png")

    res = {
        "method": "iris3b_turbo_zeroshot",
        "task": task,
        "steps": len(sigmas),
        "size": size,
        "load_s": round(load_s, 3),
        "gen_s": round(gen_s, 3),
        "out_dir": str(out_path),
    }
    if match_info:
        res.update(match_info)
    return res


# ---------------------------------------------------------------------------
# 2. SD-Turbo / SDXL-Turbo (4-6 steps UNet + TAESD/TAESDXL)
# ---------------------------------------------------------------------------

def run_sd_turbo_zeroshot(
    task: str,
    prompts: list[str],
    model_id: str = "stabilityai/sdxl-turbo",
    num_steps: int = 4,
    size: int = 512,
    out_dir: str = "/tmp/out_sdxl_turbo",
    seed: int = 42,
    reconcile_steps: int = 50,
    pixel_reconcile: bool | None = None,
) -> dict:
    """
    4-step CFG-free UNet zero-shot sampling with SDXL-Turbo or SD-Turbo.
    - Equivariant permutations (flip): direct latent velocity/noise averaging.
    - Non-equivariant permutations (twisting_squares, rubiks_cube) & overlays: step-wise VAE decode -> pixel reconciliation -> VAE encode.
    """
    from diffusers import AutoPipelineForText2Image, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler

    op = _get_operator(task)
    assert len(prompts) == op.num_views
    if pixel_reconcile is None:
        pixel_reconcile = (not op.is_permutation) or (not op.equivariant_to_latent)

    is_xl = "xl" in model_id.lower()
    device = get_best_torch_device()
    dtype = _preferred_dtype(device, prefer_bf16=is_xl)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t_load0 = time.perf_counter()
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if (is_xl and dtype in (torch.float16, torch.bfloat16)) else None,
        safety_checker=None,
    ).to(device)
    if is_xl:
        pipe.vae.to(dtype=torch.float32)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
    else:
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
    sync_torch_device(device)
    load_s = time.perf_counter() - t_load0

    sync_torch_device(device)
    t0 = time.perf_counter()
    gen = torch.Generator(device=device).manual_seed(seed)

    def _vae_decode_chunked(z: torch.Tensor, chunk: int = 4) -> torch.Tensor:
        if z.shape[0] <= chunk:
            return pipe.vae.decode(z, return_dict=False)[0]
        return torch.cat([pipe.vae.decode(z[k : k + chunk], return_dict=False)[0] for k in range(0, z.shape[0], chunk)], dim=0)

    def _vae_encode_chunked(x_in: torch.Tensor, chunk: int = 4) -> torch.Tensor:
        if x_in.shape[0] <= chunk:
            return pipe.vae.encode(x_in).latent_dist.mean
        return torch.cat([pipe.vae.encode(x_in[k : k + chunk]).latent_dist.mean for k in range(0, x_in.shape[0], chunk)], dim=0)

    with torch.no_grad():
        if is_xl:
            (
                prompt_embeds,
                _,
                pooled_prompt_embeds,
                _,
            ) = pipe.encode_prompt(
                prompt=prompts,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            add_time_ids = pipe._get_add_time_ids(
                (size, size), (0, 0), (size, size), dtype=prompt_embeds.dtype, text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim
            ).to(device).repeat(op.num_views, 1)
            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
        else:
            prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompts,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            added_cond_kwargs = None

        pipe.scheduler.set_timesteps(num_steps, device=device)
        timesteps = pipe.scheduler.timesteps
        lat_h, lat_w = size // 8, size // 8

        if op.is_permutation and not pixel_reconcile:
            base_lat = torch.randn(op.num_primes, 4, lat_h, lat_w, generator=gen, device=device, dtype=dtype)
            latents = op.derive([base_lat[k] for k in range(op.num_primes)]) * pipe.scheduler.init_noise_sigma
        else:
            latents = torch.randn(op.num_views, 4, lat_h, lat_w, generator=gen, device=device, dtype=dtype)
            latents = latents * pipe.scheduler.init_noise_sigma

        warm_logits = None
        primes = None
        last_rec_01 = None
        match_info = None
        for i, t in enumerate(timesteps):
            latent_model_input = pipe.scheduler.scale_model_input(latents, t)
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # EulerDiscreteScheduler / EulerAncestralDiscreteScheduler step uses sigma_t and sigma_next
            sigma = pipe.scheduler.sigmas[i]
            sigma_next = pipe.scheduler.sigmas[i + 1]
            if pipe.scheduler.config.prediction_type == "epsilon":
                pred_x0 = latents.float() - sigma * noise_pred.float()
            elif pipe.scheduler.config.prediction_type == "v_prediction":
                alpha_t = 1.0 / ((sigma ** 2 + 1.0) ** 0.5)
                sigma_t = sigma * alpha_t
                pred_x0 = alpha_t * latents.float() - sigma_t * noise_pred.float()
            else:
                pred_x0 = noise_pred.float()

            if op.is_permutation and not pixel_reconcile:
                if pipe.scheduler.config.prediction_type == "epsilon":
                    noise_pred_avg = op.invert_velocities(noise_pred.float()).to(dtype)
                    step_kw = {"generator": gen} if is_xl else {}
                    latents = pipe.scheduler.step(noise_pred_avg, t, latents, **step_kw, return_dict=False)[0]
                else:
                    pred_x0_rec = op.invert_and_average(pred_x0)
                    eps_eff = (latents.float() - pred_x0_rec) / sigma.clamp(min=1e-5)
                    latents = (latents.float() + (sigma_next - sigma) * eps_eff).to(dtype)
                    pipe.scheduler._step_index = i + 1
            else:
                # Reconcile in pixel space via exact VAE
                pred_x0_scaled = (pred_x0 / pipe.vae.config.scaling_factor).to(pipe.vae.dtype)
                imgs = _vae_decode_chunked(pred_x0_scaled)
                imgs_01 = ((imgs.float() + 1.0) * 0.5).clamp(0.0, 1.0)
                if i == 0 and isinstance(op, RubiksCubeOperator):
                    extras = [latents, prompt_embeds]
                    if added_cond_kwargs is not None:
                        extras.extend([added_cond_kwargs.get("text_embeds"), added_cond_kwargs.get("time_ids")])
                    match_info = _reorder_rubiks_scrambled(op, imgs_01, prompts, *extras)
                if op.is_permutation:
                    rec_01 = op.invert_and_average(imgs_01)
                    primes = [rec_01[k] for k in range(op.num_primes)]
                elif isinstance(op, RotationOverlayOperator):
                    rec_01, primes, warm_logits = op.reconcile(
                        imgs_01, init_logits=warm_logits, steps=reconcile_steps
                    )
                else:
                    rec_01, primes = op.reconcile(imgs_01)
                last_rec_01 = rec_01
                rec_lat = _vae_encode_chunked((2.0 * rec_01 - 1.0).to(pipe.vae.dtype)).float() * pipe.vae.config.scaling_factor
                # Step along Euler ODE toward reconciled x0
                if sigma > 1e-4:
                    deriv = (latents.float() - rec_lat) / sigma
                    latents = (latents.float() + (sigma_next - sigma) * deriv).to(dtype)
                else:
                    latents = rec_lat.to(dtype)

        if op.is_permutation and not pixel_reconcile:
            lat_f = (latents.float() / pipe.vae.config.scaling_factor).to(pipe.vae.dtype)
            imgs = _vae_decode_chunked(lat_f)
            decoded_01 = ((imgs.float() + 1.0) * 0.5).clamp(0.0, 1.0)
            primes = [decoded_01[k] for k in range(op.num_primes)]
            views_01 = op.derive(primes)
        elif op.is_permutation:
            primes = [last_rec_01[k] for k in range(op.num_primes)]
            views_01 = op.derive(primes)
        else:
            views_01 = op.derive(primes)

    sync_torch_device(device)
    gen_s = time.perf_counter() - t0

    for idx, p in enumerate(primes):
        _tensor_to_pil(p).save(out_path / f"prime_{idx}.png")
    view_pils = [_tensor_to_pil(v) for v in views_01]
    for idx, v in enumerate(view_pils):
        v.save(out_path / f"view_{idx}.png")
    _make_grid(view_pils).save(out_path / "final_grid.png")

    tag = "sdxl_turbo" if is_xl else "sd_turbo"
    res = {
        "method": f"{tag}_zeroshot",
        "task": task,
        "steps": num_steps,
        "size": size,
        "pixel_reconcile": bool(pixel_reconcile),
        "load_s": round(load_s, 3),
        "gen_s": round(gen_s, 3),
        "out_dir": str(out_path),
    }
    if match_info:
        res.update(match_info)
    return res


# ---------------------------------------------------------------------------
# 3. FLUX.2 [klein] 4B (4-step Rectified Flow, Apache 2.0)
# ---------------------------------------------------------------------------

def run_flux2_klein_zeroshot(
    task: str,
    prompts: list[str],
    model_id: str = "black-forest-labs/FLUX.2-klein-4B",
    num_steps: int = 4,
    size: int = 512,
    out_dir: str = "/tmp/out_flux2_klein",
    seed: int = 42,
    reconcile_steps: int = 60,
    pixel_reconcile: bool | None = None,
) -> dict:
    """
    4-8 step Rectified Flow multi-view / reconciled illusion sampling with FLUX.2-klein-4B.
    """
    from diffusers import Flux2KleinPipeline
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps

    op = _get_operator(task)
    assert len(prompts) == op.num_views
    if pixel_reconcile is None:
        pixel_reconcile = (not op.is_permutation) or (not op.equivariant_to_latent)

    device = get_best_torch_device()
    dtype = _preferred_dtype(device, prefer_bf16=True)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t_load0 = time.perf_counter()
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    sync_torch_device(device)
    load_s = time.perf_counter() - t_load0

    sync_torch_device(device)
    t0 = time.perf_counter()
    gen = torch.Generator(device=device).manual_seed(seed)

    def _vae_decode_chunked(z: torch.Tensor, chunk: int = 4) -> torch.Tensor:
        if z.shape[0] <= chunk:
            return pipe.vae.decode(z, return_dict=False)[0]
        return torch.cat([pipe.vae.decode(z[k : k + chunk], return_dict=False)[0] for k in range(0, z.shape[0], chunk)], dim=0)

    def _vae_encode_chunked(x_in: torch.Tensor, chunk: int = 4) -> torch.Tensor:
        if x_in.shape[0] <= chunk:
            return pipe.vae.encode(x_in).latent_dist.mean
        return torch.cat([pipe.vae.encode(x_in[k : k + chunk]).latent_dist.mean for k in range(0, x_in.shape[0], chunk)], dim=0)

    with torch.no_grad():
        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompts,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        num_channels_latents = pipe.transformer.config.in_channels // 4
        latent_height = 2 * (int(size) // (pipe.vae_scale_factor * 2))
        latent_width = 2 * (int(size) // (pipe.vae_scale_factor * 2))
        pack_h, pack_w = latent_height // 2, latent_width // 2

        latents, latent_ids = pipe.prepare_latents(
            batch_size=op.num_views,
            num_latents_channels=num_channels_latents,
            height=size,
            width=size,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=gen,
            latents=None,
        )

        # For equivariant permutation illusions (flip), tie initial noise in unpatchified spatial space
        if op.is_permutation and not pixel_reconcile:
            spatial = pipe._unpack_latents_with_ids(latents[:op.num_primes], latent_ids[:op.num_primes], pack_h, pack_w)
            unpatch = pipe._unpatchify_latents(spatial)  # [P, 32, H//8, W//8]
            derived_unpatch = op.derive([unpatch[k] for k in range(op.num_primes)])
            latents = pipe._pack_latents(pipe._patchify_latents(derived_unpatch)).to(dtype)

        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=num_steps)
        timesteps, _ = retrieve_timesteps(pipe.scheduler, num_steps, device, sigmas=sigmas, mu=mu)
        sched_sigmas = pipe.scheduler.sigmas.to(device=device, dtype=torch.float32)

        # bn_mean / bn_std have shape [1, 128, 1, 1] -> applied in patchified (128-ch) space!
        bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(device, torch.float32)
        bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(device, torch.float32)

        def _packed_to_vae_latents(packed: torch.Tensor) -> torch.Tensor:
            sp = pipe._unpack_latents_with_ids(packed, latent_ids, pack_h, pack_w).float()
            sp = sp * bn_std + bn_mean
            return pipe._unpatchify_latents(sp)

        def _vae_latents_to_packed(vae_lat: torch.Tensor) -> torch.Tensor:
            patch = pipe._patchify_latents(vae_lat.float())
            patch_norm = (patch - bn_mean) / bn_std
            return pipe._pack_latents(patch_norm)

        warm_logits = None
        primes = None
        last_rec_01 = None
        match_info = None
        tx_chunk = 6 if (latents.shape[0] > 6 and size >= 576) else latents.shape[0]
        for i, t in enumerate(timesteps):
            if tx_chunk < latents.shape[0]:
                outs = []
                for s0 in range(0, latents.shape[0], tx_chunk):
                    s1 = s0 + tx_chunk
                    ts_chunk = t.expand(s1 - s0).to(latents.dtype)
                    outs.append(
                        pipe.transformer(
                            hidden_states=latents[s0:s1],
                            timestep=ts_chunk / 1000.0,
                            guidance=None,
                            encoder_hidden_states=prompt_embeds[s0:s1],
                            txt_ids=text_ids[s0:s1],
                            img_ids=latent_ids[s0:s1],
                            return_dict=False,
                        )[0][:, : latents.size(1)]
                    )
                noise_pred = torch.cat(outs, dim=0)
            else:
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                noise_pred = pipe.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000.0,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0][:, : latents.size(1)]

            sigma = sched_sigmas[i]
            sigma_next = sched_sigmas[i + 1]
            # Predict clean x0 in packed normalized space
            x0_packed = latents.float() - sigma * noise_pred.float()

            if op.is_permutation and not pixel_reconcile:
                # Reconcile in unpatchified VAE-latent space (valid for equivariant rot180 flip)
                x0_lat = _packed_to_vae_latents(x0_packed)
                x0_lat_rec = op.invert_and_average(x0_lat)
                rec_packed = _vae_latents_to_packed(x0_lat_rec)
                v_eff = ((latents.float() - rec_packed) / sigma.clamp(min=1e-4)).to(dtype)
                latents = pipe.scheduler.step(v_eff, t, latents, return_dict=False)[0]
            else:
                x0_lat = _packed_to_vae_latents(x0_packed).to(pipe.vae.dtype)
                imgs_01 = ((_vae_decode_chunked(x0_lat).float() + 1.0) * 0.5).clamp(0.0, 1.0)

                if i == 0 and isinstance(op, RubiksCubeOperator):
                    match_info = _reorder_rubiks_scrambled(
                        op, imgs_01, prompts, latents, prompt_embeds, text_ids, latent_ids
                    )

                if op.is_permutation:
                    rec_01 = op.invert_and_average(imgs_01)
                    primes = [rec_01[k] for k in range(op.num_primes)]
                elif isinstance(op, RotationOverlayOperator):
                    rec_01, primes, warm_logits = op.reconcile(
                        imgs_01, init_logits=warm_logits, steps=reconcile_steps
                    )
                else:
                    rec_01, primes = op.reconcile(imgs_01)
                last_rec_01 = rec_01

                if i + 1 < len(timesteps):
                    rec_in = (2.0 * rec_01 - 1.0).to(pipe.vae.dtype)
                    rec_lat = _vae_encode_chunked(rec_in)
                    rec_packed = _vae_latents_to_packed(rec_lat)
                    v_rec = (latents.float() - rec_packed) / sigma.clamp(min=1e-4)
                    latents = (latents.float() + (sigma_next - sigma) * v_rec).to(dtype)

        if op.is_permutation and not pixel_reconcile:
            x0_lat = _packed_to_vae_latents(latents).to(pipe.vae.dtype)
            imgs = _vae_decode_chunked(x0_lat)
            decoded_01 = ((imgs.float() + 1.0) * 0.5).clamp(0.0, 1.0)
            primes = [decoded_01[k] for k in range(op.num_primes)]
            views_01 = op.derive(primes)
        elif op.is_permutation:
            primes = [last_rec_01[k] for k in range(op.num_primes)]
            views_01 = op.derive(primes)
        else:
            views_01 = op.derive(primes)

    sync_torch_device(device)
    gen_s = time.perf_counter() - t0

    for idx, p in enumerate(primes):
        _tensor_to_pil(p).save(out_path / f"prime_{idx}.png")
    view_pils = [_tensor_to_pil(v) for v in views_01]
    for idx, v in enumerate(view_pils):
        v.save(out_path / f"view_{idx}.png")
    _make_grid(view_pils).save(out_path / "final_grid.png")

    res = {
        "method": "flux2_klein_zeroshot",
        "task": task,
        "steps": num_steps,
        "size": size,
        "pixel_reconcile": bool(pixel_reconcile),
        "load_s": round(load_s, 3),
        "gen_s": round(gen_s, 3),
        "out_dir": str(out_path),
    }
    if match_info:
        res.update(match_info)
    return res


# ---------------------------------------------------------------------------
# 4. SANA-Sprint 1.6B (2-4 steps Linear DiT + DC-AE, Apache 2.0)
# ---------------------------------------------------------------------------

def run_sana_sprint_zeroshot(
    task: str,
    prompts: list[str],
    model_id: str = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    num_steps: int = 2,
    size: int = 1024,
    out_dir: str = "/tmp/out_sana_sprint",
    seed: int = 42,
    reconcile_steps: int = 60,
    pixel_reconcile: bool | None = None,
) -> dict:
    """
    2-4 step SANA-Sprint (Linear DiT + DC-AE) zero-shot multi-view / reconciled sampling.
    """
    from diffusers import SanaSprintPipeline
    from diffusers.pipelines.sana.pipeline_sana_sprint import retrieve_timesteps

    op = _get_operator(task)
    assert len(prompts) == op.num_views
    if pixel_reconcile is None:
        pixel_reconcile = (not op.is_permutation) or (not op.equivariant_to_latent)

    device = get_best_torch_device()
    dtype = _preferred_dtype(device, prefer_bf16=True)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t_load0 = time.perf_counter()
    pipe = SanaSprintPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    pipe.vae.to(dtype=torch.float32)  # Prevent DC-AE bfloat16 NaN overflow
    sync_torch_device(device)
    load_s = time.perf_counter() - t_load0

    sync_torch_device(device)
    t0 = time.perf_counter()
    gen = torch.Generator(device=device).manual_seed(seed)

    def _vae_decode_chunked(z: torch.Tensor, chunk: int = 4) -> torch.Tensor:
        if z.shape[0] <= chunk:
            return pipe.vae.decode(z, return_dict=False)[0]
        return torch.cat([pipe.vae.decode(z[k : k + chunk], return_dict=False)[0] for k in range(0, z.shape[0], chunk)], dim=0)

    def _vae_encode_chunked(x_in: torch.Tensor, chunk: int = 4) -> torch.Tensor:
        if x_in.shape[0] <= chunk:
            return pipe.vae.encode(x_in).latent
        return torch.cat([pipe.vae.encode(x_in[k : k + chunk]).latent for k in range(0, x_in.shape[0], chunk)], dim=0)

    with torch.no_grad():
        prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
            prompts,
            num_images_per_prompt=1,
            device=device,
            clean_caption=False,
            max_sequence_length=300,
        )
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler,
            num_steps,
            device,
            timesteps=None,
            sigmas=None,
            max_timesteps=1.57080,
            intermediate_timesteps=1.3 if num_steps == 2 else None,
        )
        if hasattr(pipe.scheduler, "set_begin_index"):
            pipe.scheduler.set_begin_index(0)

        latent_channels = pipe.transformer.config.in_channels
        latents = pipe.prepare_latents(
            op.num_views, latent_channels, size, size, torch.float32, device, gen, None
        )
        if op.is_permutation and not pixel_reconcile:
            latents = op.derive([latents[k] for k in range(op.num_primes)])
        latents = latents * pipe.scheduler.config.sigma_data

        guidance = torch.full([op.num_views], 4.5, device=device, dtype=torch.float32).to(prompt_embeds.dtype)
        guidance = guidance * pipe.transformer.config.guidance_embeds_scale

        timesteps_loop = timesteps[:-1]
        warm_logits = None
        primes = None
        denoised = None
        last_rec_01 = None
        match_info = None
        sigma_data = pipe.scheduler.config.sigma_data
        tx_chunk = 6 if (latents.shape[0] > 6 and size >= 768) else latents.shape[0]

        for i, t in enumerate(timesteps_loop):
            timestep = t.expand(latents.shape[0])
            latents_model_input = latents / sigma_data
            scm_t = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
            scm_exp = scm_t.view(-1, 1, 1, 1)
            latent_in = latents_model_input * torch.sqrt(scm_exp ** 2 + (1.0 - scm_exp) ** 2)

            if tx_chunk < latents.shape[0]:
                outs = []
                for s0 in range(0, latents.shape[0], tx_chunk):
                    s1 = s0 + tx_chunk
                    outs.append(
                        pipe.transformer(
                            latent_in[s0:s1].to(dtype=dtype),
                            encoder_hidden_states=prompt_embeds[s0:s1].to(dtype=dtype),
                            encoder_attention_mask=prompt_attention_mask[s0:s1],
                            guidance=guidance[s0:s1],
                            timestep=scm_t[s0:s1],
                            return_dict=False,
                        )[0].float()
                    )
                noise_pred = torch.cat(outs, dim=0)
            else:
                noise_pred = pipe.transformer(
                    latent_in.to(dtype=dtype),
                    encoder_hidden_states=prompt_embeds.to(dtype=dtype),
                    encoder_attention_mask=prompt_attention_mask,
                    guidance=guidance,
                    timestep=scm_t,
                    return_dict=False,
                )[0].float()
            noise_pred = (
                (1.0 - 2.0 * scm_exp) * latent_in
                + (1.0 - 2.0 * scm_exp + 2.0 * scm_exp ** 2) * noise_pred
            ) / torch.sqrt(scm_exp ** 2 + (1.0 - scm_exp) ** 2)
            noise_pred = noise_pred * sigma_data

            if op.is_permutation and not pixel_reconcile:
                noise_pred = op.invert_velocities(noise_pred)
                latents, denoised = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)
            else:
                _, denoised_step = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)
                x0_lat = (denoised_step / sigma_data).to(pipe.vae.dtype)
                imgs_01 = ((_vae_decode_chunked(x0_lat / pipe.vae.config.scaling_factor).float() + 1.0) * 0.5).clamp(0.0, 1.0)
                if i == 0 and isinstance(op, RubiksCubeOperator):
                    match_info = _reorder_rubiks_scrambled(
                        op, imgs_01, prompts, latents, prompt_embeds, prompt_attention_mask
                    )
                if op.is_permutation:
                    rec_01 = op.invert_and_average(imgs_01)
                    primes = [rec_01[k] for k in range(op.num_primes)]
                elif isinstance(op, RotationOverlayOperator):
                    rec_01, primes, warm_logits = op.reconcile(
                        imgs_01, init_logits=warm_logits, steps=reconcile_steps
                    )
                else:
                    rec_01, primes = op.reconcile(imgs_01)
                last_rec_01 = rec_01
                if i + 1 < len(timesteps_loop):
                    rec_lat = _vae_encode_chunked((2.0 * rec_01 - 1.0).to(pipe.vae.dtype)) * pipe.vae.config.scaling_factor
                    rec_denoised = rec_lat.float() * sigma_data
                    t_next = timesteps_loop[i + 1]
                    noise_est = torch.randn_like(rec_denoised)
                    latents = math.cos(float(t_next)) * rec_denoised + math.sin(float(t_next)) * noise_est * sigma_data
                    denoised = rec_denoised

        if op.is_permutation and not pixel_reconcile:
            x0_lat = (denoised / sigma_data).to(pipe.vae.dtype)
            imgs = _vae_decode_chunked(x0_lat / pipe.vae.config.scaling_factor)
            decoded_01 = ((imgs.float() + 1.0) * 0.5).clamp(0.0, 1.0)
            primes = [decoded_01[k] for k in range(op.num_primes)]
            views_01 = op.derive(primes)
        elif op.is_permutation:
            primes = [last_rec_01[k] for k in range(op.num_primes)]
            views_01 = op.derive(primes)
        else:
            views_01 = op.derive(primes)

    sync_torch_device(device)
    gen_s = time.perf_counter() - t0

    for idx, p in enumerate(primes):
        _tensor_to_pil(p).save(out_path / f"prime_{idx}.png")
    view_pils = [_tensor_to_pil(v) for v in views_01]
    for idx, v in enumerate(view_pils):
        v.save(out_path / f"view_{idx}.png")
    _make_grid(view_pils).save(out_path / "final_grid.png")

    res = {
        "method": "sana_sprint_zeroshot",
        "task": task,
        "steps": num_steps,
        "size": size,
        "pixel_reconcile": bool(pixel_reconcile),
        "load_s": round(load_s, 3),
        "gen_s": round(gen_s, 3),
        "out_dir": str(out_path),
    }
    if match_info:
        res.update(match_info)
    return res
