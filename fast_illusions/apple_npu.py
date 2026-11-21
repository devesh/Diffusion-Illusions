"""Apple Neural Engine (ANE / Apple NPU) Core ML export and accelerated zero-shot illusion runner.

PyTorch's ``mps`` backend runs only on the Apple GPU, not on the Apple Neural Engine (NPU).
This module provides:
  1. ANE-optimized PyTorch attention rewriting (``ANEAttentionProcessor``: 4D ``NCHW`` / ``BC1S``
     1x1 Conv2d QKV/out projections + head-split explicit matmul/softmax without SDPA).
  2. One-command Core ML ``mlprogram`` (``.mlpackage``) exporter for CFG-free few-step models
     (``sd_turbo``, ``sdxl_turbo``, ``sana_sprint``), with FP16 compute precision and optional
     6-bit / 8-bit k-means weight palettization.
  3. On-device ``.mlmodelc`` compilation caching and Core ML ``CPU_AND_NE`` (Apple NPU)
     execution for all 5 illusion tasks (``flip``, ``twisting_squares``, ``rubiks_cube``,
     ``rotation_overlay``, ``hidden_overlay``), plus drop-in compatibility with external
     ``apple/ml-stable-diffusion`` Core ML bundles.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fast_illusions.operators import (
    FlipOperator,
    TwistingSquaresOperator,
    RubiksCubeOperator,
    RotationOverlayOperator,
    HiddenOverlayOperator,
)


# ---------------------------------------------------------------------------
# Device & Hardware Capability Helpers
# ---------------------------------------------------------------------------

def get_best_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_torch_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def apple_npu_info() -> dict[str, Any]:
    """Return host Apple Silicon / Core ML / Apple Neural Engine capability summary."""
    sys_name = platform.system()
    machine = platform.machine()
    mac_ver_str = platform.mac_ver()[0] if sys_name == "Darwin" else ""
    mac_major = 0
    if mac_ver_str:
        try:
            mac_major = int(mac_ver_str.split(".")[0])
        except ValueError:
            mac_major = 0

    try:
        import coremltools as ct
        ct_ver = getattr(ct, "__version__", "unknown")
    except Exception:
        ct_ver = None

    is_apple_silicon = (sys_name == "Darwin") and (machine in ("arm64", "arm64e"))
    ane_supported = is_apple_silicon and (mac_major == 0 or mac_major >= 13) and (ct_ver is not None)
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    return {
        "system": sys_name,
        "machine": machine,
        "macos_version": mac_ver_str or None,
        "is_apple_silicon": is_apple_silicon,
        "mps_available": mps_available,
        "coremltools_version": ct_ver,
        "ane_runtime_supported": ane_supported,
        "ane_export_supported": ct_ver is not None,
        "recommended_compute_unit": "CPU_AND_NE" if ane_supported else ("MPS_GPU" if mps_available else "CPU"),
    }


def _resolve_compute_unit(compute_units: str):
    import coremltools as ct

    key = compute_units.upper()
    mapping = {
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ANE": ct.ComputeUnit.CPU_AND_NE,
        "NPU": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported compute_units={compute_units!r}; choose from {sorted(mapping)}")
    return mapping[key]


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


def _reorder_rubiks_scrambled_npu(
    op,
    imgs: torch.Tensor,
    prompts: list[str],
    latents: torch.Tensor,
    *np_arrays: np.ndarray | None,
) -> dict | None:
    if not isinstance(op, RubiksCubeOperator):
        return None
    perm, mse_before, mse_after = op.match_scrambled_faces(imgs)
    idx_cpu = torch.tensor(perm, dtype=torch.long)
    imgs[6:12] = imgs[6:12][idx_cpu.to(imgs.device)]
    latents[6:12] = latents[6:12][idx_cpu.to(latents.device)]
    for arr in np_arrays:
        if arr is not None and arr.shape[0] >= 12:
            arr[6:12] = arr[6:12][perm]
    return {
        "scrambled_perm": perm,
        "scrambled_face_prompts": {fn: prompts[6 + p] for fn, p in zip(op.FACE_NAMES, perm)},
        "match_mse_before": round(mse_before, 5),
        "match_mse_after": round(mse_after, 5),
    }


# ---------------------------------------------------------------------------
# ANE-Friendly Attention Rewriting (BC1S Conv2d + Head-Split Matmul)
# ---------------------------------------------------------------------------

class ANEAttentionProcessor:
    """
    Apple Neural Engine (ANE) friendly attention processor for Diffusers ``Attention`` modules:
      - Runs Q, K, V, and Out projections as 4D ``NCHW`` 1x1 ``F.conv2d`` over ``(B, C, 1, S)``
        (eliminates transpose overhead and matches ANE SRAM DMA alignment).
      - Replaces ``F.scaled_dot_product_attention`` with head-split rank-4 ``torch.matmul`` +
        ``torch.softmax`` (``SPLIT_EINSUM`` pattern) so Core ML dispatches attention cleanly to ANE.
      - Uses existing ``attn.to_q / to_k / to_v / to_out`` weights directly without mutating keys.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if attention_mask is not None:
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, 1, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Reshape sequence to ANE-friendly 4D channels-first [B, C, 1, S]
        hs_4d = hidden_states.transpose(1, 2).unsqueeze(2)
        if encoder_hidden_states is None:
            enc_4d = hs_4d
        else:
            if attn.norm_cross is not None:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            enc_4d = encoder_hidden_states.transpose(1, 2).unsqueeze(2)

        q_4d = F.conv2d(hs_4d, attn.to_q.weight[:, :, None, None], attn.to_q.bias)
        k_4d = F.conv2d(enc_4d, attn.to_k.weight[:, :, None, None], attn.to_k.bias)
        v_4d = F.conv2d(enc_4d, attn.to_v.weight[:, :, None, None], attn.to_v.bias)

        head_dim = int(attn.to_q.out_features // attn.heads)
        q_heads = q_4d.split(head_dim, dim=1)
        k_heads = k_4d.split(head_dim, dim=1)
        v_heads = v_4d.split(head_dim, dim=1)

        out_heads = []
        for h_idx, (qh, kh, vh) in enumerate(zip(q_heads, k_heads, v_heads)):
            # Keep every tensor rank <= 4 for ANE scheduling
            q_mat = qh.permute(0, 2, 3, 1)  # [B, 1, S_q, D]
            k_mat = kh.permute(0, 2, 1, 3)  # [B, 1, D, S_k]
            v_mat = vh.permute(0, 2, 3, 1)  # [B, 1, S_k, D]

            scores = torch.matmul(q_mat, k_mat) * attn.scale  # [B, 1, S_q, S_k]
            if attention_mask is not None:
                scores = scores + attention_mask[:, h_idx : h_idx + 1, :, :]
            probs = torch.softmax(scores, dim=-1)
            attn_h = torch.matmul(probs, v_mat)  # [B, 1, S_q, D]
            out_heads.append(attn_h.permute(0, 3, 1, 2))  # [B, D, 1, S_q]

        out_4d = torch.cat(out_heads, dim=1)  # [B, H*D, 1, S_q]
        out_proj = attn.to_out[0]
        out_4d = F.conv2d(out_4d, out_proj.weight[:, :, None, None], out_proj.bias)
        hidden_states = out_4d.squeeze(2).transpose(1, 2)
        if len(attn.to_out) > 1:
            hidden_states = attn.to_out[1](hidden_states)  # Dropout (no-op in eval)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def enable_ane_attention(module: nn.Module) -> int:
    """Install ``ANEAttentionProcessor`` on all compatible Diffusers attention blocks.

    Returns the number of attention layers rewritten.
    """
    count = 0
    for sub in module.modules():
        if hasattr(sub, "set_processor") and hasattr(sub, "to_q") and hasattr(sub, "to_out"):
            # Skip specialized processors (e.g. SANA linear attention) that lack standard softmax scaling
            proc_name = type(getattr(sub, "processor", None)).__name__
            if "Sana" in proc_name or getattr(sub, "is_cross_attention", False) and not hasattr(sub, "scale"):
                continue
            if hasattr(sub, "to_k") and hasattr(sub, "to_v") and hasattr(sub, "heads") and hasattr(sub, "scale"):
                sub.set_processor(ANEAttentionProcessor())
                count += 1
    return count


# ---------------------------------------------------------------------------
# Traceable Wrappers for Core ML Export
# ---------------------------------------------------------------------------

class _TextEncoderWrapper(nn.Module):
    def __init__(self, text_encoder: nn.Module):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.text_encoder(input_ids, return_dict=False)
        return out[0]


class _SDTurboUNetWrapper(nn.Module):
    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


class _SDXLTurboUNetWrapper(nn.Module):
    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        text_embeds: torch.Tensor,
        time_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
        )[0]


class _SanaTransformerWrapper(nn.Module):
    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        guidance: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            guidance=guidance,
            timestep=timestep,
            return_dict=False,
        )[0]


class _VAEDecoderWrapper(nn.Module):
    """Expects scaled latent ``z`` (matching Apple ml-stable-diffusion signature) -> image in [-1, 1]."""

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z, return_dict=False)[0]


class _VAEEncoderWrapper(nn.Module):
    """Encodes image in [-1, 1] -> unscaled latent mean (or DC-AE latent)."""

    def __init__(self, vae: nn.Module, is_dcae: bool = False):
        super().__init__()
        self.vae = vae
        self.is_dcae = is_dcae

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        out = self.vae.encode(image)
        if self.is_dcae:
            return out.latent
        return out.latent_dist.mean


# ---------------------------------------------------------------------------
# Core ML Conversion & Post-Training Weight Palettization
# ---------------------------------------------------------------------------

def _apply_palettization(mlmodel, nbits: int | None):
    if nbits is None or nbits >= 16:
        return mlmodel
    import coremltools.optimize as cto

    # Choose k-means if scikit-learn is available, else fall back cleanly to uniform LUTs
    try:
        import sklearn  # noqa: F401
        mode = "kmeans"
    except Exception:
        mode = "uniform"

    op_name_configs = {}
    try:
        meta = cto.coreml.get_weights_metadata(mlmodel, weight_threshold=2048)
        if meta:
            last_weight = list(meta.keys())[-1]
            op_name_configs[last_weight] = None  # Preserve final output layer precision
    except Exception:
        pass

    op_cfg = cto.coreml.OpPalettizerConfig(
        mode=mode,
        nbits=int(nbits),
        weight_threshold=512,
    )
    opt_cfg = cto.coreml.OptimizationConfig(
        global_config=op_cfg,
        op_name_configs=op_name_configs,
    )
    return cto.coreml.palettize_weights(mlmodel, opt_cfg)


def _ensure_numpy2_scalar_cast_compat() -> None:
    """Fix coremltools TorchScript `_cast` op failing under NumPy >= 2.0 on 1-element arrays."""
    try:
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.frontend.torch import ops as torch_ops
        from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    except Exception:
        return
    if getattr(torch_ops, "_ambigram_cast_patched", False):
        return
    orig_cast = getattr(torch_ops, "_cast", None)
    if orig_cast is None:
        return

    def _safe_cast(context, node, dtype, dtype_name):
        try:
            return orig_cast(context, node, dtype, dtype_name)
        except TypeError:
            inputs = _get_inputs(context, node, expected=1)
            x = inputs[0]
            if x.val is not None:
                arr = np.asarray(x.val)
                if arr.size == 1:
                    res = mb.const(val=dtype(arr.reshape(()).item()), name=node.name)
                    context.add(res)
                    return
            raise

    torch_ops._cast = _safe_cast
    torch_ops._ambigram_cast_patched = True


def _trace_and_convert(
    module: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    ct_inputs: list,
    ct_outputs: list,
    out_path: Path,
    quantize_nbits: int | None = None,
    fp16: bool = True,
) -> Path:
    import coremltools as ct

    _ensure_numpy2_scalar_cast_compat()
    module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, example_inputs, check_trace=False)

    precision = ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=ct_inputs,
        outputs=ct_outputs,
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS14 if (quantize_nbits and quantize_nbits < 16) else ct.target.macOS13,
    )
    if quantize_nbits is not None and quantize_nbits < 16:
        mlmodel = _apply_palettization(mlmodel, quantize_nbits)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    return out_path


def export_coreml_models(
    model_key: str = "sd_turbo",
    out_dir: str | Path = "./coreml_models/sd_turbo_512",
    size: int | None = None,
    quantize_nbits: int | None = 6,
    ane_attention: bool = True,
    export_text_encoder: bool = True,
    verify_equivalence: bool = True,
) -> dict[str, Any]:
    """
    Export CFG-free few-step Diffusers models to ANE-ready Core ML ``.mlpackage`` bundles
    (with FP16 precision and optional 6-bit/8-bit weight palettization).

    Supported ``model_key`` values:
      - ``"sd_turbo"`` (stabilityai/sd-turbo, default size=512; use 480 for rubiks_cube)
      - ``"sdxl_turbo"`` (stabilityai/sdxl-turbo + fp16-safe VAE, default size=512; use 576 for rubiks_cube)
      - ``"sana_sprint"`` (Sana_Sprint_1.6B_1024px_diffusers Linear DiT + DC-AE, default size=1024; 768 for rubiks_cube)
    """
    import coremltools as ct  # noqa: F401
    from diffusers import AutoPipelineForText2Image, AutoencoderKL

    t0 = time.perf_counter()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if model_key == "sd_turbo":
        model_id = "stabilityai/sd-turbo"
        size = size or 512
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False
        )
        lat_h = lat_w = size // 8
        latent_ch = 4
        seq_len = pipe.text_encoder.config.max_position_embeddings
        txt_dim = pipe.text_encoder.config.hidden_size
        scaling_factor = float(pipe.vae.config.scaling_factor)

        unet_raw = pipe.unet.eval()
        sample = torch.randn(1, latent_ch, lat_h, lat_w)
        ts = torch.tensor([999.0], dtype=torch.float32)
        enc = torch.randn(1, seq_len, txt_dim)

        if verify_equivalence:
            with torch.no_grad():
                ref_out = unet_raw(sample, ts, encoder_hidden_states=enc, return_dict=False)[0]

        rewritten = enable_ane_attention(unet_raw) if ane_attention else 0
        if verify_equivalence and rewritten > 0:
            with torch.no_grad():
                ane_out = unet_raw(sample, ts, encoder_hidden_states=enc, return_dict=False)[0]
                max_diff = (ref_out - ane_out).abs().max().item()
                if max_diff > 1e-3:
                    raise RuntimeError(f"ANE attention equivalence check failed: max_diff={max_diff}")

        _trace_and_convert(
            _SDTurboUNetWrapper(unet_raw),
            (sample, ts, enc),
            [
                ct.TensorType(name="sample", shape=(1, latent_ch, lat_h, lat_w), dtype=np.float16),
                ct.TensorType(name="timestep", shape=(1,), dtype=np.float16),
                ct.TensorType(name="encoder_hidden_states", shape=(1, seq_len, txt_dim), dtype=np.float16),
            ],
            [ct.TensorType(name="noise_pred", dtype=np.float16)],
            out_path / "unet.mlpackage",
            quantize_nbits=quantize_nbits,
            fp16=True,
        )

        # VAE Decoder + Encoder (crucial for pixel-reconciled illusions on ANE)
        _trace_and_convert(
            _VAEDecoderWrapper(pipe.vae.eval()),
            (sample,),
            [ct.TensorType(name="z", shape=(1, latent_ch, lat_h, lat_w), dtype=np.float16)],
            [ct.TensorType(name="image", dtype=np.float16)],
            out_path / "vae_decoder.mlpackage",
            quantize_nbits=None,  # Keep VAE FP16 uncompressed for visual fidelity
            fp16=True,
        )
        img_ex = torch.randn(1, 3, size, size)
        _trace_and_convert(
            _VAEEncoderWrapper(pipe.vae.eval(), is_dcae=False),
            (img_ex,),
            [ct.TensorType(name="image", shape=(1, 3, size, size), dtype=np.float16)],
            [ct.TensorType(name="latent", dtype=np.float16)],
            out_path / "vae_encoder.mlpackage",
            quantize_nbits=None,
            fp16=True,
        )

        if export_text_encoder:
            input_ids = torch.zeros((1, seq_len), dtype=torch.int32)
            _trace_and_convert(
                _TextEncoderWrapper(pipe.text_encoder.eval()),
                (input_ids,),
                [ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32)],
                [ct.TensorType(name="last_hidden_state", dtype=np.float16)],
                out_path / "text_encoder.mlpackage",
                quantize_nbits=quantize_nbits,
                fp16=True,
            )

        meta = {
            "model_key": model_key,
            "model_id": model_id,
            "size": size,
            "latent_h": lat_h,
            "latent_w": lat_w,
            "latent_channels": latent_ch,
            "seq_len": seq_len,
            "scaling_factor": scaling_factor,
            "ane_attention": ane_attention,
            "rewritten_attn_layers": rewritten,
            "quantize_nbits": quantize_nbits,
            "has_text_encoder_coreml": export_text_encoder,
            "has_vae_encoder_coreml": True,
        }

    elif model_key == "sdxl_turbo":
        model_id = "stabilityai/sdxl-turbo"
        size = size or 512
        vae_fp16 = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32)
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, vae=vae_fp16, torch_dtype=torch.float32
        )
        lat_h = lat_w = size // 8
        latent_ch = 4
        seq_len = pipe.text_encoder.config.max_position_embeddings
        txt_dim = pipe.unet.config.cross_attention_dim
        pooled_dim = pipe.text_encoder_2.config.projection_dim
        scaling_factor = float(pipe.vae.config.scaling_factor)

        unet_raw = pipe.unet.eval()
        sample = torch.randn(1, latent_ch, lat_h, lat_w)
        ts = torch.tensor([999.0], dtype=torch.float32)
        enc = torch.randn(1, seq_len, txt_dim)
        text_embeds = torch.randn(1, pooled_dim)
        time_ids = torch.zeros(1, 6, dtype=torch.float32)

        if verify_equivalence:
            with torch.no_grad():
                ref_out = unet_raw(
                    sample, ts, encoder_hidden_states=enc,
                    added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids}, return_dict=False
                )[0]
        rewritten = enable_ane_attention(unet_raw) if ane_attention else 0
        if verify_equivalence and rewritten > 0:
            with torch.no_grad():
                ane_out = unet_raw(
                    sample, ts, encoder_hidden_states=enc,
                    added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids}, return_dict=False
                )[0]
                max_diff = (ref_out - ane_out).abs().max().item()
                if max_diff > 1e-3:
                    raise RuntimeError(f"SDXL ANE attention equivalence check failed: max_diff={max_diff}")

        _trace_and_convert(
            _SDXLTurboUNetWrapper(unet_raw),
            (sample, ts, enc, text_embeds, time_ids),
            [
                ct.TensorType(name="sample", shape=(1, latent_ch, lat_h, lat_w), dtype=np.float16),
                ct.TensorType(name="timestep", shape=(1,), dtype=np.float16),
                ct.TensorType(name="encoder_hidden_states", shape=(1, seq_len, txt_dim), dtype=np.float16),
                ct.TensorType(name="text_embeds", shape=(1, pooled_dim), dtype=np.float16),
                ct.TensorType(name="time_ids", shape=(1, 6), dtype=np.float16),
            ],
            [ct.TensorType(name="noise_pred", dtype=np.float16)],
            out_path / "unet.mlpackage",
            quantize_nbits=quantize_nbits,
            fp16=True,
        )
        _trace_and_convert(
            _VAEDecoderWrapper(pipe.vae.eval()),
            (sample,),
            [ct.TensorType(name="z", shape=(1, latent_ch, lat_h, lat_w), dtype=np.float16)],
            [ct.TensorType(name="image", dtype=np.float16)],
            out_path / "vae_decoder.mlpackage",
            quantize_nbits=None,
            fp16=True,
        )
        img_ex = torch.randn(1, 3, size, size)
        _trace_and_convert(
            _VAEEncoderWrapper(pipe.vae.eval(), is_dcae=False),
            (img_ex,),
            [ct.TensorType(name="image", shape=(1, 3, size, size), dtype=np.float16)],
            [ct.TensorType(name="latent", dtype=np.float16)],
            out_path / "vae_encoder.mlpackage",
            quantize_nbits=None,
            fp16=True,
        )

        meta = {
            "model_key": model_key,
            "model_id": model_id,
            "vae_id": "madebyollin/sdxl-vae-fp16-fix",
            "size": size,
            "latent_h": lat_h,
            "latent_w": lat_w,
            "latent_channels": latent_ch,
            "seq_len": seq_len,
            "scaling_factor": scaling_factor,
            "ane_attention": ane_attention,
            "rewritten_attn_layers": rewritten,
            "quantize_nbits": quantize_nbits,
            "has_text_encoder_coreml": False,
            "has_vae_encoder_coreml": True,
        }

    elif model_key == "sana_sprint":
        from diffusers import SanaSprintPipeline

        model_id = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"
        size = size or 1024
        pipe = SanaSprintPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        lat_h = lat_w = size // pipe.vae_scale_factor
        latent_ch = pipe.transformer.config.in_channels
        seq_len = 300
        cap_dim = pipe.transformer.config.caption_channels
        scaling_factor = float(pipe.vae.config.scaling_factor)
        sigma_data = float(pipe.scheduler.config.sigma_data)

        hidden_states = torch.randn(1, latent_ch, lat_h, lat_w)
        ts = torch.tensor([0.99], dtype=torch.float32)
        enc = torch.randn(1, seq_len, cap_dim)
        enc_mask = torch.ones(1, seq_len, dtype=torch.float32)
        guidance = torch.tensor([4.5 * pipe.transformer.config.guidance_embeds_scale], dtype=torch.float32)

        _trace_and_convert(
            _SanaTransformerWrapper(pipe.transformer.eval()),
            (hidden_states, ts, enc, enc_mask, guidance),
            [
                ct.TensorType(name="hidden_states", shape=(1, latent_ch, lat_h, lat_w), dtype=np.float16),
                ct.TensorType(name="timestep", shape=(1,), dtype=np.float16),
                ct.TensorType(name="encoder_hidden_states", shape=(1, seq_len, cap_dim), dtype=np.float16),
                ct.TensorType(name="encoder_attention_mask", shape=(1, seq_len), dtype=np.float16),
                ct.TensorType(name="guidance", shape=(1,), dtype=np.float16),
            ],
            [ct.TensorType(name="noise_pred", dtype=np.float16)],
            out_path / "unet.mlpackage",
            quantize_nbits=quantize_nbits,
            fp16=True,
        )
        _trace_and_convert(
            _VAEDecoderWrapper(pipe.vae.eval()),
            (hidden_states,),
            [ct.TensorType(name="z", shape=(1, latent_ch, lat_h, lat_w), dtype=np.float16)],
            [ct.TensorType(name="image", dtype=np.float16)],
            out_path / "vae_decoder.mlpackage",
            quantize_nbits=None,
            fp16=False,  # DC-AE requires fp32 precision to avoid activation overflow
        )
        img_ex = torch.randn(1, 3, size, size)
        _trace_and_convert(
            _VAEEncoderWrapper(pipe.vae.eval(), is_dcae=True),
            (img_ex,),
            [ct.TensorType(name="image", shape=(1, 3, size, size), dtype=np.float16)],
            [ct.TensorType(name="latent", dtype=np.float16)],
            out_path / "vae_encoder.mlpackage",
            quantize_nbits=None,
            fp16=False,
        )
        meta = {
            "model_key": model_key,
            "model_id": model_id,
            "size": size,
            "latent_h": lat_h,
            "latent_w": lat_w,
            "latent_channels": latent_ch,
            "seq_len": seq_len,
            "scaling_factor": scaling_factor,
            "sigma_data": sigma_data,
            "ane_attention": False,
            "rewritten_attn_layers": 0,
            "quantize_nbits": quantize_nbits,
            "has_text_encoder_coreml": False,
            "has_vae_encoder_coreml": True,
        }
    else:
        raise ValueError(f"Unsupported Apple NPU Core ML model_key: {model_key!r}")

    meta["export_s"] = round(time.perf_counter() - t0, 3)
    (out_path / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------------------
# Compiled Model Loader & Cache (.mlpackage -> .mlmodelc)
# ---------------------------------------------------------------------------

def _load_coreml_artifact(model_path: Path, compute_units: str = "CPU_AND_NE", cache_compiled: bool = True):
    """Load an ``.mlmodelc`` or ``.mlpackage``, compiling and caching ``.mlmodelc`` on macOS."""
    import coremltools as ct
    from coremltools.models import CompiledMLModel
    from coremltools.models.utils import compile_model

    cu = _resolve_compute_unit(compute_units)
    if model_path.suffix == ".mlmodelc" and model_path.exists():
        return CompiledMLModel(str(model_path), compute_units=cu)

    compiled_dir = model_path.with_suffix(".mlmodelc")
    if cache_compiled and compiled_dir.exists():
        return CompiledMLModel(str(compiled_dir), compute_units=cu)

    if cache_compiled:
        compile_model(str(model_path), str(compiled_dir))
        return CompiledMLModel(str(compiled_dir), compute_units=cu)

    return ct.models.MLModel(str(model_path), compute_units=cu)


class _CoreMLBundle:
    """Thin wrapper unifying our exported bundles and external apple/ml-stable-diffusion bundles."""

    def __init__(
        self,
        coreml_dir: Path,
        compute_units: str = "CPU_AND_NE",
        cache_compiled: bool = True,
    ):
        self.coreml_dir = coreml_dir
        self.compute_units = compute_units
        self.meta: dict[str, Any] = {}
        meta_file = coreml_dir / "meta.json"
        if meta_file.exists():
            self.meta = json.loads(meta_file.read_text())

        self.unet = None
        self.unet_chunks: list[Any] = []
        self.vae_decoder = None
        self.vae_encoder = None
        self.text_encoder = None

        def _find(names: tuple[str, ...]) -> Path | None:
            for nm in names:
                for ext in (".mlmodelc", ".mlpackage"):
                    cand = coreml_dir / f"{nm}{ext}"
                    if cand.exists():
                        return cand
            return None

        u_path = _find(("unet", "Unet", "Stable_diffusion_unet"))
        if u_path is not None:
            self.unet = _load_coreml_artifact(u_path, compute_units, cache_compiled)
        else:
            c1 = _find(("UnetChunk1", "unet_chunk1"))
            c2 = _find(("UnetChunk2", "unet_chunk2"))
            if c1 is not None and c2 is not None:
                self.unet_chunks = [
                    _load_coreml_artifact(c1, compute_units, cache_compiled),
                    _load_coreml_artifact(c2, compute_units, cache_compiled),
                ]
            else:
                raise FileNotFoundError(f"No UNet/Transformer Core ML model found in {coreml_dir}")

        vd_path = _find(("vae_decoder", "VAEDecoder"))
        if vd_path is None:
            raise FileNotFoundError(f"No VAE Decoder Core ML model found in {coreml_dir}")
        self.vae_decoder = _load_coreml_artifact(vd_path, compute_units, cache_compiled)

        ve_path = _find(("vae_encoder", "VAEEncoder"))
        if ve_path is not None:
            self.vae_encoder = _load_coreml_artifact(ve_path, compute_units, cache_compiled)

        te_path = _find(("text_encoder", "TextEncoder"))
        if te_path is not None:
            self.text_encoder = _load_coreml_artifact(te_path, compute_units, cache_compiled)

        self._unet_enc_rank: int | None = None
        first_unet = self.unet or (self.unet_chunks[0] if self.unet_chunks else None)
        if first_unet is not None:
            try:
                for inp in first_unet.get_spec().description.input:
                    if inp.name == "encoder_hidden_states" and inp.type.HasField("multiArrayType"):
                        self._unet_enc_rank = len(inp.type.multiArrayType.shape)
                        break
            except Exception:
                pass

    def predict_unet(self, feed: dict[str, np.ndarray]) -> np.ndarray:
        if self._unet_enc_rank == 4 and "encoder_hidden_states" in feed:
            enc = feed["encoder_hidden_states"]
            if enc.ndim == 3:
                feed = dict(feed)
                # [B, S, C] -> [B, C, 1, S] (ml-stable-diffusion UNet layout)
                feed["encoder_hidden_states"] = np.transpose(enc, (0, 2, 1))[:, :, None, :]
        if self.unet is not None:
            out = self.unet.predict(feed)
            return out["noise_pred"] if "noise_pred" in out else next(iter(out.values()))
        # Two-chunk UNet support (e.g. ml-stable-diffusion --chunk-unet)
        out1 = self.unet_chunks[0].predict(feed)
        feed2 = dict(feed)
        feed2.update(out1)
        out2 = self.unet_chunks[1].predict(feed2)
        return out2["noise_pred"] if "noise_pred" in out2 else next(iter(out2.values()))

    def decode_vae(self, z: np.ndarray) -> np.ndarray:
        out = self.vae_decoder.predict({"z": z})
        return out["image"] if "image" in out else next(iter(out.values()))

    def encode_vae(self, image: np.ndarray) -> np.ndarray | None:
        if self.vae_encoder is None:
            return None
        key = "image"
        try:
            in_names = [i.name for i in self.vae_encoder.get_spec().description.input]
            if "x" in in_names:
                key = "x"
        except Exception:
            pass
        out = self.vae_encoder.predict({key: image})
        return out["latent"] if "latent" in out else next(iter(out.values()))

    def encode_text(self, input_ids: np.ndarray) -> np.ndarray | None:
        if self.text_encoder is None:
            return None
        out = self.text_encoder.predict({"input_ids": input_ids})
        return out["last_hidden_state"] if "last_hidden_state" in out else next(iter(out.values()))


# ---------------------------------------------------------------------------
# Zero-Shot Multi-View Illusion Runner on Apple NPU (Core ML CPU_AND_NE)
# ---------------------------------------------------------------------------

def run_apple_npu_zeroshot(
    task: str,
    prompts: list[str],
    model_key: str = "sd_turbo",
    coreml_dir: str | Path | None = None,
    compute_units: str = "CPU_AND_NE",
    num_steps: int = 4,
    size: int | None = None,
    quantize_nbits: int | None = 6,
    auto_export: bool = True,
    cache_compiled: bool = True,
    out_dir: str = "/tmp/out_apple_npu",
    seed: int = 42,
    reconcile_steps: int = 50,
    pixel_reconcile: bool | None = None,
    _backend: Any = None,
) -> dict[str, Any]:
    """
    Run zero-shot few-step multi-view illusion generation accelerated on the Apple Neural Engine (NPU).

    Executes batch-1 UNet / Transformer denoising and VAE decode/encode on Core ML ``CPU_AND_NE``
    while performing exact pixel/latent reconciliation across all ``op.num_views`` views.
    Works for all 5 tasks: ``flip``, ``twisting_squares``, ``rubiks_cube``,
    ``rotation_overlay``, and ``hidden_overlay``.
    """
    op = _get_operator(task)
    assert len(prompts) == op.num_views, f"Expected {op.num_views} prompts for {task}, got {len(prompts)}"
    if pixel_reconcile is None:
        pixel_reconcile = (not op.is_permutation) or (not op.equivariant_to_latent)

    is_rubiks = task == "rubiks_cube"
    if size is None:
        if model_key == "sana_sprint":
            size = 768 if is_rubiks else 1024
        elif model_key == "sdxl_turbo":
            size = 576 if is_rubiks else 512
        else:
            size = 480 if is_rubiks else 512

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    torch_device = get_best_torch_device()

    t_load0 = time.perf_counter()
    if _backend is not None:
        bundle = _backend
        meta = getattr(bundle, "meta", {})
        model_key = meta.get("model_key", model_key)
    else:
        info = apple_npu_info()
        if not info["ane_runtime_supported"]:
            raise RuntimeError(
                "Apple NPU Core ML runtime (CompiledMLModel / CPU_AND_NE) requires macOS >= 13 on Apple Silicon "
                f"with coremltools installed (detected system={info['system']}, machine={info['machine']}, "
                f"coremltools={info['coremltools_version']}). Note: PyTorch MPS only targets Apple GPU, not NPU; "
                "you can still pre-export .mlpackage models here with export_coreml_models()."
            )
        bundle_dir = Path(coreml_dir) if coreml_dir else Path(f"./coreml_models/{model_key}_{size}")
        if not bundle_dir.exists() or not (bundle_dir / "meta.json").exists():
            if not auto_export:
                raise FileNotFoundError(f"Core ML bundle not found at {bundle_dir} (set auto_export=True to build it).")
            export_coreml_models(
                model_key=model_key,
                out_dir=bundle_dir,
                size=size,
                quantize_nbits=quantize_nbits,
                ane_attention=True,
            )
        bundle = _CoreMLBundle(bundle_dir, compute_units=compute_units, cache_compiled=cache_compiled)
        meta = bundle.meta

    model_id = meta.get(
        "model_id",
        {
            "sd_turbo": "stabilityai/sd-turbo",
            "sdxl_turbo": "stabilityai/sdxl-turbo",
            "sana_sprint": "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        }.get(model_key, "stabilityai/sd-turbo"),
    )
    size = int(meta.get("size", size))
    lat_h = int(meta.get("latent_h", size // (32 if model_key == "sana_sprint" else 8)))
    lat_w = int(meta.get("latent_w", size // (32 if model_key == "sana_sprint" else 8)))
    lat_c = int(meta.get("latent_channels", 32 if model_key == "sana_sprint" else 4))
    scaling_factor = float(meta.get("scaling_factor", 0.41407 if model_key == "sana_sprint" else (0.13025 if model_key == "sdxl_turbo" else 0.18215)))

    load_s = time.perf_counter() - t_load0

    # Prepare lightweight tokenizer / scheduler (and optional MPS/CPU fallback encoder if needed)
    sync_torch_device(torch_device)
    t0 = time.perf_counter()
    gen = torch.Generator(device="cpu").manual_seed(seed)

    # Encode prompts
    fallback_vae = None
    match_info = None
    if model_key == "sd_turbo":
        from diffusers import EulerDiscreteScheduler
        from transformers import CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler", timestep_spacing="trailing"
        )
        toks = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        if bundle.text_encoder is not None:
            cond_list = [bundle.encode_text(toks.input_ids[k : k + 1].astype(np.int32)) for k in range(op.num_views)]
            prompt_embeds_np = np.concatenate(cond_list, axis=0).astype(np.float16)
        else:
            from transformers import CLIPTextModel

            te = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(torch_device).eval()
            with torch.no_grad():
                prompt_embeds_np = te(torch.from_numpy(toks.input_ids).to(torch_device))[0].cpu().numpy().astype(np.float16)
            del te

        if bundle.vae_encoder is None and pixel_reconcile:
            from diffusers import AutoencoderKL
            fallback_vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(torch_device).eval()

        scheduler.set_timesteps(num_steps, device="cpu")
        timesteps = scheduler.timesteps
        sched_sigmas = scheduler.sigmas.numpy().astype(np.float32)
        pred_type = scheduler.config.prediction_type
        init_sigma = float(scheduler.init_noise_sigma)

        if op.is_permutation and not pixel_reconcile:
            base_lat = torch.randn(op.num_primes, lat_c, lat_h, lat_w, generator=gen)
            latents = op.derive([base_lat[k] for k in range(op.num_primes)]) * init_sigma
        else:
            latents = torch.randn(op.num_views, lat_c, lat_h, lat_w, generator=gen) * init_sigma

        warm_logits = None
        primes = None
        last_rec_01 = None
        for i, t in enumerate(timesteps):
            scale_in = float(scheduler.scale_model_input(torch.ones(1), t).item())
            t_np = np.array([float(t.item())], dtype=np.float16)
            noise_preds = []
            for v in range(op.num_views):
                s_np = (latents[v : v + 1].numpy() * scale_in).astype(np.float16)
                np_v = bundle.predict_unet({
                    "sample": s_np,
                    "timestep": t_np,
                    "encoder_hidden_states": prompt_embeds_np[v : v + 1],
                })
                noise_preds.append(torch.from_numpy(np_v.astype(np.float32)))
            noise_pred = torch.cat(noise_preds, dim=0)

            sigma = float(sched_sigmas[i])
            sigma_next = float(sched_sigmas[i + 1])
            if pred_type == "epsilon":
                pred_x0 = latents - sigma * noise_pred
            elif pred_type == "v_prediction":
                alpha_t = 1.0 / ((sigma ** 2 + 1.0) ** 0.5)
                sigma_t = sigma * alpha_t
                pred_x0 = alpha_t * latents - sigma_t * noise_pred
            else:
                pred_x0 = noise_pred

            if op.is_permutation and not pixel_reconcile:
                if pred_type == "epsilon":
                    noise_pred_avg = op.invert_velocities(noise_pred)
                    latents = scheduler.step(noise_pred_avg, t, latents, return_dict=False)[0]
                else:
                    pred_x0_rec = op.invert_and_average(pred_x0)
                    eps_eff = (latents - pred_x0_rec) / max(sigma, 1e-5)
                    latents = latents + (sigma_next - sigma) * eps_eff
            else:
                scaled_x0 = (pred_x0 / scaling_factor).numpy().astype(np.float16)
                dec_imgs = [torch.from_numpy(bundle.decode_vae(scaled_x0[v : v + 1]).astype(np.float32)) for v in range(op.num_views)]
                imgs_01 = ((torch.cat(dec_imgs, dim=0) + 1.0) * 0.5).clamp(0.0, 1.0).to(torch_device)
                if i == 0 and isinstance(op, RubiksCubeOperator):
                    match_info = _reorder_rubiks_scrambled_npu(
                        op, imgs_01, prompts, latents, prompt_embeds_np
                    )
                if op.is_permutation:
                    rec_01 = op.invert_and_average(imgs_01)
                    primes = [rec_01[k].cpu() for k in range(op.num_primes)]
                elif isinstance(op, RotationOverlayOperator):
                    rec_01, primes_dev, warm_logits = op.reconcile(
                        imgs_01, init_logits=warm_logits, steps=reconcile_steps
                    )
                    primes = [p.cpu() for p in primes_dev]
                else:
                    rec_01, primes_dev = op.reconcile(imgs_01)
                    primes = [p.cpu() for p in primes_dev]
                last_rec_01 = rec_01.cpu()

                if i + 1 < len(timesteps) or sigma > 1e-4:
                    rec_m11 = (2.0 * last_rec_01 - 1.0)
                    if bundle.vae_encoder is not None:
                        rec_np = rec_m11.numpy().astype(np.float16)
                        rec_lats = [
                            torch.from_numpy(bundle.encode_vae(rec_np[v : v + 1]).astype(np.float32))
                            for v in range(op.num_views)
                        ]
                        rec_lat = torch.cat(rec_lats, dim=0) * scaling_factor
                    else:
                        with torch.no_grad():
                            rec_lat = (
                                fallback_vae.encode(rec_m11.to(torch_device)).latent_dist.mean.cpu() * scaling_factor
                            )
                    if sigma > 1e-4:
                        deriv = (latents - rec_lat) / sigma
                        latents = latents + (sigma_next - sigma) * deriv
                    else:
                        latents = rec_lat

        if op.is_permutation and not pixel_reconcile:
            z_final = (latents / scaling_factor).numpy().astype(np.float16)
            dec_imgs = [torch.from_numpy(bundle.decode_vae(z_final[v : v + 1]).astype(np.float32)) for v in range(op.num_views)]
            decoded_01 = ((torch.cat(dec_imgs, dim=0) + 1.0) * 0.5).clamp(0.0, 1.0)
            primes = [decoded_01[k] for k in range(op.num_primes)]
        elif op.is_permutation:
            primes = [last_rec_01[k] for k in range(op.num_primes)]
        views_01 = op.derive(primes)

    elif model_key == "sdxl_turbo":
        from diffusers import AutoPipelineForText2Image, EulerAncestralDiscreteScheduler

        pipe_txt = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float32).to(torch_device)
        with torch.no_grad():
            prompt_embeds, _, pooled_prompt_embeds, _ = pipe_txt.encode_prompt(
                prompt=prompts, device=torch_device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )
            add_time_ids = pipe_txt._get_add_time_ids(
                (size, size), (0, 0), (size, size),
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=pipe_txt.text_encoder_2.config.projection_dim,
            ).cpu().numpy().astype(np.float16)
            prompt_embeds_np = prompt_embeds.cpu().numpy().astype(np.float16)
            pooled_np = pooled_prompt_embeds.cpu().numpy().astype(np.float16)
        scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_txt.scheduler.config, timestep_spacing="trailing")
        del pipe_txt

        scheduler.set_timesteps(num_steps, device="cpu")
        timesteps = scheduler.timesteps
        sched_sigmas = scheduler.sigmas.numpy().astype(np.float32)
        init_sigma = float(scheduler.init_noise_sigma)

        if op.is_permutation and not pixel_reconcile:
            base_lat = torch.randn(op.num_primes, lat_c, lat_h, lat_w, generator=gen)
            latents = op.derive([base_lat[k] for k in range(op.num_primes)]) * init_sigma
        else:
            latents = torch.randn(op.num_views, lat_c, lat_h, lat_w, generator=gen) * init_sigma

        warm_logits = None
        primes = None
        last_rec_01 = None
        for i, t in enumerate(timesteps):
            scale_in = float(scheduler.scale_model_input(torch.ones(1), t).item())
            t_np = np.array([float(t.item())], dtype=np.float16)
            noise_preds = []
            for v in range(op.num_views):
                np_v = bundle.predict_unet({
                    "sample": (latents[v : v + 1].numpy() * scale_in).astype(np.float16),
                    "timestep": t_np,
                    "encoder_hidden_states": prompt_embeds_np[v : v + 1],
                    "text_embeds": pooled_np[v : v + 1],
                    "time_ids": add_time_ids,
                })
                noise_preds.append(torch.from_numpy(np_v.astype(np.float32)))
            noise_pred = torch.cat(noise_preds, dim=0)
            sigma = float(sched_sigmas[i])
            sigma_next = float(sched_sigmas[i + 1])
            pred_x0 = latents - sigma * noise_pred

            if op.is_permutation and not pixel_reconcile:
                noise_pred_avg = op.invert_velocities(noise_pred)
                latents = scheduler.step(noise_pred_avg, t, latents, generator=gen, return_dict=False)[0]
            else:
                scaled_x0 = (pred_x0 / scaling_factor).numpy().astype(np.float16)
                dec_imgs = [torch.from_numpy(bundle.decode_vae(scaled_x0[v : v + 1]).astype(np.float32)) for v in range(op.num_views)]
                imgs_01 = ((torch.cat(dec_imgs, dim=0) + 1.0) * 0.5).clamp(0.0, 1.0).to(torch_device)
                if i == 0 and isinstance(op, RubiksCubeOperator):
                    match_info = _reorder_rubiks_scrambled_npu(
                        op, imgs_01, prompts, latents, prompt_embeds_np, pooled_np
                    )
                if op.is_permutation:
                    rec_01 = op.invert_and_average(imgs_01)
                    primes = [rec_01[k].cpu() for k in range(op.num_primes)]
                elif isinstance(op, RotationOverlayOperator):
                    rec_01, primes_dev, warm_logits = op.reconcile(imgs_01, init_logits=warm_logits, steps=reconcile_steps)
                    primes = [p.cpu() for p in primes_dev]
                else:
                    rec_01, primes_dev = op.reconcile(imgs_01)
                    primes = [p.cpu() for p in primes_dev]
                last_rec_01 = rec_01.cpu()
                rec_np = (2.0 * last_rec_01 - 1.0).numpy().astype(np.float16)
                rec_lat = torch.cat(
                    [torch.from_numpy(bundle.encode_vae(rec_np[v : v + 1]).astype(np.float32)) for v in range(op.num_views)],
                    dim=0,
                ) * scaling_factor
                if sigma > 1e-4:
                    deriv = (latents - rec_lat) / sigma
                    latents = latents + (sigma_next - sigma) * deriv
                else:
                    latents = rec_lat

        if op.is_permutation and not pixel_reconcile:
            z_final = (latents / scaling_factor).numpy().astype(np.float16)
            dec_imgs = [torch.from_numpy(bundle.decode_vae(z_final[v : v + 1]).astype(np.float32)) for v in range(op.num_views)]
            decoded_01 = ((torch.cat(dec_imgs, dim=0) + 1.0) * 0.5).clamp(0.0, 1.0)
            primes = [decoded_01[k] for k in range(op.num_primes)]
        elif op.is_permutation:
            primes = [last_rec_01[k] for k in range(op.num_primes)]
        views_01 = op.derive(primes)

    elif model_key == "sana_sprint":
        from diffusers import SanaSprintPipeline
        from diffusers.pipelines.sana.pipeline_sana_sprint import retrieve_timesteps

        pipe_txt = SanaSprintPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(torch_device)
        with torch.no_grad():
            prompt_embeds, prompt_attention_mask = pipe_txt.encode_prompt(
                prompts, num_images_per_prompt=1, device=torch_device, clean_caption=False, max_sequence_length=300
            )
            prompt_embeds_np = prompt_embeds.cpu().numpy().astype(np.float16)
            mask_np = prompt_attention_mask.cpu().numpy().astype(np.float16)
        scheduler = pipe_txt.scheduler
        sigma_data = float(meta.get("sigma_data", scheduler.config.sigma_data))
        guidance_val = float(4.5 * pipe_txt.transformer.config.guidance_embeds_scale)
        del pipe_txt

        timesteps, _ = retrieve_timesteps(
            scheduler, num_steps, torch.device("cpu"), timesteps=None, sigmas=None,
            max_timesteps=1.57080, intermediate_timesteps=1.3 if num_steps == 2 else None,
        )
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(0)

        latents = torch.randn(op.num_views, lat_c, lat_h, lat_w, generator=gen)
        if op.is_permutation and not pixel_reconcile:
            latents = op.derive([latents[k] for k in range(op.num_primes)])
        latents = latents * sigma_data

        timesteps_loop = timesteps[:-1]
        warm_logits = None
        primes = None
        denoised = None
        last_rec_01 = None
        guidance_np = np.array([guidance_val], dtype=np.float16)

        for i, t in enumerate(timesteps_loop):
            timestep = t.expand(op.num_views)
            scm_t = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
            scm_exp = scm_t.view(-1, 1, 1, 1)
            latent_in = (latents / sigma_data) * torch.sqrt(scm_exp ** 2 + (1.0 - scm_exp) ** 2)

            raw_preds = []
            for v in range(op.num_views):
                np_v = bundle.predict_unet({
                    "hidden_states": latent_in[v : v + 1].numpy().astype(np.float16),
                    "timestep": np.array([float(scm_t[v].item())], dtype=np.float16),
                    "encoder_hidden_states": prompt_embeds_np[v : v + 1],
                    "encoder_attention_mask": mask_np[v : v + 1],
                    "guidance": guidance_np,
                })
                raw_preds.append(torch.from_numpy(np_v.astype(np.float32)))
            noise_pred = torch.cat(raw_preds, dim=0)
            noise_pred = (
                (1.0 - 2.0 * scm_exp) * latent_in + (1.0 - 2.0 * scm_exp + 2.0 * scm_exp ** 2) * noise_pred
            ) / torch.sqrt(scm_exp ** 2 + (1.0 - scm_exp) ** 2)
            noise_pred = noise_pred * sigma_data

            if op.is_permutation and not pixel_reconcile:
                noise_pred = op.invert_velocities(noise_pred)
                latents, denoised = scheduler.step(noise_pred, timestep, latents, return_dict=False)
            else:
                _, denoised_step = scheduler.step(noise_pred, timestep, latents, return_dict=False)
                z_in = ((denoised_step / sigma_data) / scaling_factor).numpy().astype(np.float16)
                dec_imgs = [torch.from_numpy(bundle.decode_vae(z_in[v : v + 1]).astype(np.float32)) for v in range(op.num_views)]
                imgs_01 = ((torch.cat(dec_imgs, dim=0) + 1.0) * 0.5).clamp(0.0, 1.0).to(torch_device)
                if i == 0 and isinstance(op, RubiksCubeOperator):
                    match_info = _reorder_rubiks_scrambled_npu(
                        op, imgs_01, prompts, latents, prompt_embeds_np, mask_np
                    )
                if op.is_permutation:
                    rec_01 = op.invert_and_average(imgs_01)
                    primes = [rec_01[k].cpu() for k in range(op.num_primes)]
                elif isinstance(op, RotationOverlayOperator):
                    rec_01, primes_dev, warm_logits = op.reconcile(imgs_01, init_logits=warm_logits, steps=reconcile_steps)
                    primes = [p.cpu() for p in primes_dev]
                else:
                    rec_01, primes_dev = op.reconcile(imgs_01)
                    primes = [p.cpu() for p in primes_dev]
                last_rec_01 = rec_01.cpu()
                if i + 1 < len(timesteps_loop):
                    rec_np = (2.0 * last_rec_01 - 1.0).numpy().astype(np.float16)
                    rec_lat = torch.cat(
                        [torch.from_numpy(bundle.encode_vae(rec_np[v : v + 1]).astype(np.float32)) for v in range(op.num_views)],
                        dim=0,
                    ) * scaling_factor
                    rec_denoised = rec_lat * sigma_data
                    t_next = timesteps_loop[i + 1]
                    latents = math.cos(float(t_next)) * rec_denoised + math.sin(float(t_next)) * torch.randn(rec_denoised.shape, generator=gen) * sigma_data
                    denoised = rec_denoised

        if op.is_permutation and not pixel_reconcile:
            z_final = ((denoised / sigma_data) / scaling_factor).numpy().astype(np.float16)
            dec_imgs = [torch.from_numpy(bundle.decode_vae(z_final[v : v + 1]).astype(np.float32)) for v in range(op.num_views)]
            decoded_01 = ((torch.cat(dec_imgs, dim=0) + 1.0) * 0.5).clamp(0.0, 1.0)
            primes = [decoded_01[k] for k in range(op.num_primes)]
        elif op.is_permutation:
            primes = [last_rec_01[k] for k in range(op.num_primes)]
        views_01 = op.derive(primes)
    else:
        raise ValueError(f"Unsupported Apple NPU Core ML model_key: {model_key!r}")

    sync_torch_device(torch_device)
    gen_s = time.perf_counter() - t0

    for idx, p in enumerate(primes):
        _tensor_to_pil(p).save(out_path / f"prime_{idx}.png")
    view_pils = [_tensor_to_pil(v) for v in views_01]
    for idx, v in enumerate(view_pils):
        v.save(out_path / f"view_{idx}.png")
    _make_grid(view_pils).save(out_path / "final_grid.png")

    res = {
        "method": f"apple_npu_{model_key}",
        "task": task,
        "compute_units": compute_units,
        "reconcile_device": str(torch_device),
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
# CLI Entrypoint
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(description="Apple Neural Engine (NPU) export & zero-shot illusion CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="Print Apple Silicon / NPU / Core ML capability status")

    exp = sub.add_parser("export", help="Export a Diffusers model to ANE-ready .mlpackage files")
    exp.add_argument("--model", default="sd_turbo", choices=["sd_turbo", "sdxl_turbo", "sana_sprint"])
    exp.add_argument("--out-dir", required=True)
    exp.add_argument("--size", type=int, default=None)
    exp.add_argument("--quantize-nbits", type=int, default=6)
    exp.add_argument("--no-ane-attention", action="store_true")

    run = sub.add_parser("run", help="Run zero-shot illusion generation on Apple NPU (Core ML CPU_AND_NE)")
    run.add_argument("--task", required=True, choices=["flip", "twisting_squares", "rubiks_cube", "rotation_overlay", "hidden_overlay"])
    run.add_argument("--prompts", nargs="+", required=True)
    run.add_argument("--model", default="sd_turbo", choices=["sd_turbo", "sdxl_turbo", "sana_sprint"])
    run.add_argument("--coreml-dir", default=None)
    run.add_argument("--compute-units", default="CPU_AND_NE")
    run.add_argument("--steps", type=int, default=4)
    run.add_argument("--size", type=int, default=None)
    run.add_argument("--quantize-nbits", type=int, default=6)
    run.add_argument("--out-dir", default="./outputs/apple_npu")
    run.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.cmd == "info":
        print(json.dumps(apple_npu_info(), indent=2))
    elif args.cmd == "export":
        res = export_coreml_models(
            model_key=args.model,
            out_dir=args.out_dir,
            size=args.size,
            quantize_nbits=args.quantize_nbits if args.quantize_nbits > 0 else None,
            ane_attention=not args.no_ane_attention,
        )
        print(json.dumps(res, indent=2))
    elif args.cmd == "run":
        res = run_apple_npu_zeroshot(
            task=args.task,
            prompts=args.prompts,
            model_key=args.model,
            coreml_dir=args.coreml_dir,
            compute_units=args.compute_units,
            num_steps=args.steps,
            size=args.size,
            quantize_nbits=args.quantize_nbits if args.quantize_nbits > 0 else None,
            out_dir=args.out_dir,
            seed=args.seed,
        )
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    _cli()
