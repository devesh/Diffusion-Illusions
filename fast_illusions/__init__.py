# Fast Diffusion Illusions package

from fast_illusions.apple_npu import (
    ANEAttentionProcessor,
    apple_npu_info,
    enable_ane_attention,
    export_coreml_models,
    get_best_torch_device,
    run_apple_npu_zeroshot,
    sync_torch_device,
)

__all__ = [
    "ANEAttentionProcessor",
    "apple_npu_info",
    "enable_ane_attention",
    "export_coreml_models",
    "get_best_torch_device",
    "run_apple_npu_zeroshot",
    "sync_torch_device",
]

