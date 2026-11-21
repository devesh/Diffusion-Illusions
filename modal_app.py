import gc
import json
import sys
from pathlib import Path
import modal

app = modal.App("diffusion-illusions-fast")

hf_cache = modal.Volume.from_name("ambigram-hf-cache", create_if_missing=True)
LOCAL_REPO_DIR = Path(__file__).resolve().parent

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.6.0",
        "torchvision",
        "diffusers>=0.37.0",
        "transformers>=4.50.0",
        "accelerate>=1.3.0",
        "safetensors",
        "einops",
        "easydict",
        "sentencepiece",
        "protobuf",
        "pyyaml",
        "scipy",
        "icecream",
        "pillow",
        "opencv-python-headless",
        "rp",
        "omegaconf",
        "huggingface_hub",
    )
    .run_commands("pip install --no-deps git+https://github.com/speridlabs/iris-3b.git")
    .env({
        "HF_HOME": "/hf-cache",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
)

image = base_image.add_local_dir(
    LOCAL_REPO_DIR,
    remote_path="/workspace",
    ignore=[".git", "assets", "notes", "other", "UV_Maps", "outputs", "coreml_models", "__pycache__", "*.ipynb"],
)


def _collect_files(out_dir: str) -> dict[str, bytes]:
    files = {}
    for p in sorted(Path(out_dir).glob("*.png")):
        files[p.name] = p.read_bytes()
    return files


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={"/hf-cache": hf_cache},
)
def run_single(
    method: str,
    task: str,
    prompts: list[str],
    negative_prompt: str = "",
    seed: int = 42,
    size: int | None = None,
    orig_iters: int = 2000,
    sds_iters: int = 350,
    dtl_model: str = "sana_sprint",
    num_steps: int = 4,
) -> tuple[dict, dict[str, bytes]]:
    import torch

    if "/workspace" not in sys.path:
        sys.path.insert(0, "/workspace")

    out_dir = f"/tmp/run_{task}_{method}"
    is_rubiks = task == "rubiks_cube"
    if method == "orig_sds":
        from fast_illusions.orig_baseline import run_orig_baseline

        metrics = run_orig_baseline(
            task=task,
            prompts=prompts,
            negative_prompt=negative_prompt,
            num_iters=orig_iters,
            size=size or (252 if is_rubiks else 256),
            snapshot_iters=(500, 1000, orig_iters),
            out_dir=out_dir,
            seed=seed,
        )
    elif method == "fast_sds_dtl":
        from fast_illusions.fast_sds_dtl import run_fast_sds_dtl

        metrics = run_fast_sds_dtl(
            task=task,
            prompts=prompts,
            negative_prompt=negative_prompt,
            sds_iters=sds_iters,
            dtl_rounds=2,
            dtl_steps_per_round=120,
            dtl_model=dtl_model,
            size=size or (480 if is_rubiks else 512),
            use_taesd=True,
            use_compile=True,
            out_dir=out_dir,
            seed=seed,
        )
    elif method == "iris3b_turbo":
        from fast_illusions.zeroshot_modern import run_iris3b_turbo_zeroshot

        metrics = run_iris3b_turbo_zeroshot(
            task=task,
            prompts=prompts,
            size=size or (768 if is_rubiks else 1024),
            out_dir=out_dir,
            seed=seed,
        )
    elif method == "sdxl_turbo":
        from fast_illusions.zeroshot_modern import run_sd_turbo_zeroshot

        metrics = run_sd_turbo_zeroshot(
            task=task,
            prompts=prompts,
            model_id="stabilityai/sdxl-turbo",
            num_steps=num_steps,
            size=size or (576 if is_rubiks else 512),
            out_dir=out_dir,
            seed=seed,
        )
    elif method == "sd_turbo":
        from fast_illusions.zeroshot_modern import run_sd_turbo_zeroshot

        metrics = run_sd_turbo_zeroshot(
            task=task,
            prompts=prompts,
            model_id="stabilityai/sd-turbo",
            num_steps=num_steps,
            size=size or (480 if is_rubiks else 512),
            out_dir=out_dir,
            seed=seed,
        )
    elif method == "flux2_klein":
        from fast_illusions.zeroshot_modern import run_flux2_klein_zeroshot

        metrics = run_flux2_klein_zeroshot(
            task=task,
            prompts=prompts,
            num_steps=num_steps,
            size=size or (576 if is_rubiks else 512),
            out_dir=out_dir,
            seed=seed,
        )
    elif method == "sana_sprint":
        from fast_illusions.zeroshot_modern import run_sana_sprint_zeroshot

        metrics = run_sana_sprint_zeroshot(
            task=task,
            prompts=prompts,
            num_steps=num_steps,
            size=size or (768 if is_rubiks else 1024),
            out_dir=out_dir,
            seed=seed,
        )
    elif method == "hybrid_sana_sds":
        from fast_illusions.fast_sds_dtl import run_hybrid_zeroshot_sds

        metrics = run_hybrid_zeroshot_sds(
            task=task,
            prompts=prompts,
            negative_prompt=negative_prompt,
            init_method="sana_sprint",
            init_steps=num_steps,
            sds_iters=sds_iters,
            size=size or (480 if is_rubiks else 512),
            out_dir=out_dir,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    files = _collect_files(out_dir)
    hf_cache.commit()
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, files


def _save_result(task: str, method: str, metrics: dict, files: dict[str, bytes]) -> Path:
    dest = LOCAL_REPO_DIR / "outputs" / task / method
    dest.mkdir(parents=True, exist_ok=True)
    for fname, data in files.items():
        (dest / fname).write_bytes(data)
    (dest / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return dest


@app.local_entrypoint()
def main(
    mode: str = "single",
    method: str = "iris3b_turbo",
    task: str = "flip",
    prompts_json: str = "",
    methods_csv: str = "",
    out_name: str = "",
    seed: int = 42,
    size: int = 0,
    num_steps: int = 4,
    orig_iters: int = 2000,
    sds_iters: int = 350,
    dtl_model: str = "sana_sprint",
):
    import yaml

    ex = yaml.safe_load((LOCAL_REPO_DIR / "source" / "example_prompts.yaml").read_text())

    if prompts_json:
        prompts = json.loads(prompts_json)
    elif task == "flip":
        prompts = [ex["pencil_giraffe_head"], ex["pencil_penguin"]]
    elif task == "twisting_squares":
        prompts = ["Hatsune Miku", "A dog's face photograph 4k"]
    elif task == "rubiks_cube":
        prompts = [
            "vibrant oil painting of a sunflower field under blue sky",
            "vibrant oil painting of a red fox in a green forest",
            "vibrant oil painting of a majestic lion portrait",
            "vibrant oil painting of a colorful parrot in tropical leaves",
            "vibrant oil painting of a sailing ship on blue ocean waves",
            "vibrant oil painting of a snow-capped mountain at sunset",
            "vibrant oil painting of a golden retriever dog portrait",
            "vibrant oil painting of an owl perched in autumn woods",
            "vibrant oil painting of a Bengal tiger portrait",
            "vibrant oil painting of a lighthouse by crashing ocean waves",
            "vibrant oil painting of a vintage red steam train",
            "vibrant oil painting of a butterfly on colorful flowers",
        ]
    elif task == "rotation_overlay":
        prompts = [ex["miku"], ex["froggo"], ex["lipstick"], ex["pyramids"]]
    elif task == "hidden_overlay":
        prompts = [
            "oil painting of a red fox in a forest",
            "oil painting of a snowy mountain cabin",
            "oil painting of a sailing ship on the ocean",
            "oil painting of a sunflower field at sunset",
            "oil painting of a majestic lion portrait",
        ]
    else:
        raise ValueError(f"Unknown task: {task}")

    size_arg = size if size > 0 else None
    if mode == "single":
        metrics, files = run_single.remote(
            method=method,
            task=task,
            prompts=prompts,
            seed=seed,
            size=size_arg,
            num_steps=num_steps,
            orig_iters=orig_iters,
            sds_iters=sds_iters,
            dtl_model=dtl_model,
        )
        dest = _save_result(task, out_name or method, metrics, files)
        print(json.dumps(metrics, indent=2))
        print("Saved to:", dest)
    elif mode == "benchmark":
        methods_to_run = [
            ("orig_sds", {}),
            ("fast_sds_dtl", {"dtl_model": dtl_model}),
            ("sdxl_turbo", {"num_steps": 4}),
            ("sd_turbo", {"num_steps": 4}),
            ("sana_sprint", {"num_steps": 4 if task == "rubiks_cube" else 2}),
            ("flux2_klein", {"num_steps": num_steps}),
            ("iris3b_turbo", {}),
        ]
        if methods_csv:
            wanted = {m.strip() for m in methods_csv.split(",") if m.strip()}
            methods_to_run = [(m, kw) for (m, kw) in methods_to_run if m in wanted]
        handles = []
        for m_name, kw in methods_to_run:
            h = run_single.spawn(
                method=m_name,
                task=task,
                prompts=prompts,
                seed=seed,
                size=size_arg,
                orig_iters=orig_iters,
                sds_iters=sds_iters,
                **kw,
            )
            handles.append((m_name, h))

        summary = []
        for m_name, h in handles:
            try:
                metrics, files = h.get()
                dest = _save_result(task, m_name, metrics, files)
                metrics["saved_dir"] = str(dest)
                summary.append(metrics)
                print(f"[{m_name}] DONE:", json.dumps(metrics))
            except Exception as e:
                err = {"method": m_name, "task": task, "error": str(e)}
                summary.append(err)
                print(f"[{m_name}] ERROR:", e)

        summary_path = LOCAL_REPO_DIR / "outputs" / task / "benchmark_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        print("Benchmark summary saved to:", summary_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")
