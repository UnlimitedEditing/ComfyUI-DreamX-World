"""
ComfyUI-DreamX-World nodes.

Wraps the DreamX-World-5B model (https://huggingface.co/GD-ML/DreamX-World-5B)
for camera-controlled video generation from a single init image.

Architecture: rolling init-frame chunks — each predefined camera move generates
one video chunk; the last frame of each chunk becomes the init image for the next.
This mirrors the interactive session mechanism and produces a single stitched video.

Requires the AMAP-ML/DreamX-World repo cloned into the same custom_nodes directory.
"""

import os
import sys
import pathlib
import tempfile
import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from einops import rearrange

# ─── locate the cloned DreamX-World inference repo ───────────────────────────

def _find_dreamx_repo():
    """Find the DreamX-World repo cloned alongside this package in custom_nodes."""
    # This package lives at custom_nodes/ComfyUI-DreamX-World/
    # DreamX-World is cloned to custom_nodes/DreamX-World/
    custom_nodes = pathlib.Path(__file__).parent.parent
    candidate = custom_nodes / "DreamX-World"
    if candidate.is_dir():
        return str(candidate)
    raise RuntimeError(
        "DreamX-World repo not found. Expected at: " + str(candidate) + "\n"
        "Ensure https://github.com/AMAP-ML/DreamX-World is in requirements.github."
    )

def _ensure_dreamx_on_path():
    dreamx_dir = _find_dreamx_repo()

    # Ensure subdirs have __init__.py so they register as regular packages
    # and aren't shadowed by namespace packages from other custom nodes.
    for subdir in ("utils", "pipeline", "wan"):
        init = os.path.join(dreamx_dir, subdir, "__init__.py")
        if not os.path.exists(init):
            open(init, "w").close()

    # Move DreamX-World to the front of sys.path unconditionally.
    if dreamx_dir in sys.path:
        sys.path.remove(dreamx_dir)
    sys.path.insert(0, dreamx_dir)

    # If 'utils' was already imported from a different package, evict it so
    # the next import picks up DreamX-World/utils/ instead.
    existing = sys.modules.get("utils")
    if existing is not None:
        existing_file = getattr(existing, "__file__", "") or ""
        if dreamx_dir not in existing_file:
            for key in list(sys.modules.keys()):
                if key == "utils" or key.startswith("utils."):
                    del sys.modules[key]

    return dreamx_dir

# ─── predefined camera moves ─────────────────────────────────────────────────

# Keys are user-facing names passed in move_sequence (comma-separated).
# Values are (action_str, speed_weight). action_str uses DreamX's WASD/IJKL encoding:
#   w=forward  s=back  a=strafe-left  d=strafe-right
#   i=tilt-up  k=tilt-down  j=pan-left  l=pan-right
#   Combinations: "wi" = forward while tilting up, etc.

CAMERA_MOVES = {
    # Translations
    "forward":         ("w",  1),
    "back":            ("s",  1),
    "strafe_left":     ("a",  1),
    "strafe_right":    ("d",  1),
    # Rotations
    "pan_left":        ("j",  1),
    "pan_right":       ("l",  1),
    "tilt_up":         ("i",  1),
    "tilt_down":       ("k",  1),
    # Cinematic arcs (dolly + rotation)
    "arc_left":        ("wj", 1),   # forward + pan-left  = orbit/arc left
    "arc_right":       ("wl", 1),   # forward + pan-right = orbit/arc right
    "crane_up":        ("wi", 1),   # forward + tilt-up   = ascending crane
    "crane_down":      ("wk", 1),   # forward + tilt-down = descending crane
    "retreat_up":      ("si", 1),   # pull-back + tilt-up = reveal shot
    "retreat_down":    ("sk", 1),   # pull-back + tilt-down
    # No movement
    "static":          ("",   1),
}

MOVE_NAMES = sorted(CAMERA_MOVES.keys())

# ─── flash-attn → SDPA fallback ──────────────────────────────────────────────

def _patch_flash_attn_fallback():
    """
    Replace DreamX's flash_attention with a SDPA fallback when flash_attn2 is
    not installed. The patch respects k_lens masking so padding tokens in the
    text context are correctly excluded from cross-attention.
    """
    try:
        import wan.modules.attention as attn_mod
        if getattr(attn_mod, "FLASH_ATTN_2_AVAILABLE", True):
            return  # flash_attn2 is available — nothing to patch
        import torch, torch.nn.functional as F

        def _sdpa(q, k, v, q_lens=None, k_lens=None, window_size=(-1, -1), **kw):
            # Incoming shape: (batch, seq, heads, head_dim)
            b, s_q, h, d = q.shape
            q = q.transpose(1, 2)   # → (b, h, s_q, d)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            mask = None
            if k_lens is not None:
                s_k = k.shape[2]
                # True = attend, False = ignore padding
                m = torch.arange(s_k, device=q.device).unsqueeze(0) < k_lens.to(q.device).unsqueeze(1)
                mask = m.unsqueeze(1).unsqueeze(1).expand(b, h, s_q, s_k)

            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return out.transpose(1, 2).contiguous()  # → (b, s_q, h, d)

        attn_mod.flash_attention = _sdpa
        print("[DreamXModelLoader] patched flash_attention → SDPA fallback")
    except Exception as e:
        print(f"[DreamXModelLoader] flash_attention patch skipped: {e}")

# ─── pipeline cache (one per (wan_path, checkpoint_path) pair) ───────────────

_PIPELINE_CACHE: dict = {}

# ─── image helpers ───────────────────────────────────────────────────────────

_DREAMX_TRANSFORM = transforms.Compose([
    transforms.Resize((704, 1280)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def _pil_to_dreamx_tensor(pil_img, device, dtype=torch.bfloat16):
    """PIL RGB → (1, C, 1, H, W) bfloat16 on device, values in [-1, 1]."""
    t = _DREAMX_TRANSFORM(pil_img)           # (C, H, W)
    t = t.unsqueeze(0).unsqueeze(2)           # (1, C, 1, H, W)
    return t.to(device=device, dtype=dtype)

def _comfy_image_to_pil(image_tensor):
    """ComfyUI IMAGE (B, H, W, C) float32 [0,1] → PIL, first frame."""
    frame = image_tensor[0].cpu().float().numpy()
    frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(frame).convert("RGB")

def _video_tensor_to_comfy(video_uint8):
    """
    video_uint8: (T, H, W, C) numpy uint8 [0, 255]
    → ComfyUI IMAGE tensor (T, H, W, C) float32 [0, 1]
    """
    return torch.from_numpy(video_uint8.astype(np.float32) / 255.0)

# ─── nodes ───────────────────────────────────────────────────────────────────

class DreamXModelLoader:
    """
    Loads the DreamX-World pipeline once and caches it for the process lifetime.

    wan_base_path   — local directory of Wan-AI/Wan2.2-TI2V-5B (contains
                      models_t5_umt5-xxl-enc-bf16.pth, Wan2.2_VAE.pth,
                      google/umt5-xxl/, and transformer config.json).
    checkpoint_path — local path to GD-ML/DreamX-World-5B/model.safetensors
                      (the DreamX fine-tuned transformer weights).
    frames_per_chunk — latent frame count used for each sequence chunk.
                       pixel frames = (N - 1) * 4 + 1.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wan_base_path": ("STRING", {
                    "default": "/comfyui/models/dreamx/Wan2.2-TI2V-5B",
                    "multiline": False,
                }),
                "checkpoint_path": ("STRING", {
                    "default": "/comfyui/models/dreamx/DreamX-World-5B-bf16/model.safetensors",
                    "multiline": False,
                }),
                "frames_per_chunk": ("INT", {
                    "default": 12, "min": 3, "max": 123, "step": 3,
                    "display": "number",
                }),
            }
        }

    RETURN_TYPES = ("DREAMX_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "DreamX"

    @staticmethod
    def _resolve_path(p: str) -> str:
        """If a path starts with /comfyui/ but doesn't exist, remap it to
        the actual ComfyUI models directory via folder_paths."""
        if os.path.exists(p):
            return p
        if p.startswith("/comfyui/models/"):
            try:
                import folder_paths
                actual = p.replace("/comfyui/models", folder_paths.models_dir, 1)
                print(f"[DreamXModelLoader] remapped {p} -> {actual}")
                return actual
            except Exception:
                pass
        return p

    @staticmethod
    def _wget(url: str, dest: str) -> bool:
        import subprocess
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".tmp"
        print(f"[DreamXModelLoader] downloading {os.path.basename(dest)} ...")
        r = subprocess.run(["wget", "-q", "-O", tmp, url], check=False)
        if r.returncode == 0 and os.path.exists(tmp):
            os.rename(tmp, dest)
            print(f"[DreamXModelLoader]   -> {os.path.getsize(dest)/1e9:.1f} GB saved")
            return True
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"[DreamXModelLoader]   FAILED: {url}")
        return False

    def _ensure_models(self, wan_base: str, fp8_ckpt: str):
        """Download Wan2.2 support files and DreamX fp8 checkpoint if absent."""
        HF = "https://huggingface.co"
        WAN = f"{HF}/Wan-AI/Wan2.2-TI2V-5B/resolve/main"
        DX  = f"{HF}/GD-ML/DreamX-World-5B/resolve/main"

        single_files = [
            (f"{WAN}/models_t5_umt5-xxl-enc-bf16.pth",
             os.path.join(wan_base, "models_t5_umt5-xxl-enc-bf16.pth")),
            (f"{WAN}/Wan2.2_VAE.pth",
             os.path.join(wan_base, "Wan2.2_VAE.pth")),
            (f"{WAN}/config.json",
             os.path.join(wan_base, "config.json")),
            (f"{WAN}/google/umt5-xxl/special_tokens_map.json",
             os.path.join(wan_base, "google", "umt5-xxl", "special_tokens_map.json")),
            (f"{WAN}/google/umt5-xxl/spiece.model",
             os.path.join(wan_base, "google", "umt5-xxl", "spiece.model")),
            (f"{WAN}/google/umt5-xxl/tokenizer.json",
             os.path.join(wan_base, "google", "umt5-xxl", "tokenizer.json")),
            (f"{WAN}/google/umt5-xxl/tokenizer_config.json",
             os.path.join(wan_base, "google", "umt5-xxl", "tokenizer_config.json")),
        ]
        for url, dest in single_files:
            if not os.path.exists(dest):
                self._wget(url, dest)

        if not os.path.exists(fp8_ckpt):
            # concept_mapping places the fp32 at DreamX-World-5B/model.safetensors;
            # fall back to a self-downloaded temp path if it isn't there yet.
            fp32_cm  = fp8_ckpt.replace("DreamX-World-5B-fp8", "DreamX-World-5B")
            fp32_tmp = fp8_ckpt.replace("DreamX-World-5B-fp8", "DreamX-World-5B-fp32-tmp")
            fp32_src = fp32_cm if os.path.exists(fp32_cm) else fp32_tmp
            if not os.path.exists(fp32_src):
                self._wget(f"{DX}/model.safetensors", fp32_tmp)
                fp32_src = fp32_tmp
            if os.path.exists(fp32_src):
                print("[DreamXModelLoader] converting fp32 -> fp8 (~5.25 GB) ...")
                from safetensors.torch import load_file as _lf, save_file as _sf
                sd = _lf(fp32_src)
                sd_fp8 = {
                    k: v.to(torch.float8_e4m3fn)
                    if v.dtype in (torch.float32, torch.float16, torch.bfloat16)
                    else v
                    for k, v in sd.items()
                }
                os.makedirs(os.path.dirname(fp8_ckpt), exist_ok=True)
                _sf(sd_fp8, fp8_ckpt)
                print(f"[DreamXModelLoader] fp8 saved: {os.path.getsize(fp8_ckpt)/1e9:.1f} GB")
                # Only remove the self-downloaded temp; leave concept_mapping file intact
                if fp32_src == fp32_tmp and os.path.exists(fp32_tmp):
                    os.remove(fp32_tmp)

    def load(self, wan_base_path, checkpoint_path, frames_per_chunk):
        wan_base_path  = self._resolve_path(wan_base_path)
        checkpoint_path = self._resolve_path(checkpoint_path)
        cache_key = (wan_base_path, checkpoint_path, frames_per_chunk)
        if cache_key in _PIPELINE_CACHE:
            print("[DreamXModelLoader] using cached pipeline")
            return (_PIPELINE_CACHE[cache_key],)

        self._ensure_models(wan_base_path, checkpoint_path)

        dreamx_dir = _ensure_dreamx_on_path()
        _patch_flash_attn_fallback()

        from omegaconf import OmegaConf
        from pipeline.pipeline_causal_camera import CausalCameraInferencePipeline
        from safetensors.torch import load_file

        config_path = os.path.join(
            dreamx_dir, "configs/dreamx-ar/causal_camera_forcing_5b.yaml"
        )
        default_path = os.path.join(os.path.dirname(config_path), "default_config.yaml")

        config = OmegaConf.load(config_path)
        if os.path.exists(default_path):
            config = OmegaConf.merge(OmegaConf.load(default_path), config)

        device = torch.device("cuda")

        text_encoder_path = os.path.join(wan_base_path, "models_t5_umt5-xxl-enc-bf16.pth")
        tokenizer_path    = os.path.join(wan_base_path, "google/umt5-xxl")
        vae_path          = os.path.join(wan_base_path, "Wan2.2_VAE.pth")
        transformer_cfg   = os.path.join(wan_base_path, "config.json")
        model_config_path = transformer_cfg if os.path.exists(transformer_cfg) else None

        # wan_wrapper.py expects ./wan_models/Wan2.2-TI2V-5B-Camera/config.json
        # relative to the repo root. This file isn't shipped in the repo — write it.
        _cam_cfg_dir = os.path.join(dreamx_dir, "wan_models", "Wan2.2-TI2V-5B-Camera")
        _cam_cfg_file = os.path.join(_cam_cfg_dir, "config.json")
        if not os.path.exists(_cam_cfg_file):
            os.makedirs(_cam_cfg_dir, exist_ok=True)
            with open(_cam_cfg_file, "w") as _f:
                _f.write(json.dumps({
                    "_class_name": "FSDPWan2_2Transformer3DModel",
                    "_diffusers_version": "0.36.0.dev0",
                    "add_control_adapter": True,
                    "add_ref_conv": False,
                    "attn_compress": 1,
                    "cam_method": "prope",
                    "cam_self_attn_layers": None,
                    "control_adapter_type": "baseline",
                    "cross_attn_norm": True,
                    "cross_attn_type": "cross_attn",
                    "dim": 3072,
                    "downscale_factor_control_adapter": 16,
                    "eps": 1e-06,
                    "ffn_dim": 14336,
                    "freq_dim": 256,
                    "hidden_size": 3072,
                    "in_channels": 48,
                    "in_dim": 48,
                    "in_dim_control_adapter": 24,
                    "in_dim_ref_conv": 16,
                    "model_type": "ti2v",
                    "num_heads": 24,
                    "num_layers": 30,
                    "out_dim": 48,
                    "patch_size": [1, 2, 2],
                    "prope_compress_mode": "none",
                    "prope_compress_ratio": 2,
                    "qk_norm": True,
                    "text_dim": 4096,
                    "text_len": 512,
                    "traning": False,
                    "window_size": [-1, -1],
                }, indent=2))
            print(f"[DreamXModelLoader] wrote wan_models config to {_cam_cfg_file}")

        # DreamX-World's wan_wrapper.py uses relative paths (e.g. ./wan_models/...)
        # that resolve against the repo root. chdir there before construction.
        _prev_cwd = os.getcwd()
        os.chdir(dreamx_dir)
        try:
            print(f"[DreamXModelLoader] initialising pipeline (frames_per_chunk={frames_per_chunk})")
            pipeline = CausalCameraInferencePipeline(
                config,
                device=device,
                num_output_frames=frames_per_chunk,
                model_config_path=model_config_path,
                text_encoder_path=text_encoder_path,
                tokenizer_path=tokenizer_path,
                vae_path=vae_path,
            )
        finally:
            os.chdir(_prev_cwd)

        print(f"[DreamXModelLoader] loading checkpoint: {checkpoint_path}")
        if checkpoint_path.endswith(".safetensors"):
            sd = load_file(checkpoint_path)
            sd = {"model." + k: v for k, v in sd.items()}
        else:
            # .pt with generator_ema / generator key
            raw = torch.load(checkpoint_path, map_location="cpu")
            key = "generator_ema" if "generator_ema" in raw else "generator"
            sd = raw.get(key, raw)

        missing, unexpected = pipeline.generator.load_state_dict(sd, strict=False)
        print(f"[DreamXModelLoader] checkpoint loaded — missing={len(missing)}, unexpected={len(unexpected)}")

        pipeline = pipeline.to(dtype=torch.bfloat16)

        # Mirror DreamX's own low_memory path (triggers for any card < 40 GB).
        # DynamicSwap keeps T5 encoder in CPU RAM and moves individual layers to
        # GPU only during text encoding, freeing ~11.4 GB for generator + KV cache.
        from utils.memory import DynamicSwapInstaller
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)

        pipeline.generator.to(device=device)
        pipeline.vae.to(device=device)

        # Patch vae.decode_to_pixel to evict the generator from VRAM first.
        # After diffusion completes, the generator (10.5 GB bf16) + KV cache
        # leave no room for VAE decode activations (~2.6 GB). Offloading the
        # generator to CPU before each decode and restoring it after costs ~0.3s
        # per chunk (PCIe transfer) but prevents the OOM.
        _gen = pipeline.generator
        _orig_decode = pipeline.vae.decode_to_pixel
        def _decode_with_generator_offload(latents):
            import gc
            # Offload generator (~10.5 GB bf16)
            _gen.cpu()
            # Nuke every CUDA tensor > 100 MB on the pipeline and its children.
            # Attribute-name guessing failed (KV cache not found by name), so
            # scan all object dicts and zero out any large CUDA tensors directly.
            def _free_cuda_tensors(obj, depth=0):
                if obj is None or depth > 3:
                    return
                try:
                    d = vars(obj)
                except TypeError:
                    return
                for k, v in list(d.items()):
                    if isinstance(v, torch.Tensor):
                        if v.device.type == 'cuda' and v.numel() * v.element_size() > 100e6:
                            print(f"[decode-offload] freeing {type(obj).__name__}.{k}: "
                                  f"{v.shape} {v.dtype} "
                                  f"{v.numel()*v.element_size()/1e9:.2f}GB")
                            setattr(obj, k, None)
                    elif isinstance(v, (list, tuple)) and v:
                        # check if it's a list of tensors or dicts-of-tensors (KV cache pattern)
                        first = v[0] if v else None
                        if isinstance(first, torch.Tensor) and first.device.type == 'cuda':
                            total = sum(x.numel()*x.element_size() for x in v if isinstance(x, torch.Tensor))
                            if total > 100e6:
                                print(f"[decode-offload] freeing list {type(obj).__name__}.{k}: "
                                      f"{len(v)} tensors {total/1e9:.2f}GB")
                                setattr(obj, k, None)
                        elif isinstance(first, dict):
                            _free_cuda_tensors(type('_', (), first)(), depth+1)
                    elif hasattr(v, '__dict__') and not isinstance(v, type):
                        _free_cuda_tensors(v, depth+1)

            for _obj in [pipeline, pipeline.generator,
                          getattr(pipeline.generator, 'model', None)]:
                _free_cuda_tensors(_obj)
            gc.collect()
            torch.cuda.empty_cache()
            free_gb = (torch.cuda.get_device_properties(0).total_memory
                       - torch.cuda.memory_allocated()) / 1e9
            print(f"[decode-offload] VRAM free before decode: {free_gb:.1f} GB")
            out = _orig_decode(latents)
            _gen.to(device)
            return out
        pipeline.vae.decode_to_pixel = _decode_with_generator_offload

        _PIPELINE_CACHE[cache_key] = pipeline
        print("[DreamXModelLoader] pipeline ready")
        return (pipeline,)


class DreamXCameraSequence:
    """
    Generates a video by chaining predefined camera moves as rolling chunks.

    Each move in move_sequence triggers one DreamX inference pass.
    The last frame of each chunk becomes the init image for the next,
    maintaining visual continuity across the full sequence.

    move_sequence — comma-separated list of move names, e.g.:
        "forward,arc_left,forward,tilt_up,back"

    Available moves:
        forward, back, strafe_left, strafe_right
        pan_left, pan_right, tilt_up, tilt_down
        arc_left, arc_right, crane_up, crane_down
        retreat_up, retreat_down, static

    frames_per_chunk must match the value used in DreamXModelLoader.
    Pixel frames per chunk = (frames_per_chunk - 1) * 4 + 1.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline":        ("DREAMX_PIPELINE",),
                "init_image":      ("IMAGE",),
                "caption":         ("STRING", {
                    "multiline": True,
                    "default": "Style: Photorealistic.",
                }),
                "move_sequence":   ("STRING", {
                    "multiline": False,
                    "default": "forward,arc_left,forward,static",
                }),
                "frames_per_chunk": ("INT", {
                    "default": 12, "min": 3, "max": 123, "step": 3,
                }),
                "fps":   ("INT", {"default": 16, "min": 8, "max": 30}),
                "seed":  ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
                "color_correction": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "execute"
    CATEGORY = "DreamX"

    def execute(
        self, pipeline, init_image, caption,
        move_sequence, frames_per_chunk, fps, seed, color_correction
    ):
        _ensure_dreamx_on_path()

        from utils.trajectory_processor import generate_trajectory_from_json, Camera
        from utils.misc import set_seed
        from utils.postprocess import postprocess_video_frames

        # Auto-round frames_per_chunk to the nearest multiple of 3.
        # The pipeline asserts num_frames % num_frame_per_block == 0 (block=3),
        # so we silently fix any value the user passes rather than erroring.
        frames_per_chunk = max(3, (frames_per_chunk // 3) * 3)

        device = torch.device("cuda")
        num_pixel_frames = (frames_per_chunk - 1) * 4 + 1

        moves = [m.strip() for m in move_sequence.split(",") if m.strip()]
        if not moves:
            raise ValueError("move_sequence is empty — provide at least one move name.")

        unknown = [m for m in moves if m not in CAMERA_MOVES]
        if unknown:
            raise ValueError(
                f"Unknown move(s): {unknown}. "
                f"Valid moves: {MOVE_NAMES}"
            )

        print(f"[DreamXCameraSequence] {len(moves)} chunk(s): {moves}")

        all_frames = []
        current_pil = _comfy_image_to_pil(init_image)

        for i, move_name in enumerate(moves):
            action_str, speed = CAMERA_MOVES[move_name]
            set_seed(seed + i)

            print(f"[DreamXCameraSequence] chunk {i+1}/{len(moves)}: {move_name!r} (action={action_str!r})")

            # 1. Encode current init image → first latent frame
            img_tensor = _pil_to_dreamx_tensor(current_pil, device)
            initial_latent = pipeline.vae.encode_to_latent(img_tensor).to(
                device=device, dtype=torch.bfloat16
            )

            # 2. Build noise; condition first latent frame on the init image
            noise = torch.randn(
                [1, frames_per_chunk, 48, 44, 80],
                device=device, dtype=torch.bfloat16
            )
            noise[:, 0] = initial_latent

            # 3. Build camera trajectory → PRoPE conditioning dict
            # 'static' uses an empty action string — treat as identity trajectory
            eff_action = action_str if action_str else "w"  # minimal forward for neutral
            trajectory_spec = [(eff_action, speed)]

            _, cam_params_np, _ = generate_trajectory_from_json(
                trajectory_spec=trajectory_spec,
                num_frames=num_pixel_frames,
                return_cam_params=True,
            )
            cam_objects = [
                Camera(cam_params_np[j].tolist())
                for j in range(cam_params_np.shape[0])
            ]
            control_camera = _cam_params_to_prope_dict(cam_objects, device=device)

            # 4. Run inference
            video, _ = pipeline.inference(
                noise=noise,
                text_prompts=[caption],
                y=None,
                y_camera=control_camera,
                return_latents=True,
            )

            # 5. Decode: (1, T, C, H, W) → (1, T, H, W, C), scale to [0, 255]
            video = rearrange(video, "b t c h w -> b t h w c").cpu()
            video = (video * 255.0).clamp(0, 255)

            reference_frame = video[0, 0]
            video = postprocess_video_frames(
                video,
                reference_frame=reference_frame,
                color_correction_strength=color_correction,
            )
            # video[0]: (T, H, W, C) float, ~[0, 255]

            chunk_np = video[0].numpy()
            all_frames.append(_video_tensor_to_comfy(chunk_np))

            # 6. Extract last frame → next init image
            last_np = chunk_np[-1].clip(0, 255).astype(np.uint8)
            current_pil = Image.fromarray(last_np).convert("RGB")

            # Free VAE cache between chunks
            if hasattr(pipeline.vae, "model") and hasattr(pipeline.vae.model, "clear_cache"):
                pipeline.vae.model.clear_cache()

        # Concatenate all chunks along time axis → (total_T, H, W, C)
        combined = torch.cat(all_frames, dim=0)
        print(f"[DreamXCameraSequence] done — {combined.shape[0]} total frames")
        return (combined,)


# ─── PRoPE helper (copied from inference_ar_forcing.py) ─────────────────────
# Duplicated here so the node works even when inference_ar_forcing.py isn't
# directly importable as a module.

def _invert_SE3(mats):
    R_inv = mats[..., :3, :3].transpose(-1, -2)
    result = torch.zeros_like(mats)
    result[..., :3, :3] = R_inv
    result[..., :3, 3] = -torch.einsum("...ij,...j->...i", R_inv, mats[..., :3, 3])
    result[..., 3, 3] = 1.0
    return result

def _get_relative_pose(cam_params):
    abs_w2cs = [cp.w2c_mat for cp in cam_params]
    abs_c2ws = [cp.c2w_mat for cp in cam_params]
    target = np.eye(4, dtype=np.float32)
    abs2rel = target @ abs_w2cs[0]
    return np.array([target] + [abs2rel @ c2w for c2w in abs_c2ws[1:]], dtype=np.float32)

def _cam_params_to_prope_dict(cam_params, device, dtype=torch.bfloat16):
    """Camera objects → PRoPE conditioning dict {viewmats, K}."""
    num_frames = len(cam_params)
    latent_count = 1 + (num_frames - 1) // 4
    indices = [0] + [1 + 4 * i for i in range(latent_count - 1)]
    sub = [cam_params[i] for i in indices]

    c2w_poses = _get_relative_pose(sub)
    c2ws = torch.as_tensor(c2w_poses, dtype=dtype, device=device)
    viewmats = _invert_SE3(c2ws)                                   # (T_lat, 4, 4)
    viewmats = viewmats.unsqueeze(1).expand(-1, 880, -1, -1).reshape(1, -1, 4, 4)

    # Fixed normalised intrinsics matching DreamX training config
    fx_norm = 969.6969696969696 / (960.0 * 2)
    fy_norm = 969.6969696969696 / (540.0 * 2)
    K = torch.zeros((1, 3, 3), dtype=dtype, device=device)
    K[:, 0, 0] = fx_norm
    K[:, 1, 1] = fy_norm
    K[:, 0, 2] = 0.5
    K[:, 1, 2] = 0.5
    K[:, 2, 2] = 1.0
    K = K.unsqueeze(1).expand(-1, viewmats.shape[1], -1, -1).reshape(1, -1, 3, 3)

    return {"viewmats": viewmats, "K": K}


# ─── registration ────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "DreamXModelLoader":     DreamXModelLoader,
    "DreamXCameraSequence":  DreamXCameraSequence,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamXModelLoader":     "DreamX Model Loader",
    "DreamXCameraSequence":  "DreamX Camera Sequence",
}
