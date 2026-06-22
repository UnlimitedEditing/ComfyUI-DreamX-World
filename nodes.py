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
                    "default": 13, "min": 7, "max": 123, "step": 2,
                    "display": "number",
                }),
            }
        }

    RETURN_TYPES = ("DREAMX_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "DreamX"

    def load(self, wan_base_path, checkpoint_path, frames_per_chunk):
        cache_key = (wan_base_path, checkpoint_path, frames_per_chunk)
        if cache_key in _PIPELINE_CACHE:
            print("[DreamXModelLoader] using cached pipeline")
            return (_PIPELINE_CACHE[cache_key],)

        dreamx_dir = _ensure_dreamx_on_path()

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
        pipeline.text_encoder.to(device=device)
        pipeline.generator.to(device=device)
        pipeline.vae.to(device=device)

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
                    "default": 13, "min": 7, "max": 123, "step": 2,
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
