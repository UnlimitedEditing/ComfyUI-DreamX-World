import traceback

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print(f"[ComfyUI-DreamX-World] {len(NODE_CLASS_MAPPINGS)} nodes registered")
except Exception as e:
    print(f"[ComfyUI-DreamX-World] WARNING: failed to load nodes: {e}")
    traceback.print_exc()
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
