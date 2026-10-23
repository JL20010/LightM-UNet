import torch
import os
import ctypes


def enable_tf32() -> bool:
    """Enable TF32 fast matmul where available (Ampere+). Idempotent and safe.

    Returns True if settings were applied, False otherwise.
    """
    try:
        if torch.cuda.is_available():
            # Matmul/conv: allow TF32 for FP32 ops
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Prefer higher precision policy for FP32 matmul (still allows TF32)
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass
            return True
    except Exception:
        pass
    return False


def preload_torch_cuda_libs() -> bool:
    """Preload Torch shared libs (c10, libtorch, etc.) so third-party CUDA extensions
    can resolve symbols like c10::Warning at import time. No-op if not CUDA.

    Returns True if at least one library was preloaded, False otherwise.
    """
    loaded_any = False
    try:
        if not torch.cuda.is_available():
            return False
        torch_dir = os.path.dirname(torch.__file__)
        lib_dir = os.path.join(torch_dir, 'lib')
        if not os.path.isdir(lib_dir):
            return False
        # Load a few common libs with RTLD_GLOBAL so their symbols are globally visible
        libs = [
            'libc10.so', 'libtorch.so', 'libtorch_cpu.so',
            # Load CUDA libs best-effort; silently ignore if not present
            'libtorch_cuda.so', 'libc10_cuda.so'
        ]
        for lib in libs:
            p = os.path.join(lib_dir, lib)
            if os.path.isfile(p):
                try:
                    ctypes.CDLL(p, mode=getattr(ctypes, 'RTLD_GLOBAL', os.RTLD_GLOBAL))
                    loaded_any = True
                except Exception:
                    # ignore and continue
                    pass
    except Exception:
        return loaded_any
    return loaded_any


def apply_allocator_tuning() -> bool:
    """Set sane CUDA allocator defaults to help reduce fragmentation/spikes.

    Returns True if the environment variable was set by this function, False if it
    was already present or CUDA is unavailable. This should be called early in
    process startup and before the first CUDA tensor is allocated.
    """
    try:
        # Only set if not provided by user/environment
        if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
            # Conservative, portable defaults
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
                'max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True'
            )
            return True
    except Exception:
        pass
    return False
