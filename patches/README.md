# Local dependency patches

These patches are snapshots of changes made to third-party checkouts for this
machine. They are kept here for reference and are **not** applied automatically.

## `mamba-forward-only.patch`

Applies to `state-spaces/mamba` (`setup.py`) and makes two changes:

1. Adds `-allow-unsupported-compiler` to the CUDA build flags.
2. Removes all backward `.cu` sources from the `selective_scan_cuda` extension,
   leaving a **forward-only** build.

> **Warning:** with the backward kernels removed, the extension is suitable for
> inference only. Training through Mamba will fail when gradients are needed.

Apply from the mamba checkout with:

```bash
git apply /path/to/mamba-forward-only.patch
```
