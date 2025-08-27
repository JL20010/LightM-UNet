import sys
import types
import pytest

try:  # pragma: no cover - handled gracefully if torch is missing
    import torch
except Exception:  # pragma: no cover
    pytest.skip("torch is required for this test", allow_module_level=True)

# Create a dummy mamba_ssm module so that LightMUNet can be imported
mamba_ssm = types.ModuleType("mamba_ssm")

class DummyMamba(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

mamba_ssm.Mamba = DummyMamba
sys.modules["mamba_ssm"] = mamba_ssm

from nnunetv2.nets.LightMUNet import MambaLayer


def test_mambalayer_preserves_input_dtype():
    layer = MambaLayer(input_dim=4, output_dim=4)
    x = torch.randn(2, 4, 3, 3, dtype=torch.float16)
    out = layer(x)
    assert out.dtype == x.dtype
