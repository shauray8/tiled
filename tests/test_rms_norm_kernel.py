import pytest
import torch
import torch.nn.functional as F
from tests.conftest import skip_no_cuda
from tiled.kernels.rms_norm import rms_norm, _fwd_launch, _bwd_launch

DEVICE = "cuda"


@skip_no_cuda
@pytest.mark.parametrize(
  "M,N,dtype",
  [
    (128, 1024, torch.bfloat16),
    (512, 3840, torch.bfloat16),
    (4224, 3840, torch.bfloat16),
    (4224, 7680, torch.bfloat16),
    (256, 4096, torch.bfloat16),
  ],
)
def test_fwd(M, N, dtype):
  x = torch.randn(M, N, device=DEVICE, dtype=dtype)
  w = torch.randn(N, device=DEVICE, dtype=dtype)
  out, _ = _fwd_launch(x, w, 1e-5)
  ref = F.rms_norm(x.float(), [N], weight=w.float()).to(dtype)
  torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@skip_no_cuda
@pytest.mark.parametrize("M,N", [(128, 1024), (512, 3840), (1024, 4096)])
def test_bwd(M, N):
  x = torch.randn(M, N, device=DEVICE, dtype=torch.bfloat16).requires_grad_(True)
  w = torch.randn(N, device=DEVICE, dtype=torch.bfloat16).requires_grad_(True)
  xr = x.detach().float().requires_grad_(True)
  wr = w.detach().float().requires_grad_(True)
  F.rms_norm(xr, [N], weight=wr).sum().backward()
  rms_norm(x, w).float().sum().backward()
  torch.testing.assert_close(x.grad.float(), xr.grad, atol=5e-2, rtol=5e-2)
  torch.testing.assert_close(w.grad.float(), wr.grad, atol=5e-2, rtol=5e-2)


@skip_no_cuda
def test_autograd_grads_not_none():
  x = torch.randn(64, 1024, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
  w = torch.randn(1024, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
  rms_norm(x, w).sum().backward()
  assert x.grad is not None and w.grad is not None


@skip_no_cuda
def test_eps_zero_input_stable():
  x = torch.zeros(16, 512, device=DEVICE, dtype=torch.bfloat16)
  w = torch.ones(512, device=DEVICE, dtype=torch.bfloat16)
  out, _ = _fwd_launch(x, w, 1e-5)
  assert not out.isnan().any()

