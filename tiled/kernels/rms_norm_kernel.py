import torch
import cuda.tile as ct
from tiled._configs import OCC_FWD, OCC_BWD
from tiled._types import ConstInt, ConstFloat
from tiled._utils import next_power_of_2, cdiv

def _fwd_tiles(N: int) -> tuple[int, int]:
  tn = next_power_of_2(N)
  tm = 16 if tn <= 1024 else (2 if tn >= 16384 else 4)
  return tm, tn

def _bwd_tiles(N: int) -> tuple[int, int]:
  # backward holds dy+x simultaneously: 2*TILE_M*TILE_N*4 bytes
  tn = next_power_of_2(N)
  if tn <= 1024:
    tm = 4
  elif tn <= 4096:
    tm = 2
  else:
    tm = 1
  return tm, tn

@ct.kernel(occupancy=OCC_FWD, opt_level=3)
def _fwd(X, W, Out, Rstd, TILE_M: ConstInt, TILE_N: ConstInt, eps: ConstFloat):
  bid, N = ct.bid(0), X.shape[1]
  w = ct.load(W, index=(0,), shape=(TILE_N,), latency=1).astype(ct.float32)
  w2d = ct.reshape(w, (1, TILE_N))
  nb = ct.num_blocks(0)
  for i in range(bid, ct.cdiv(X.shape[0], TILE_M), nb):
    x = ct.load(
      X, index=(i, 0), shape=(TILE_M, TILE_N), padding_mode=ct.PaddingMode.ZERO, latency=10
    ).astype(ct.float32)
    rstd = ct.rsqrt(ct.sum(x * x, axis=1, keepdims=True) / N + eps)
    ct.store(Rstd, index=(i,), tile=ct.reshape(rstd, (TILE_M,)), allow_tma=False, latency=1)
    ct.store(Out, index=(i, 0), tile=(x * rstd * w2d).astype(X.dtype), allow_tma=False, latency=3)

@ct.kernel(occupancy=OCC_BWD, opt_level=3)
def _bwd(dY, X, W, Rstd, dX, dW_part, TILE_M: ConstInt, TILE_N: ConstInt):
  bid, N = ct.bid(0), X.shape[1]
  w = ct.load(W, index=(0,), shape=(TILE_N,), latency=1).astype(ct.float32)
  w2d = ct.reshape(w, (1, TILE_N))
  nb = ct.num_blocks(0)
  for i in range(bid, ct.cdiv(X.shape[0], TILE_M), nb):
    dy = ct.load(
      dY, index=(i, 0), shape=(TILE_M, TILE_N), padding_mode=ct.PaddingMode.ZERO, latency=8
    ).astype(ct.float32)
    x = ct.load(
      X, index=(i, 0), shape=(TILE_M, TILE_N), padding_mode=ct.PaddingMode.ZERO, latency=8
    ).astype(ct.float32)
    rstd = ct.reshape(
      ct.load(Rstd, index=(i,), shape=(TILE_M,), latency=4).astype(ct.float32), (TILE_M, 1)
    )
    xn = x * rstd
    dyw = dy * w2d
    dot = ct.sum(dyw * xn, axis=1, keepdims=True) / N
    ct.store(
      dX, index=(i, 0), tile=(rstd * (dyw - xn * dot)).astype(dY.dtype), allow_tma=False, latency=3
    )
    ct.store(dW_part, index=(i, 0), tile=(dy * xn).astype(ct.float32), allow_tma=False, latency=3)

def _fwd_launch(x: torch.Tensor, w: torch.Tensor, eps: float):
  M, N = x.shape
  tm, tn = _fwd_tiles(N)
  out = torch.empty_like(x)
  rstd = torch.empty(M, device=x.device, dtype=torch.float32)
  nsm = torch.cuda.get_device_properties(x.device).multi_processor_count
  ct.launch(
    torch.cuda.current_stream(), (min(nsm, cdiv(M, tm)),), _fwd, (x, w, out, rstd, tm, tn, eps)
  )
  return out, rstd

def _bwd_launch(dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor, rstd: torch.Tensor):
  M, N = x.shape
  tm, tn = _bwd_tiles(N)
  dx = torch.empty_like(x)
  dw_part = torch.empty(M, tn, device=x.device, dtype=torch.float32)
  nsm = torch.cuda.get_device_properties(x.device).multi_processor_count
  ct.launch(
    torch.cuda.current_stream(),
    (min(nsm, cdiv(M, tm)),),
    _bwd,
    (dy, x, w, rstd, dx, dw_part, tm, tn),
  )
  return dx, dw_part[:, :N].sum(0).to(w.dtype)

class _Fn(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, w, eps):
    out, rstd = _fwd_launch(x, w, eps)
    ctx.save_for_backward(x, w, rstd)
    ctx.eps = eps
    return out

  @staticmethod
  def backward(ctx, dy):
    dx, dw = _bwd_launch(dy.contiguous(), *ctx.saved_tensors[:2], ctx.saved_tensors[2])
    return dx, dw, None

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
  return _Fn.apply(x, w, eps)
