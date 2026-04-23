import pytest
import torch


def _gpu_available() -> bool:
  try:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0
  except Exception:
    return False


skip_no_cuda = pytest.mark.skipif(not _gpu_available(), reason="No CUDA GPU available")


@pytest.fixture(scope="session")
def device() -> str:
  return "cuda" if _gpu_available() else "cpu"


@pytest.fixture(scope="session")
def sm_version() -> str:
  if not _gpu_available():
    return "unknown"
  major, minor = torch.cuda.get_device_capability(0)
  return f"sm_{major}{minor}0"


ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-2, torch.float16: 1e-3}
RTOL = {torch.float32: 1e-5, torch.bfloat16: 1e-2, torch.float16: 1e-3}


@pytest.fixture
def tols():
  return {"atol": ATOL, "rtol": RTOL}

