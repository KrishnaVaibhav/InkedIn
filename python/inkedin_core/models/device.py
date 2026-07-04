"""Runtime device selection per Research.MD: preference -> probe -> warmup -> fallback."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeviceInfo:
    device: str  # torch device string: "cuda", "cpu", "mps"
    name: str
    vram_total_mb: int | None = None


def probe(preference: str = "auto") -> DeviceInfo:
    import torch

    pref = preference.lower()
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info = DeviceInfo("cuda", props.name, props.total_memory // (1 << 20))
        if _warmup_ok("cuda"):
            return info
    if pref in ("auto", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        if _warmup_ok("mps"):
            return DeviceInfo("mps", "Apple MPS")
    return DeviceInfo("cpu", "CPU")


def _warmup_ok(device: str) -> bool:
    import torch

    try:
        x = torch.ones(8, 8, device=device)
        y = (x @ x).sum().item()
        return y == 512.0
    except Exception:
        return False
