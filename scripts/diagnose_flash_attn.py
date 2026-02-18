#!/usr/bin/env python
"""Diagnose flash-attn install/runtime compatibility for this environment."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from typing import Any


def _try_import(module_name: str) -> tuple[bool, Any, Exception | None]:
    try:
        module = importlib.import_module(module_name)
        return True, module, None
    except Exception as exc:  # pragma: no cover - utility script for env checks
        return False, None, exc


def _cmd_output(cmd: list[str]) -> str:
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return output.strip()
    except Exception as exc:  # pragma: no cover - utility script for env checks
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def _print_runtime() -> None:
    print(f"python: {sys.version.replace(chr(10), ' ')}")
    print(f"executable: {sys.executable}")
    print(f"platform: {sys.platform}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', '<unset>')}")
    print(f"nvcc: {_cmd_output(['nvcc', '--version']).splitlines()[-1]}")
    print(f"nvidia-smi: {_cmd_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']).splitlines()[0]}")


def main() -> int:
    _print_runtime()

    ok_torch, torch, torch_exc = _try_import("torch")
    if not ok_torch:
        print(f"torch import failed: {type(torch_exc).__name__}: {torch_exc}")
        return 1

    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")

    ok_flash, flash_attn, flash_exc = _try_import("flash_attn")
    if not ok_flash:
        print(f"flash_attn import failed: {type(flash_exc).__name__}: {flash_exc}")
        print("Diagnosis: flash-attn is missing or not loadable in this environment.")
        return 2

    print(f"flash_attn: {getattr(flash_attn, '__version__', '<unknown>')}")

    ok_ext, _, ext_exc = _try_import("flash_attn_2_cuda")
    if not ok_ext:
        msg = f"{type(ext_exc).__name__}: {ext_exc}"
        print(f"flash_attn_2_cuda import failed: {msg}")
        if "undefined symbol" in str(ext_exc):
            print(
                "Diagnosis: ABI mismatch. flash-attn was built against a different "
                f"PyTorch/CUDA runtime than the current one (torch={torch.__version__}, cuda={torch.version.cuda})."
            )
            print("Fix: reinstall flash-attn in this exact env after torch is finalized.")
            print("Command: uv run python scripts/install_prebuilt_flash_attn.py")
        return 3

    print("flash_attn_2_cuda import succeeded.")
    print("flash-attn looks ABI-compatible with this runtime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
