#!/usr/bin/env python
"""Install a matching prebuilt flash-attn wheel for the active torch runtime."""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

GITHUB_REPO = "Dao-AILab/flash-attention"
GITHUB_API_RELEASE_LATEST = (
    f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
)
DEFAULT_RELEASE_TAG = "v2.8.3"


@dataclass
class RuntimeInfo:
    torch_version: str
    torch_major_minor: str
    torch_cuda: str
    python_tag: str
    cxx11abi: str
    platform_tag: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install a prebuilt flash-attn wheel matching the active "
            "torch/python/ABI runtime."
        )
    )
    parser.add_argument(
        "--release-tag",
        default=None,
        help=(
            "FlashAttention release tag to install from (e.g., v2.8.3). "
            "Default: latest GitHub release, with fallback to v2.8.3."
        ),
    )
    parser.add_argument(
        "--print-url",
        action="store_true",
        help="Only print the resolved wheel URL and exit.",
    )
    parser.add_argument(
        "--installer",
        choices=("auto", "uv", "pip"),
        default="auto",
        help="Installer backend. auto prefers uv when available.",
    )
    return parser.parse_args()


def _parse_torch_major_minor(torch_version: str) -> str:
    match = re.match(r"^(\d+)\.(\d+)", torch_version)
    if not match:
        raise RuntimeError(f"Unable to parse torch version: {torch_version}")
    return f"{match.group(1)}.{match.group(2)}"


def _platform_tag() -> str:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Prebuilt flash-attn wheel install is supported on Linux only.")

    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "linux_x86_64"
    if machine in {"aarch64", "arm64"}:
        return "linux_aarch64"
    raise RuntimeError(f"Unsupported machine architecture: {machine}")


def _get_latest_release_tag() -> str:
    req = urllib.request.Request(
        GITHUB_API_RELEASE_LATEST,
        headers={"Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    tag_name = payload.get("tag_name")
    if not tag_name:
        raise RuntimeError("GitHub API response did not include tag_name.")
    return tag_name


def _gather_runtime() -> RuntimeInfo:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "torch must be installed in this environment before installing flash-attn."
        ) from exc

    torch_version = torch.__version__
    torch_major_minor = _parse_torch_major_minor(torch_version)
    torch_cuda = torch.version.cuda or ""
    if not torch_cuda:
        raise RuntimeError("CUDA runtime not detected in torch (torch.version.cuda is empty).")
    if not torch_cuda.startswith("12."):
        raise RuntimeError(
            f"Unsupported torch CUDA runtime {torch_cuda}. "
            "This script currently targets flash-attn cu12 wheels."
        )

    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    cxx11abi = "TRUE" if bool(torch._C._GLIBCXX_USE_CXX11_ABI) else "FALSE"

    return RuntimeInfo(
        torch_version=torch_version,
        torch_major_minor=torch_major_minor,
        torch_cuda=torch_cuda,
        python_tag=python_tag,
        cxx11abi=cxx11abi,
        platform_tag=_platform_tag(),
    )


def _wheel_url(release_tag: str, runtime: RuntimeInfo) -> str:
    version = release_tag.lstrip("v")
    filename = (
        f"flash_attn-{version}%2Bcu12torch{runtime.torch_major_minor}"
        f"cxx11abi{runtime.cxx11abi}-{runtime.python_tag}-{runtime.python_tag}-"
        f"{runtime.platform_tag}.whl"
    )
    return f"https://github.com/{GITHUB_REPO}/releases/download/{release_tag}/{filename}"


def _wheel_exists(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=15):
            return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def _install_cmd(installer: str, wheel_url: str) -> list[str]:
    if installer == "auto":
        installer = "uv" if shutil.which("uv") else "pip"

    if installer == "uv":
        return ["uv", "pip", "install", "--force-reinstall", "--no-deps", wheel_url]
    return [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", wheel_url]


def main() -> int:
    args = _parse_args()
    runtime = _gather_runtime()

    release_tag = args.release_tag
    if release_tag is None:
        try:
            release_tag = _get_latest_release_tag()
            print(f"Using latest release tag: {release_tag}")
        except Exception as exc:
            release_tag = DEFAULT_RELEASE_TAG
            print(
                f"Could not resolve latest release tag ({type(exc).__name__}: {exc}). "
                f"Falling back to {DEFAULT_RELEASE_TAG}."
            )

    wheel_url = _wheel_url(release_tag, runtime)
    print(f"torch={runtime.torch_version} cuda={runtime.torch_cuda} python={runtime.python_tag}")
    print(f"Resolved wheel: {wheel_url}")

    if not _wheel_exists(wheel_url):
        print("No matching prebuilt wheel found for this runtime.")
        print(
            "Try a different --release-tag, or align python/torch to a runtime "
            "with published flash-attn wheels."
        )
        return 2

    if args.print_url:
        return 0

    cmd = _install_cmd(args.installer, wheel_url)
    print("Installing via:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
