"""
idd_bootstrap.py — Provenance manifest verification and environment bootstrap
for Intelligent Data Detective canonical Python modules.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Optional

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

MANIFEST_FILENAME = "idd_manifest.json"
CANONICAL_MODULES = [
    "idd_models.py",
    "idd_registry.py",
    "idd_state.py",
    "idd_tools.py",
    "idd_graph.py",
    "idd_core.py",
    "idd_bootstrap.py",
]


def compute_sha256(path: Path | str) -> str:
    """Compute the SHA-256 hex digest of a canonical module with normalized newline bytes.

    Normalizing CRLF to LF ensures cryptographic digests are deterministic across
    Windows, Linux, and macOS platforms regardless of git line ending configuration.
    """
    with open(path, "rb") as f:
        content = f.read()
    normalized = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def find_repo_root(start_dir: Optional[Path | str] = None) -> Optional[Path]:
    """Find repository root containing idd_manifest.json starting from start_dir or cwd."""
    cur = Path(start_dir or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / MANIFEST_FILENAME).is_file():
            return parent
    return None


def verify_manifest(
    repo_root: Optional[Path | str] = None,
    manifest_path: Optional[Path | str] = None,
) -> Dict[str, str]:
    """Verify that canonical modules match the expected SHA-256 digests in idd_manifest.json.

    Args:
        repo_root: Directory containing canonical modules. If omitted, discovered via find_repo_root().
        manifest_path: Path to idd_manifest.json. If omitted, repo_root / idd_manifest.json.

    Returns:
        Dict of module name -> verified SHA-256 hex digest.

    Raises:
        FileNotFoundError: If manifest or any canonical module is missing.
        RuntimeError: If SHA-256 digest does not match manifest.
    """
    root = Path(repo_root).resolve() if repo_root else find_repo_root()
    if root is None and manifest_path is None:
        raise FileNotFoundError(
            f"Cannot locate {MANIFEST_FILENAME}. Ensure repository root is on sys.path or provide repo_root."
        )

    mpath = Path(manifest_path).resolve() if manifest_path else root / MANIFEST_FILENAME
    if not mpath.is_file():
        raise FileNotFoundError(f"Manifest not found: {mpath}")

    with open(mpath, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    expected_files: Dict[str, str] = (
        manifest_data.get("files") or manifest_data.get("modules") or {}
    )
    verified: Dict[str, str] = {}

    for module_name in CANONICAL_MODULES:
        if module_name not in expected_files:
            raise RuntimeError(
                f"Manifest {mpath} is missing required canonical entry: '{module_name}'."
            )

        module_path = (root if root else mpath.parent) / module_name
        if not module_path.is_file():
            raise FileNotFoundError(
                f"Canonical module '{module_name}' missing from {module_path.parent}."
            )

        actual_sha = compute_sha256(module_path)
        expected_sha = expected_files[module_name]

        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Digest mismatch for '{module_name}' in {module_path.parent}!\n"
                f"  Expected: {expected_sha}\n"
                f"  Actual:   {actual_sha}\n"
                "Refusing to import untrusted or corrupted canonical module."
            )

        verified[module_name] = actual_sha

    return verified


def bootstrap_environment(
    start_dir: Optional[Path | str] = None,
    allow_unverified: bool = False,
    *,
    repo_root: Optional[Path | str] = None,
    manifest_path: Optional[Path | str] = None,
) -> Path:
    """Bootstrap the runtime environment by discovering repo root, adding to sys.path, and verifying manifest.

    Returns:
        Path to verified repository root.

    Raises:
        RuntimeError: If running in standalone ephemeral Colab without cloned repo,
                      or if manifest verification fails.
    """
    is_colab = False
    try:
        import google.colab  # type: ignore # noqa: F401

        is_colab = True
    except Exception:
        pass

    resolved_root: Optional[Path] = (
        Path(repo_root).resolve() if repo_root else find_repo_root(start_dir)
    )

    if resolved_root is None:
        if is_colab:
            raise RuntimeError(
                "IDD Bootstrap Error: Running in Google Colab without an explicitly cloned repository.\n"
                "Standalone ephemeral Colab module acquisition requires an approved distribution mechanism\n"
                "(e.g., pinned release asset or package wheel per ADR Item 4, currently marked REQUIRES DECISION).\n"
                "To run on Colab, clone the repository first:\n"
                "    !git clone https://github.com/dhar174/intelligent_data_detective.git\n"
                "    %cd intelligent_data_detective"
            )
        else:
            raise RuntimeError(
                f"IDD Bootstrap Error: Could not locate repository root with {MANIFEST_FILENAME}."
            )

    if not allow_unverified:
        verify_manifest(repo_root=resolved_root, manifest_path=manifest_path)

    root_str = str(resolved_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    return resolved_root


if __name__ == "__main__":
    try:
        verified = verify_manifest()
        print(
            f"✅ Manifest verified successfully ({len(verified)} canonical modules match digests):"
        )
        for k, v in verified.items():
            print(f"   {k}: {v[:16]}...")
    except Exception as exc:
        print(f"❌ Manifest verification failed: {exc}", file=sys.stderr)
        sys.exit(1)
