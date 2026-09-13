import os
import sys
from pathlib import Path

from pdf2image import convert_from_path


def safe_user_path(path_value, base_dir="."):
    """Resolve a CLI path under the current workspace."""
    if base_dir != ".":
        raise ValueError("Custom base directories are not supported for CLI paths")
    base_path = Path.cwd().resolve()
    resolved_path = Path(path_value).expanduser().resolve()
    try:
        resolved_path.relative_to(base_path)
    except ValueError as exc:
        raise ValueError(f"Path escapes allowed directory: {path_value}") from exc
    return resolved_path


def _get_artifacts_base(config=None) -> Path:
    """Resolve the active runtime or configured artifacts directory."""
    if config:
        try:
            cfg = getattr(config, "configurable", None) or (config if isinstance(config, dict) else {})
            runtime = cfg.get("runtime") if isinstance(cfg, dict) else None
            if runtime is not None and getattr(runtime, "artifacts_dir", None):
                base = Path(runtime.artifacts_dir).resolve()
                base.mkdir(parents=True, exist_ok=True)
                return base
        except Exception:
            pass

    for mod_name in ("__main__", "idd_core", "intelligentdatadetective_beta_v5"):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            runtime = getattr(mod, "RUNTIME", None)
            if runtime is not None and getattr(runtime, "artifacts_dir", None):
                base = Path(runtime.artifacts_dir).resolve()
                base.mkdir(parents=True, exist_ok=True)
                return base

    env_dir = os.environ.get("IDD_ARTIFACTS_DIR")
    if env_dir:
        base = Path(env_dir).resolve()
        base.mkdir(parents=True, exist_ok=True)
        return base

    base = (Path.cwd() / "artifacts").resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _resolve_artifact_path(
    file_name: str,
    *,
    config=None,
    subdir: str | None = None,
    create_parents: bool = True,
) -> Path:
    """Resolve file_name within the active runtime/configured artifacts directory."""
    if not file_name or not isinstance(file_name, str):
        raise ValueError("file_name must be a non-empty string.")

    base = _get_artifacts_base(config)
    if subdir:
        base = (base / subdir).resolve()

    base.mkdir(parents=True, exist_ok=True)

    candidate = Path(file_name).expanduser()
    if candidate.is_absolute():
        path = candidate.resolve()
    else:
        cwd_candidate = (Path.cwd() / candidate).resolve()
        try:
            cwd_candidate.relative_to(base)
            path = cwd_candidate
        except ValueError:
            path = (base / candidate).resolve()

    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"Refusing to access path outside artifacts root ({base}): {path}") from exc

    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


# Converts each page of a PDF to a PNG image.


def convert(pdf_path, output_dir, max_dim=1000, dpi=200):
    resolved_output_dir = _resolve_artifact_path(str(output_dir), config=None)
    os.makedirs(resolved_output_dir, exist_ok=True)
    images = convert_from_path(str(pdf_path), dpi=dpi)

    for i, image in enumerate(images):
        # Scale image if needed to keep width/height under `max_dim`
        width, height = image.size
        scale_factor = 1.0
        if width > max_dim or height > max_dim:
            scale_factor = min(max_dim / width, max_dim / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height))
        
        effective_dpi = dpi * scale_factor
        image_path = os.path.join(resolved_output_dir, f"page_{i+1}.png")
        image.save(image_path)
        print(f"Saved page {i+1} as {image_path} (size: {image.size}, dpi: {effective_dpi:.1f})")

    print(f"Converted {len(images)} pages to PNG images")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: convert_pdf_to_images.py [input pdf] [output directory]")
        sys.exit(1)
    pdf_path = safe_user_path(sys.argv[1])
    output_directory = sys.argv[2]
    convert(pdf_path, output_directory)
