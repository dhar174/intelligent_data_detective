import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw


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


# Creates "validation" images with rectangles for the bounding box information that
# Claude creates when determining where to add text annotations in PDFs. See forms.md.


def create_validation_image(page_number, fields_json_path, input_path, output_path):
    # Input file should be in the `fields.json` format described in forms.md.
    with open(fields_json_path, 'r') as f:
        data = json.load(f)

        img = Image.open(input_path)
        draw = ImageDraw.Draw(img)
        num_boxes = 0
        
        for field in data["form_fields"]:
            if field["page_number"] == page_number:
                entry_box = field['entry_bounding_box']
                label_box = field['label_bounding_box']
                # Draw red rectangle over entry bounding box and blue rectangle over the label.
                draw.rectangle(entry_box, outline='red', width=2)
                draw.rectangle(label_box, outline='blue', width=2)
                num_boxes += 2
        
        resolved_output_path = _resolve_artifact_path(str(output_path), config=None)
        img.save(resolved_output_path)
        print(f"Created validation image at {resolved_output_path} with {num_boxes} bounding boxes")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: create_validation_image.py [page number] [fields.json file] [input image path] [output image path]")
        sys.exit(1)
    page_number = int(sys.argv[1])
    fields_json_path = safe_user_path(sys.argv[2])
    input_image_path = safe_user_path(sys.argv[3])
    output_image_path = sys.argv[4]
    create_validation_image(page_number, fields_json_path, input_image_path, output_image_path)
