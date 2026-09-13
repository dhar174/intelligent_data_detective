import json
import os
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.annotations import FreeText

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


# Fills a PDF by adding text annotations defined in `fields.json`. See forms.md.


def transform_coordinates(bbox, image_width, image_height, pdf_width, pdf_height):
    """Transform bounding box from image coordinates to PDF coordinates"""
    # Image coordinates: origin at top-left, y increases downward
    # PDF coordinates: origin at bottom-left, y increases upward
    x_scale = pdf_width / image_width
    y_scale = pdf_height / image_height
    
    left = bbox[0] * x_scale
    right = bbox[2] * x_scale
    
    # Flip Y coordinates for PDF
    top = pdf_height - (bbox[1] * y_scale)
    bottom = pdf_height - (bbox[3] * y_scale)
    
    return left, bottom, right, top


def fill_pdf_form(input_pdf_path, fields_json_path, output_pdf_path):
    """Fill the PDF form with data from fields.json"""
    
    # `fields.json` format described in forms.md.
    with open(fields_json_path, "r") as f:
        fields_data = json.load(f)
    
    # Open the PDF
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()
    
    # Copy all pages to writer
    writer.append(reader)
    
    # Get PDF dimensions for each page
    pdf_dimensions = {}
    for i, page in enumerate(reader.pages):
        mediabox = page.mediabox
        pdf_dimensions[i + 1] = [mediabox.width, mediabox.height]
    
    # Process each form field
    annotations = []
    for field in fields_data["form_fields"]:
        page_num = field["page_number"]
        
        # Get page dimensions and transform coordinates.
        page_info = next(p for p in fields_data["pages"] if p["page_number"] == page_num)
        image_width = page_info["image_width"]
        image_height = page_info["image_height"]
        pdf_width, pdf_height = pdf_dimensions[page_num]
        
        transformed_entry_box = transform_coordinates(
            field["entry_bounding_box"],
            image_width, image_height,
            pdf_width, pdf_height
        )
        
        # Skip empty fields
        if "entry_text" not in field or "text" not in field["entry_text"]:
            continue
        entry_text = field["entry_text"]
        text = entry_text["text"]
        if not text:
            continue
        
        font_name = entry_text.get("font", "Arial")
        font_size = str(entry_text.get("font_size", 14)) + "pt"
        font_color = entry_text.get("font_color", "000000")

        # Font size/color seems to not work reliably across viewers:
        # https://github.com/py-pdf/pypdf/issues/2084
        annotation = FreeText(
            text=text,
            rect=transformed_entry_box,
            font=font_name,
            font_size=font_size,
            font_color=font_color,
            border_color=None,
            background_color=None,
        )
        annotations.append(annotation)
        # page_number is 0-based for pypdf
        writer.add_annotation(page_number=page_num - 1, annotation=annotation)
        
    # Save the filled PDF via artifact resolver
    resolved_output_path = _resolve_artifact_path(output_pdf_path, config=None)
    with open(resolved_output_path, "wb") as output:
        writer.write(output)
    
    print(f"Successfully filled PDF form and saved to {resolved_output_path}")
    print(f"Added {len(annotations)} text annotations")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: fill_pdf_form_with_annotations.py [input pdf] [fields.json] [output pdf]")
        sys.exit(1)
    input_pdf = sys.argv[1]
    fields_json = sys.argv[2]
    output_pdf = sys.argv[3]
    
    fill_pdf_form(input_pdf, fields_json, output_pdf)