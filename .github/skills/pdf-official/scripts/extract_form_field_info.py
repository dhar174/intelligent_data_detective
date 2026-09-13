import json
import os
import sys
from pathlib import Path

from pypdf import PdfReader

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


# Extracts data for the fillable form fields in a PDF and outputs JSON that
# Claude uses to fill the fields. See forms.md.


# This matches the format used by PdfReader `get_fields` and `update_page_form_field_values` methods.
def get_full_annotation_field_id(annotation):
    components = []
    while annotation:
        if hasattr(annotation, "get_object"):
            annotation = annotation.get_object()
        if not isinstance(annotation, dict):
            break
        field_name = annotation.get('/T')
        if field_name:
            components.append(field_name)
        parent = annotation.get('/Parent')
        annotation = parent
    return ".".join(reversed(components)) if components else None


def make_field_dict(field, field_id):
    field_dict = {"field_id": field_id}
    ft = field.get('/FT')
    if ft == "/Tx":
        field_dict["type"] = "text"
    elif ft == "/Btn":
        field_dict["type"] = "checkbox"  # radio groups handled separately
        states = field.get("/_States_", [])
        if len(states) == 2:
            # "/Off" seems to always be the unchecked value, as suggested by
            # https://opensource.adobe.com/dc-acrobat-sdk-docs/standards/pdfstandards/pdf/PDF32000_2008.pdf#page=448
            # It can be either first or second in the "/_States_" list.
            if "/Off" in states:
                field_dict["checked_value"] = states[0] if states[0] != "/Off" else states[1]
                field_dict["unchecked_value"] = "/Off"
            else:
                print(f"Unexpected state values for checkbox `${field_id}`. Its checked and unchecked values may not be correct; if you're trying to check it, visually verify the results.")
                field_dict["checked_value"] = states[0]
                field_dict["unchecked_value"] = states[1]
    elif ft == "/Ch":
        field_dict["type"] = "choice"
        states = field.get("/_States_", [])
        choice_options = []
        for state in states:
            if isinstance(state, (list, tuple)) and len(state) >= 2:
                choice_options.append({"value": state[0], "text": state[1]})
            elif isinstance(state, (list, tuple)) and len(state) == 1:
                choice_options.append({"value": state[0], "text": state[0]})
            else:
                choice_options.append({"value": str(state), "text": str(state)})
        field_dict["choice_options"] = choice_options
    else:
        field_dict["type"] = f"unknown ({ft})"
    return field_dict


# Returns a list of fillable PDF fields:
# [
#   {
#     "field_id": "name",
#     "page": 1,
#     "type": ("text", "checkbox", "radio_group", or "choice")
#     // Per-type additional fields described in forms.md
#   },
# ]
def get_field_info(reader: PdfReader):
    fields = reader.get_fields()

    field_info_by_id = {}
    possible_radio_names = set()

    for field_id, field in fields.items():
        # Skip if this is a container field with children, except that it might be
        # a parent group for radio button options.
        if field.get("/Kids"):
            if field.get("/FT") == "/Btn":
                possible_radio_names.add(field_id)
            continue
        field_info_by_id[field_id] = make_field_dict(field, field_id)

    # Bounding rects are stored in annotations in page objects.

    # Radio button options have a separate annotation for each choice;
    # all choices have the same field name.
    # See https://westhealth.github.io/exploring-fillable-forms-with-pdfrw.html
    radio_fields_by_id = {}

    for page_index, page in enumerate(reader.pages):
        annotations = page.get('/Annots', [])
        for ann in annotations:
            # Dereference indirect annotation references
            ann_obj = ann.get_object() if hasattr(ann, "get_object") else ann
            if not isinstance(ann_obj, dict):
                continue
            field_id = get_full_annotation_field_id(ann_obj)
            if field_id in field_info_by_id:
                field_info_by_id[field_id]["page"] = page_index + 1
                rect = ann_obj.get('/Rect')
                field_info_by_id[field_id]["rect"] = rect.get_object() if hasattr(rect, "get_object") else rect
            elif field_id in possible_radio_names:
                try:
                    ap = ann_obj.get("/AP")
                    ap_obj = ap.get_object() if hasattr(ap, "get_object") else ap
                    n = ap_obj.get("/N") if isinstance(ap_obj, dict) else None
                    n_obj = n.get_object() if hasattr(n, "get_object") else n
                    on_values = [v for v in n_obj if v != "/Off"] if isinstance(n_obj, (dict, list)) else []
                except Exception:
                    continue
                if len(on_values) == 1:
                    rect = ann_obj.get("/Rect")
                    rect_obj = rect.get_object() if hasattr(rect, "get_object") else rect
                    if field_id not in radio_fields_by_id:
                        radio_fields_by_id[field_id] = {
                            "field_id": field_id,
                            "type": "radio_group",
                            "page": page_index + 1,
                            "radio_options": [],
                        }
                    # Note: at least on macOS 15.7, Preview.app doesn't show selected
                    # radio buttons correctly. (It does if you remove the leading slash
                    # from the value, but that causes them not to appear correctly in
                    # Chrome/Firefox/Acrobat/etc).
                    radio_fields_by_id[field_id]["radio_options"].append({
                        "value": on_values[0],
                        "rect": rect_obj,
                    })

    # Some PDFs have form field definitions without corresponding annotations,
    # so we can't tell where they are. Ignore these fields for now.
    fields_with_location = []
    for field_info in field_info_by_id.values():
        if "page" in field_info:
            fields_with_location.append(field_info)
        else:
            print(f"Unable to determine location for field id: {field_info.get('field_id')}, ignoring")

    # Sort by page number, then Y position (flipped in PDF coordinate system), then X.
    def sort_key(f):
        if "radio_options" in f:
            rect = f["radio_options"][0]["rect"] or [0, 0, 0, 0]
        else:
            rect = f.get("rect") or [0, 0, 0, 0]
        adjusted_position = [-rect[1], rect[0]]
        return [f.get("page"), adjusted_position]
    
    sorted_fields = fields_with_location + list(radio_fields_by_id.values())
    sorted_fields.sort(key=sort_key)

    return sorted_fields


def write_field_info(pdf_path: str, json_output_path: str):
    reader = PdfReader(pdf_path)
    field_info = get_field_info(reader)
    resolved_json_path = _resolve_artifact_path(json_output_path, config=None)
    with open(resolved_json_path, "w") as f:
        f.write(json.dumps(field_info, indent=2))
    print(f"Wrote {len(field_info)} fields to {resolved_json_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: extract_form_field_info.py [input pdf] [output json]")
        sys.exit(1)
    write_field_info(sys.argv[1], sys.argv[2])
