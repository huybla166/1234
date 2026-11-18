"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, ROOT, parse_yolo_line  # noqa: F401


def fix(ds, issues, cfg):
    # This step is only for the working copy. Refuse the raw source explicitly.
    if ds.root.resolve() == (ROOT / "data" / "raw").resolve():
        raise ValueError("autofix không được phép sửa data/raw")

    errors = [i for i in issues if i.severity == ERROR]
    fixes = []
    remove_codes = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}
    removed_stems = {
        i.image for i in errors
        if i.code in remove_codes or i.code.startswith("DUP_IMAGE_")
    }

    # Remove each rejected image once; any other issues for it are stale after this.
    for stem in sorted(removed_stems):
        image_paths = [p for p in ds.image_files() if p.stem == stem]
        label_path = ds.label_path(stem)
        present = [p.name for p in image_paths]
        if label_path.exists():
            present.append(label_path.name)
        if not present:
            continue
        for p in image_paths:
            p.unlink(missing_ok=True)
        label_path.unlink(missing_ok=True)
        fixes.append({
            "image": stem, "line": None, "action": "remove_image",
            "before": ", ".join(present), "after": "đã xoá ảnh và nhãn khỏi ds",
        })

    # Missing labels need a person; never create an empty label file.
    seen_missing = set()
    for issue in errors:
        if issue.code != "MISSING_LABEL_FILE" or issue.image in removed_stems:
            continue
        if issue.image in seen_missing:
            continue
        seen_missing.add(issue.image)
        image_path = ds.image_path(issue.image)
        if image_path is None or ds.label_path(issue.image).exists():
            continue
        fixes.append({
            "image": issue.image, "line": None, "action": "needs_label",
            "before": "(không có file)", "after": "chuyển người gán",
        })

    # Repair class names while preserving the other data.yaml fields.
    if any(i.code == "BAD_CLASS_NAME" and i.image == "data.yaml" for i in errors):
        yaml_path = ds.yaml_path
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) if yaml_path.exists() else {}
        if not isinstance(data, dict):
            data = {}
        target = list(cfg["classes"])
        names = data.get("names", {})
        if isinstance(names, list):
            current = {i: name for i, name in enumerate(names)}
        elif isinstance(names, dict):
            try:
                current = {int(k): v for k, v in names.items()}
            except (TypeError, ValueError):
                current = {}
        else:
            current = {}
        expected = {i: name for i, name in enumerate(target)}
        if current != expected:
            before = repr(names)
            data["names"] = target
            yaml_path.write_text(
                yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            fixes.append({
                "image": "data.yaml", "line": None, "action": "rename_classes",
                "before": before, "after": repr(target),
            })

    # Group line issues first so the line numbers stay relative to the original file.
    line_issues = defaultdict(set)
    drop_codes = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
    for issue in errors:
        if issue.image in removed_stems or issue.image == "data.yaml" or issue.line is None:
            continue
        if issue.code in drop_codes or issue.code == "BOX_OUT_OF_IMAGE":
            line_issues[(issue.image, issue.line)].add(issue.code)

    for stem in sorted({stem for stem, _ in line_issues}):
        label_path = ds.label_path(stem)
        if not label_path.exists():
            continue
        original_text = label_path.read_text(encoding="utf-8", errors="replace")
        lines = original_text.splitlines()
        had_final_newline = original_text.endswith(("\n", "\r"))
        changed = False

        for image, line_no in sorted(k for k in line_issues if k[0] == stem):
            if line_no < 1 or line_no > len(lines):
                continue
            before = lines[line_no - 1]
            codes = line_issues[(image, line_no)]
            if codes & drop_codes:
                lines[line_no - 1] = None
                fixes.append({
                    "image": stem, "line": line_no, "action": "drop_line",
                    "before": before, "after": "(đã xoá)",
                })
                changed = True
                continue

            if "BOX_OUT_OF_IMAGE" not in codes:
                continue
            try:
                class_id, cx, cy, width, height = parse_yolo_line(before)
            except (ValueError, OverflowError):
                # A stale or inconsistent issue must not make autofix crash.
                continue
            if width <= 0 or height <= 0:
                continue

            left = max(0.0, cx - width / 2)
            right = min(1.0, cx + width / 2)
            top = max(0.0, cy - height / 2)
            bottom = min(1.0, cy + height / 2)
            if right <= left or bottom <= top:
                lines[line_no - 1] = None
                fixes.append({
                    "image": stem, "line": line_no, "action": "drop_line",
                    "before": before, "after": "(box không giao với ảnh; đã xoá)",
                })
                changed = True
                continue

            clipped = (
                class_id,
                (left + right) / 2,
                (top + bottom) / 2,
                right - left,
                bottom - top,
            )
            after = " ".join([str(clipped[0]), *(format(v, ".10g") for v in clipped[1:])])
            if after == before:
                continue
            lines[line_no - 1] = after
            fixes.append({
                "image": stem, "line": line_no, "action": "clip_box",
                "before": before, "after": after,
            })
            changed = True

        if changed:
            kept = [line for line in lines if line is not None]
            text = "\n".join(kept)
            if had_final_newline:
                text += "\n"
            label_path.write_text(text, encoding="utf-8")

    return fixes
