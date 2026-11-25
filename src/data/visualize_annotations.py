"""Visual annotation gallery generator for semantic equivalence audit."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from src.data.inspect_dataset import DatasetRecord

TILE_W = 420
TILE_H = 340
HEADER_H = 36


def _render_single_tile(
    rec: DatasetRecord,
    target_class_id: int,
    target_label: str,
    source_names: dict[int, str],
) -> Image.Image | None:
    try:
        with Image.open(rec.image_path) as raw_img:
            img = ImageOps.exif_transpose(raw_img).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None

    orig_w, orig_h = img.size
    scale = min(TILE_W / orig_w, (TILE_H - HEADER_H) / orig_h)
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    tile = Image.new("RGB", (TILE_W, TILE_H), color=(24, 24, 27))
    offset_x = (TILE_W - new_w) // 2
    offset_y = HEADER_H + ((TILE_H - HEADER_H) - new_h) // 2
    tile.paste(resized, (offset_x, offset_y))

    draw = ImageDraw.Draw(tile)
    target_boxes = 0
    for cls_id, cx, cy, w, h in rec.annotations:
        x1 = offset_x + (cx - w / 2.0) * new_w
        y1 = offset_y + (cy - h / 2.0) * new_h
        x2 = offset_x + (cx + w / 2.0) * new_w
        y2 = offset_y + (cy + h / 2.0) * new_h
        if cls_id == target_class_id:
            target_boxes += 1
            draw.rectangle([x1, y1, x2, y2], outline=(239, 68, 68), width=3)
            draw.text((x1 + 2, max(HEADER_H, y1 - 12)), target_label, fill=(250, 204, 21))
        else:
            other_name = source_names.get(cls_id, str(cls_id))
            draw.rectangle([x1, y1, x2, y2], outline=(148, 163, 184), width=1)
            draw.text((x1 + 2, max(HEADER_H, y1 - 10)), other_name, fill=(203, 213, 225))

    caption = (
        f"{rec.source} [{rec.split}] | {rec.image_path.name[:24]} | "
        f"boxes={target_boxes}"
    )
    draw.text((8, 10), caption, fill=(248, 250, 252))
    return tile


def render_class_gallery(
    records: list[DatasetRecord],
    source_class_names: dict[str, dict[int, str]],
    target_classes: dict[str, dict[int, str]],
    output_dir: Path,
    max_samples_per_class: int = 6,
) -> list[Path]:
    """Render visual grids per source and candidate class for manual audit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []

    for source_name, id_to_canonical in target_classes.items():
        source_records = [r for r in records if r.source == source_name]
        names_map = source_class_names.get(source_name, {})

        for cls_id, canonical_name in id_to_canonical.items():
            matching = [
                r
                for r in source_records
                if any(ann[0] == cls_id for ann in r.annotations)
            ]
            selected = matching[:max_samples_per_class]
            if not selected:
                continue

            tiles = [
                tile
                for rec in selected
                if (
                    tile := _render_single_tile(
                        rec, cls_id, f"{canonical_name} ({names_map.get(cls_id, cls_id)})", names_map
                    )
                )
                is not None
            ]
            if not tiles:
                continue

            cols = min(3, len(tiles))
            rows = (len(tiles) + cols - 1) // cols
            canvas = Image.new("RGB", (cols * TILE_W, rows * TILE_H), color=(15, 23, 42))
            for idx, tile in enumerate(tiles):
                r_idx, c_idx = divmod(idx, cols)
                canvas.paste(tile, (c_idx * TILE_W, r_idx * TILE_H))

            out_file = output_dir / f"{canonical_name}__{source_name}.png"
            canvas.save(out_file, format="PNG")
            generated_paths.append(out_file)

    return generated_paths
