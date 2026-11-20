# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "pillow",
# ]
# ///

import io
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

# --- Configuration ---
DATASET_ID = "Vishva007/BoundingDocs-Hard-10K"  # HF repo ID or local folder path
OUTPUT_DIR = Path("saved_samples")
SAMPLES_PER_SOURCE = 3
SEED = 42
IMAGE_COLUMN_CANDIDATES = ["image", "images", "doc_images", "page_image"]


def to_pil_image(item) -> Image.Image:
    """Convert PIL Image, bytes dict, or raw bytes into a PIL Image."""
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, dict):
        if item.get("bytes"):
            return Image.open(io.BytesIO(item["bytes"])).convert("RGB")
        if item.get("path"):
            return Image.open(item["path"]).convert("RGB")
    if isinstance(item, (bytes, bytearray)):
        return Image.open(io.BytesIO(item)).convert("RGB")
    raise TypeError(f"Unsupported image item type: {type(item)}")


def detect_image_column(column_names: list[str]) -> str:
    for col in IMAGE_COLUMN_CANDIDATES:
        if col in column_names:
            return col
    raise ValueError(f"No image column found in {column_names}. Set it manually.")


def boundingdocs_to_xyxy(box):
    """Converts BoundingDocs [width, height, x, y] (0-1000, top-left) to [x1, y1, x2, y2] (0-1000)."""
    w, h, x, y = box
    return [
        max(0, min(1000, int(round(x)))),
        max(0, min(1000, int(round(y)))),
        max(0, min(1000, int(round(x + w)))),
        max(0, min(1000, int(round(y + h)))),
    ]


def draw_first_qa_boxes(pil_img: Image.Image, raw_qa, target_page_idx: int) -> tuple[Image.Image, dict]:
    """Draws bounding boxes for a grounded QA pair on target_page_idx and returns (annotated_img, qa_info)."""
    vis_img = pil_img.copy()
    draw = ImageDraw.Draw(vis_img)
    img_w, img_h = vis_img.size

    while isinstance(raw_qa, str):
        raw_qa = json.loads(raw_qa)

    qa_entries = list(raw_qa.values()) if isinstance(raw_qa, dict) else (raw_qa or [])
    qa_info = {}

    for q in qa_entries:
        if not isinstance(q, dict):
            continue
        for ans in q.get("answers", []):
            if isinstance(ans, dict) and ans.get("location"):
                ans_page = max(0, int(ans.get("page", 1)) - 1)
                if ans_page == target_page_idx:
                    locs = ans["location"]
                    if locs and isinstance(locs[0], (int, float)):
                        locs = [locs]
                    xyxy_boxes = []
                    for loc in locs:
                        x1, y1, x2, y2 = boundingdocs_to_xyxy(loc)
                        xyxy_boxes.append([x1, y1, x2, y2])
                        px_box = [
                            int(x1 / 1000.0 * img_w),
                            int(y1 / 1000.0 * img_h),
                            int(x2 / 1000.0 * img_w),
                            int(y2 / 1000.0 * img_h),
                        ]
                        draw.rectangle(px_box, outline="red", width=2)

                    qa_info = {
                        "question": q.get("rephrased_question") or q.get("question"),
                        "answer": ans.get("value"),
                        "page_index": target_page_idx,
                        "bbox_xyxy_0_1000": xyxy_boxes,
                    }
                    return vis_img, qa_info

    return vis_img, qa_info


def main():
    rng = random.Random(SEED)

    if Path(DATASET_ID).exists():
        ds = load_from_disk(DATASET_ID)
    else:
        ds = load_dataset(DATASET_ID)

    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"train": ds})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_rows = 0

    print(f"=== Dataset Summary: {DATASET_ID} ===")
    for split_name, split_ds in ds.items():
        count = len(split_ds)
        total_rows += count
        print(f"\nSplit '{split_name}': {count:,} rows | Columns: {split_ds.column_names}")

        if count == 0:
            continue

        img_col = detect_image_column(split_ds.column_names)
        split_out = OUTPUT_DIR / split_name
        split_out.mkdir(parents=True, exist_ok=True)

        # Group row indices by 'source' without decoding images yet (fast column projection)
        sources = split_ds["source"] if "source" in split_ds.column_names else ["default"] * count
        source_to_indices = defaultdict(list)
        for idx, src in enumerate(sources):
            source_to_indices[str(src)].append(idx)

        metadata = []
        for src_name, indices in sorted(source_to_indices.items()):
            chosen_indices = rng.sample(indices, min(SAMPLES_PER_SOURCE, len(indices)))
            safe_src = src_name.replace("/", "_")
            src_dir = split_out / safe_src
            src_dir.mkdir(parents=True, exist_ok=True)

            saved_images = 0
            for sample_num, row_idx in enumerate(chosen_indices, start=1):
                row = split_ds[row_idx]  # Decodes only this single row's images
                raw_val = row[img_col]
                img_list = raw_val if isinstance(raw_val, list) else [raw_val]

                # Pick the page referenced by the first grounded answer (defaults to page 0)
                target_page = 0
                if "Q&A" in row and row["Q&A"]:
                    raw_qa = row["Q&A"]
                    while isinstance(raw_qa, str):
                        raw_qa = json.loads(raw_qa)
                    entries = list(raw_qa.values()) if isinstance(raw_qa, dict) else raw_qa
                    for q in entries:
                        if isinstance(q, dict):
                            ans = next((a for a in q.get("answers", []) if isinstance(a, dict) and a.get("location")), None)
                            if ans:
                                target_page = min(len(img_list) - 1, max(0, int(ans.get("page", 1)) - 1))
                                break

                pil_img = to_pil_image(img_list[target_page])
                raw_path = src_dir / f"{sample_num:02d}_row{row_idx:04d}_p{target_page}_raw.png"
                pil_img.save(raw_path)

                qa_info = {}
                if "Q&A" in row and row["Q&A"]:
                    vis_img, qa_info = draw_first_qa_boxes(pil_img, row["Q&A"], target_page)
                    ann_path = src_dir / f"{sample_num:02d}_row{row_idx:04d}_p{target_page}_annotated.png"
                    vis_img.save(ann_path)

                metadata.append({
                    "source": src_name,
                    "row_index": row_idx,
                    "doc_id": row.get("doc_id"),
                    "raw_image": str(raw_path),
                    **qa_info,
                })
                saved_images += 1

            print(f"  [{src_name}] Total rows: {len(indices):,} -> Saved {saved_images} sample(s) to '{src_dir}/'")

        with open(split_out / "samples_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nTotal rows across all splits: {total_rows:,}")


if __name__ == "__main__":
    main()
