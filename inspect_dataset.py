# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "pillow",
# ]
# ///

import io
from pathlib import Path
from PIL import Image
from datasets import DatasetDict, load_dataset, load_from_disk

# --- Configuration ---
DATASET_ID = "Vishva007/BoundingDocs-Hard-10K"  # HF repo ID or local folder path
OUTPUT_DIR = Path("saved_samples")
NUM_SAMPLES_PER_SPLIT = 10
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


def main():
    # Load from Hugging Face Hub or local directory
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
        print(f"Split '{split_name}': {count:,} rows | Columns: {split_ds.column_names}")

        if count == 0:
            continue

        img_col = detect_image_column(split_ds.column_names)
        split_out = OUTPUT_DIR / split_name
        split_out.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        for idx in range(min(NUM_SAMPLES_PER_SPLIT, count)):
            raw_val = split_ds[idx][img_col]
            # Handle both single-image columns and list-of-images columns (e.g., doc_images)
            img_list = raw_val if isinstance(raw_val, list) else [raw_val]

            for page_idx, img_item in enumerate(img_list):
                pil_img = to_pil_image(img_item)
                out_path = split_out / f"sample_{idx:04d}_p{page_idx}.png"
                pil_img.save(out_path)
                saved_count += 1

        print(f"  -> Saved {saved_count} image(s) to '{split_out}/'")

    print(f"\nTotal rows across all splits: {total_rows:,}")


if __name__ == "__main__":
    main()
