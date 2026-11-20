import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from tqdm import tqdm

REPO_IN = "letxbe/BoundingDocs"
REPO_OUT = "Vishva007/BoundingDocs-Hard-10K"
LOCAL_OUT_DIR = "/workspace/boundingdocs_hard_parquet"
TMP_CACHE_DIR = "/workspace/tmp_hf_cache"
SEED = 42
MAX_WORKERS = 32
ROWS_PER_OUT_SHARD = 100
ROW_GROUP_SIZE = 10

SPLIT_QUOTAS = {
    "train": {
        "funsd": 149,
        "sp-docvqa": 200,
        "mp-docvqa": 2700,
        "deepform": 3401,
        "vrdu": 1651,
    },
    "validation": {
        "funsd": 50,
        "sp-docvqa": 50,
        "mp-docvqa": 300,
        "deepform": 400,
        "vrdu": 200,
    },
    "test": {
        "funsd": 50,
        "sp-docvqa": 50,
        "mp-docvqa": 500,
        "deepform": 600,
        "vrdu": 300,
    },
}


def upgrade_to_large_types(schema: pa.Schema) -> pa.Schema:
    def convert_type(dtype):
        if pa.types.is_string(dtype):
            return pa.large_string()
        if pa.types.is_binary(dtype):
            return pa.large_binary()
        if pa.types.is_list(dtype):
            return pa.large_list(convert_type(dtype.value_type))
        if pa.types.is_struct(dtype):
            return pa.struct([pa.field(f.name, convert_type(f.type)) for f in dtype])
        return dtype

    return pa.schema([pa.field(f.name, convert_type(f.type)) for f in schema])


def scan_shard_sources(shard):
    fs = HfFileSystem()
    hf_path = f"datasets/{REPO_IN}/{shard}"
    try:
        src_col = pq.read_table(hf_path, columns=["source"], filesystem=fs)["source"].to_pylist()
        return shard, src_col
    except Exception:
        return shard, []


def fetch_rows_from_shard(shard, target_indices):
    """Downloads to /workspace/tmp_hf_cache, slices target rows, and deletes the raw shard immediately."""
    os.makedirs(TMP_CACHE_DIR, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=REPO_IN,
        filename=shard,
        repo_type="dataset",
        local_dir=TMP_CACHE_DIR,
    )
    try:
        pf = pq.ParquetFile(local_path)
        large_schema = upgrade_to_large_types(pf.schema_arrow)

        rg_tables = []
        offset = 0
        for rg in range(pf.num_row_groups):
            rg_rows = pf.metadata.row_group(rg).num_rows
            local_hits = [idx - offset for idx in target_indices if offset <= idx < offset + rg_rows]
            if local_hits:
                tbl = pf.read_row_group(rg).cast(large_schema)
                rg_tables.append(tbl.take(pa.array(local_hits)))
            offset += rg_rows

        return pa.concat_tables(rg_tables) if rg_tables else None
    finally:
        # Remove downloaded shard immediately so disk quota never fills up
        if os.path.exists(local_path):
            os.remove(local_path)


def build_split_shards(all_files, split_prefix, quotas, seed=42):
    os.makedirs(LOCAL_OUT_DIR, exist_ok=True)

    existing = sorted([
        f for f in os.listdir(LOCAL_OUT_DIR)
        if f.startswith(f"{split_prefix}-") and f.endswith(".parquet")
    ])
    if existing:
        print(f"\n[{split_prefix.upper()}] Found {len(existing)} existing shards on disk — skipping re-extraction.")
        return [os.path.join(LOCAL_OUT_DIR, f) for f in existing]

    rng = random.Random(seed)
    shards = sorted([f for f in all_files if f.startswith(f"data/{split_prefix}-") and f.endswith(".parquet")])
    rng.shuffle(shards)

    print(f"\n[{split_prefix.upper()}] Scanning {len(shards)} shards in parallel ({MAX_WORKERS} threads)...")
    shard_to_sources = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(scan_shard_sources, s): s for s in shards}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Scanning {split_prefix}"):
            shard, src_list = fut.result()
            if src_list:
                shard_to_sources[shard] = src_list

    shard_selections = {}
    collected_counts = {k: 0 for k in quotas}

    for shard in shards:
        if all(collected_counts[k] >= quotas[k] for k in quotas):
            break
        src_col = shard_to_sources.get(shard, [])
        chosen_in_shard = []
        for row_idx, s in enumerate(src_col):
            if not s:
                continue
            s_low = s.lower()
            for key, max_quota in quotas.items():
                if collected_counts[key] < max_quota and key in s_low:
                    chosen_in_shard.append(row_idx)
                    collected_counts[key] += 1
                    break
        if chosen_in_shard:
            shard_selections[shard] = chosen_in_shard

    print(f"[{split_prefix.upper()}] Selected document counts: {collected_counts}")

    buffer_tables = []
    buffer_rows = 0
    out_shard_idx = 0
    written_files = []

    def flush_buffer():
        nonlocal buffer_tables, buffer_rows, out_shard_idx
        if not buffer_tables:
            return
        chunk = pa.concat_tables(buffer_tables)
        perm = list(range(chunk.num_rows))
        rng.shuffle(perm)
        chunk = chunk.take(pa.array(perm))

        out_file = os.path.join(LOCAL_OUT_DIR, f"{split_prefix}-{out_shard_idx:05d}.parquet")
        pq.write_table(
            chunk,
            out_file,
            compression="snappy",
            row_group_size=ROW_GROUP_SIZE,
            write_page_index=True,
        )
        written_files.append(out_file)
        out_shard_idx += 1
        buffer_tables = []
        buffer_rows = 0

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(fetch_rows_from_shard, shard, idxs): shard
            for shard, idxs in shard_selections.items()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Extracting {split_prefix}"):
            sub_tbl = fut.result()
            if sub_tbl is not None:
                buffer_tables.append(sub_tbl)
                buffer_rows += sub_tbl.num_rows
                if buffer_rows >= ROWS_PER_OUT_SHARD:
                    flush_buffer()

    flush_buffer()
    shutil.rmtree(TMP_CACHE_DIR, ignore_errors=True)
    print(f"[{split_prefix.upper()}] Wrote {len(written_files)} local Parquet shards.")
    return written_files


def main():
    api = HfApi()
    all_files = api.list_repo_files(REPO_IN, repo_type="dataset")

    for split_name, quotas in SPLIT_QUOTAS.items():
        build_split_shards(all_files, split_name, quotas, seed=SEED)

    all_local_shards = sorted(os.listdir(LOCAL_OUT_DIR))
    print(f"\nTotal shards ready in {LOCAL_OUT_DIR}: {len(all_local_shards)}")

    print(f"Uploading folder directly to Hugging Face Hub: {REPO_OUT} ...")
    api.create_repo(repo_id=REPO_OUT, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=LOCAL_OUT_DIR,
        path_in_repo="data",
        repo_id=REPO_OUT,
        repo_type="dataset",
        delete_patterns=["*.parquet"],  # Removes stale 200-row shards on the Hub
    )
    print(f"\nUpload complete! Dataset live at: https://huggingface.co/datasets/{REPO_OUT}")


if __name__ == "__main__":
    main()
