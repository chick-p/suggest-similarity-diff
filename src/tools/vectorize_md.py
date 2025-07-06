import logging
import os
import glob
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import hashlib
import time
from datetime import datetime
from ..mcp_app import mcp

logger = logging.getLogger(__name__)
log_level = os.environ.get("INFO_LEVEL", "INFO").upper()
log_level_enum = getattr(logging, log_level, logging.INFO)
logger.setLevel(log_level_enum)

@mcp.tool()
def vectorize_md(force: bool = False) -> Dict:
    """
    Markdownドキュメントを再帰的に探索し、日本語チャンクに分割して埋め込みを生成する。
    """
    repo_dir_path = "/app/repo"

    embed_model_name = os.environ.get("EMBED_MODEL_NAME")
    if embed_model_name is None:
        embed_model_name = "paraphrase-multilingual-MiniLM-L12-v2"

    logger.debug(f"Searching for markdown files in: {repo_dir_path}")
    md_files = _find_markdown_files(repo_dir_path)
    processed_files = []
    chunks = []
    embed_output = "embeddings.jsonl"
    hash_db_path = embed_output + ".hashes.json"
    file_hashes = _load_hash_db(hash_db_path)
    model = SentenceTransformer(embed_model_name)
    updated_hashes = {}
    logger.debug(f"Found {len(md_files)} markdown files in {repo_dir_path}.")

    logger.debug(f"Executing vectorization with model: {embed_model_name}")

    for file_path in md_files:
        file_hash = _calc_file_hash(file_path)
        if not force and file_path in file_hashes and file_hashes[file_path] == file_hash:
            logger.debug(f"skip: {file_path} (unchanged)")
            continue
        start = time.time() if logger.isEnabledFor(logging.DEBUG) else None
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        file_chunks = _split_into_chunks(text)
        token_count = sum(len(chunk) for chunk in file_chunks)
        for idx, chunk in enumerate(file_chunks):
            if _is_japanese_chunk(chunk):
                vector = model.encode(chunk, normalize_embeddings=True,show_progress_bar=False)
                chunk_obj = {
                    "file": file_path,
                    "chunk_id": idx,
                    "text": chunk,
                    "vector": vector.astype(np.float32).tolist()
                }
                chunks.append(chunk_obj)
                with open(embed_output, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(chunk_obj, ensure_ascii=False) + "\n")
        processed_files.append(file_path)
        updated_hashes[file_path] = file_hash
        if logger.isEnabledFor(logging.DEBUG):
            elapsed = time.time() - start

    _save_hash_db(hash_db_path, {**file_hashes, **updated_hashes})

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "processed_files": processed_files,
        "embed_model": embed_model_name
    }

def _find_markdown_files(dir_path: str) -> List[str]:
    md_pattern = os.path.join(dir_path, "**", "*.md")
    mdx_pattern = os.path.join(dir_path, "**", "*.mdx")
    logger.debug(f"Glob patterns: {md_pattern}, {mdx_pattern}")

    md_files = glob.glob(md_pattern, recursive=True)
    mdx_files = glob.glob(mdx_pattern, recursive=True)

    found_files = sorted(list(set(md_files + mdx_files)))
    return found_files

def _split_into_chunks(text: str) -> List[str]:
    return [chunk.strip() for chunk in re.split(r'\n\s*\n', text) if chunk.strip()]

def _is_japanese_chunk(text: str) -> bool:
    if not text:
        return False
    ja_chars = re.findall(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]', text)
    return (len(ja_chars) / len(text)) >= 0.3

def _calc_file_hash(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def _load_hash_db(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _save_hash_db(path: str, db: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
