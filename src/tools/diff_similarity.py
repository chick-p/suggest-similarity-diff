import logging
from typing import Dict
import subprocess
import re
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
from ..mcp_app import mcp

logger = logging.getLogger(__name__)
log_level = os.environ.get("INFO_LEVEL", "INFO").upper()
log_level_enum = getattr(logging, log_level, logging.INFO)
logger.setLevel(log_level_enum)

try:
    import faiss
    _faiss_available = True
except ImportError:
    _faiss_available = False

@mcp.tool()
def diff_similarity() -> Dict:
    """
    Git diffの各hunkと既存埋め込みのコサイン類似度を計算し、類似チャンク情報を返す。
    """
    threshold = float(os.environ.get("SIMILARITY_THRESHOLD", 0.75))
    base_ref = _get_base_ref()
    diff_text = _get_git_diff(base_ref)
    hunks = _parse_diff_hunks(diff_text)
    embed_model_name = os.environ.get("EMBED_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
    allow_self = os.environ.get("ALLOW_SELF_MATCH", "false").lower() == "true"
    model = SentenceTransformer(embed_model_name)

    data_dir_path = "/app/data"
    vectors_file_path = os.path.join(data_dir_path, "embeddings.jsonl")
    vectors = _load_vectors(vectors_file_path)
    results = []
    use_faiss = _faiss_available and os.environ.get("USE_FAISS", "false").lower() == "true"

    logger.debug(f"model={embed_model_name}, hunks={len(hunks)}, vectors={len(vectors)}")

    if use_faiss and vectors:
        dim = len(vectors[0]["vector"])
        index = faiss.IndexFlatIP(dim)
        xb = np.array([v["vector"] for v in vectors], dtype=np.float32)
        faiss.normalize_L2(xb)
        index.add(xb)
        for hunk in hunks:
            hunk_text = "\n".join(hunk["lines"])
            logger.debug(f"Processing hunk: {hunk_text[:50]}...")
            hunk_vector = model.encode(hunk_text, normalize_embeddings=True,show_progress_bar=False).astype(np.float32)
            faiss.normalize_L2(hunk_vector.reshape(1, -1))
            D, I = index.search(hunk_vector.reshape(1, -1), 10)

            for score, idx in zip(D[0], I[0]):
                if idx < 0 or score < threshold:
                    continue
                v = vectors[idx]
                if not allow_self and v["file"] == hunk.get("file", ""):
                    continue
                logger.debug(f"[FOUND] target_file={hunk.get('file', '')}, similar_file={v['file']}, similarity={float(score):.4f}")
                results.append({
                    "target_file": hunk.get("file", ""),
                    "similar_file": v["file"],
                    "hunk": hunk_text,
                    "similarity": float(score),
                    "vector_id": f"{v['file']}:{v['chunk_id']}",
                    "target_hunk_vector": hunk_vector.tolist(),
                    "similar_chunk_vector": v["vector"]
                })
    elif vectors:
        for hunk in hunks:
            hunk_text = "\n".join(hunk["lines"])
            hunk_vector = model.encode(hunk_text, normalize_embeddings=True,show_progress_bar=False).astype(np.float32)

            sim_list = []
            for v in vectors:
                if not allow_self and v["file"] == hunk.get("file", ""):
                    continue
                sim = _cosine_similarity(hunk_vector, np.array(v["vector"], dtype=np.float32))
                sim_list.append((sim, v))

            for sim, v in sorted(sim_list, key=lambda x: x[0], reverse=True)[:5]:
                logger.debug(f"[TOP5] target_file={hunk.get('file', '')}, similar_file={v['file']}, similarity={float(sim):.4f}")

            for sim, v in sim_list:
                if sim >= threshold:
                    logger.debug(f"[SIMILARITY] target_file={hunk.get('file', '')}, similar_file={v['file']}, similarity={float(sim):.4f}")
                    results.append({
                        "target_file": hunk.get("file", ""),
                        "similar_file": v["file"],
                        "hunk": hunk_text,
                        "similarity": float(sim),
                        "vector_id": f"{v['file']}:{v['chunk_id']}",
                        "target_hunk_vector": hunk_vector.tolist(),
                        "similar_chunk_vector": v["vector"]
                    })

    if logger.isEnabledFor(logging.DEBUG):
        for r in results:
            logger.debug(f"similarity={r['similarity']:.4f} target={r['target_file']} similar={r['similar_file']}")

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "base_ref": base_ref,
        "threshold": threshold,
        "results": results
    }

def _get_base_ref() -> str:
    default_branch_name = os.environ.get("GIT_DEFAULT_BRANCH_NAME", "main")
    base_ref = f"origin/{default_branch_name}"
    logger.debug(f"Base reference for diff: {base_ref}")
    return base_ref

def _get_git_diff(base_ref: str) -> str:
    repo_dir_path = "/app/repo"
    if not os.path.exists(repo_dir_path):
        raise FileNotFoundError(f"Repository directory not found: {repo_dir_path}")
    os.chdir(repo_dir_path)
    command = f"git diff --unified=0 {base_ref}..HEAD -- '*.md' '*.mdx'"
    result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    return result.stdout

def _parse_diff_hunks(diff_text: str):
    hunks = []
    current_file = ""
    current_hunk = None
    for line in diff_text.splitlines():
        if line.startswith("--- a/"):
            continue
        if line.startswith("+++ b/"):
            current_file = line[6:]
            continue

        if line.startswith("@@"):
            if current_hunk:
                hunks.append(current_hunk)
            current_hunk = {"file": current_file, "header": line, "lines": []}
        elif current_hunk is not None and (line.startswith("+") or line.startswith("-")):
            current_hunk["lines"].append(line)

    if current_hunk:
        hunks.append(current_hunk)
    return hunks

def _load_vectors(path: str):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0

    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))
