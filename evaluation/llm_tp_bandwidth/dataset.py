"""Token-cache helpers for the LLM tensor-parallel sidecar.

The helpers tokenize Wikitext-2 style text into fixed-length blocks, cache the
result locally, and provide deterministic batches to the worker process.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterator, List

import torch


def _tokenize_text_chunks(
    *,
    tokenizer,
    text_chunks: List[str],
    batch_size: int = 256,
) -> List[int]:
    """Tokenize text chunks in batches without model-length truncation.

    Batching avoids per-line tokenizer overhead and keeps long text streams from
    triggering unnecessary max-length warnings. The returned list is a flat token
    id stream.
    """

    token_ids: List[int] = []
    for start_idx in range(0, len(text_chunks), int(batch_size)):
        chunk_batch = text_chunks[start_idx : start_idx + int(batch_size)]
        if not chunk_batch:
            continue
        batch_encoding = tokenizer(
            chunk_batch,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        for row_ids in batch_encoding["input_ids"]:
            token_ids.extend(row_ids)
    return token_ids


def _sanitize_model_name(model_path: str) -> str:
    """Convert a model path into a cache-file-safe tag."""

    return Path(model_path).name.replace(".", "_")


def build_token_cache_path(
    *,
    cache_dir: Path,
    model_path: str,
    seq_len: int,
    split: str,
) -> Path:
    """Build the deterministic token-cache path for a model/split/sequence length."""

    model_tag = _sanitize_model_name(model_path)
    fingerprint = hashlib.sha1(f"{model_path}|{seq_len}|{split}".encode("utf-8")).hexdigest()[:10]
    return cache_dir / f"wikitext2_{model_tag}_{split}_sl{int(seq_len)}_{fingerprint}.pt"


def prepare_wikitext_token_cache(
    *,
    model_path: str,
    cache_dir: Path,
    seq_len: int,
    split: str = "train",
    min_blocks: int = 64,
) -> Path:
    """Prepare a Wikitext-2 token cache if it does not already exist.

    Non-empty rows are concatenated with EOS, tokenized in batches, and reshaped
    into fixed-length blocks for deterministic worker input.
    """

    cache_path = build_token_cache_path(
        cache_dir=cache_dir,
        model_path=model_path,
        seq_len=seq_len,
        split=split,
    )
    if cache_path.exists():
        return cache_path

    from datasets import load_dataset
    from transformers import AutoTokenizer

    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    text_chunks: List[str] = []
    eos_token = tokenizer.eos_token or ""
    for row in dataset:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        text_chunks.append(text + eos_token)
    if not text_chunks:
        raise ValueError("Wikitext-2 split does not contain usable text rows.")

    token_ids = _tokenize_text_chunks(
        tokenizer=tokenizer,
        text_chunks=text_chunks,
    )
    if len(token_ids) < seq_len:
        raise ValueError(f"Tokenized corpus is shorter than seq_len={seq_len}.")

    usable_length = (len(token_ids) // seq_len) * seq_len
    token_tensor = torch.tensor(token_ids[:usable_length], dtype=torch.long).view(-1, seq_len)
    if token_tensor.size(0) < int(min_blocks):
        raise ValueError(
            f"Prepared token blocks are insufficient: got {token_tensor.size(0)}, require at least {min_blocks}."
        )

    payload: Dict[str, object] = {
        "input_ids": token_tensor,
        "seq_len": int(seq_len),
        "split": split,
        "model_path": model_path,
        "num_blocks": int(token_tensor.size(0)),
    }
    torch.save(payload, cache_path)
    return cache_path


def load_token_blocks(cache_path: Path) -> torch.Tensor:
    """Load cached token blocks as a contiguous long tensor."""

    payload = torch.load(cache_path, map_location="cpu")
    input_ids = payload["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError(f"Cached input_ids must be a torch.Tensor, got {type(input_ids)}")
    return input_ids.long().contiguous()


def iterate_blocks(token_blocks: torch.Tensor, *, total_steps: int) -> Iterator[torch.Tensor]:
    """Yield token blocks for the requested number of steps.

    Blocks are reused cyclically when `total_steps` exceeds the cache size.
    """

    if token_blocks.ndim != 2:
        raise ValueError(f"token_blocks must be 2-D, got shape={tuple(token_blocks.shape)}")

    num_blocks = int(token_blocks.size(0))
    for step_idx in range(int(total_steps)):
        yield token_blocks[step_idx % num_blocks]
