"""
blocking.py

Candidate generation / blocking stage for the Business Entity Resolution
challenge.

Strategy (as discussed):
  1. Normalize business_name (and optionally business_address) using
     normalize.py -- strips legal suffixes, punctuation, diacritics.
  2. Hard-partition all records by their literal `country` string.
     Country is treated as an open set (works for unseen labels like
     "France" at test time) -- never hardcode to {US, India}.
  3. Within each country partition, build separate sparse TF-IDF matrices
     for Source 2 and Source 3 (normalized name tokens; optional address
     tokens can be added to documents).
  4. Query each source in batches and keep the top-k candidates by cosine
     similarity.
  5. Union the S2 and S3 candidates into a single candidate list per
     S1 entity and write candidate_pairs.tsv.

Output format (matches candidate_pairs.tsv spec exactly):
    source1_entity_id \t candidate_entity_ids
    S1-00001           \t S2-00047,S2-00193,S3-00812,S3-00999
    S1-00002           \t S3-00004
    S1-00003           \t                                   (empty = no candidates found)

No external lookups, geocoding, or APIs are used anywhere in this file --
TF-IDF is a purely statistical, on-data method, satisfying the challenge's
fair-play rule against external data augmentation.

--------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------

# 1. Make sure dependencies are installed (from repo root, venv active):
#      uv pip install -r requirements.txt
#
# 2. Generate candidates for the TRAIN FOLD (S1 train-fold entities
#    searching against the full train_source2 / train_source3 pool).
#    Use this to check blocking quality (recall ceiling) before anything
#    else, via eval_harness.candidate_recall_ceiling().
#
#    uv run python3 src/blocking.py \
#        --s1   data/folds/s1_train_fold.tsv \
#        --s2   data/train_source2.tsv \
#        --s3   data/train_source3.tsv \
#        --out  data/folds/candidate_pairs_train_fold.tsv \
#        --k    15
#
# 3. Generate candidates for the VAL FOLD (same pool, val-fold S1
#    entities) -- this is what Person B trains/scores the matching
#    model against.
#
#    uv run python3 src/blocking.py \
#        --s1   data/folds/s1_val_fold.tsv \
#        --s2   data/train_source2.tsv \
#        --s3   data/train_source3.tsv \
#        --out  data/folds/candidate_pairs_val_fold.tsv \
#        --k    15
#
# 4. Later, for the real test set (final submission):
#
#    uv run python3 src/blocking.py \
#        --s1   data/test/test_source1.tsv \
#        --s2   data/test/test_source2.tsv \
#        --s3   data/test/test_source3.tsv \
#        --out  output/candidate_pairs.tsv \
#        --k    15
#
# 5. Check recall ceiling on a fold (requires eval_harness.py):
#
#    uv run python3 -c "
#    from eval_harness import load_id_dict, candidate_recall_ceiling
#    true_dict = load_id_dict('data/folds/gt_train_fold.tsv', 'source1_entity_id', 'matched_entity_ids')
#    cand_dict = load_id_dict('data/folds/candidate_pairs_train_fold.tsv', 'source1_entity_id', 'candidate_entity_ids')
#    print('recall ceiling:', candidate_recall_ceiling(true_dict, cand_dict))
#    "
#
#    If recall ceiling is too low, raise --k (more candidates per entity)
#    or revisit normalization. If it's high but candidate sets are huge,
#    lower --k -- smaller candidate sets are explicitly rewarded by the
#    challenge's final ranking criteria.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from normalize import name_tokens, normalize_address


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_source(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"entity_id", "business_name", "business_address", "country"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def _vectorizer() -> TfidfVectorizer:
    """Use normalized whitespace tokens and compact float32 sparse matrices."""
    return TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        dtype=np.float32,
        norm="l2",
    )


def build_doc_texts(df: pd.DataFrame, address_weight: int = 0) -> List[str]:
    """Create normalized TF-IDF documents, optionally repeating address terms."""
    names = df["business_name"].map(name_tokens)
    if address_weight <= 0:
        return [" ".join(tokens) for tokens in names]

    addresses = df["business_address"].map(normalize_address)
    return [
        " ".join(name + (address.split() * address_weight if address else []))
        for name, address in zip(names, addresses)
    ]


# ---------------------------------------------------------------------------
# Per-country BM25 index + query
# ---------------------------------------------------------------------------

def build_country_index(df_source: pd.DataFrame, address_weight: int = 0):
    """Build sparse TF-IDF indexes by country for one source."""
    indexes = {}
    for country, group in df_source.groupby("country"):
        docs = build_doc_texts(group, address_weight=address_weight)
        vectorizer = _vectorizer()
        try:
            matrix = vectorizer.fit_transform(docs).tocsr()
        except ValueError:  # all names empty in this country partition
            matrix = None
        indexes[country] = (vectorizer, matrix, group["entity_id"].to_numpy())
    return indexes


def query_index(vectorizer, matrix, entity_ids, query_texts: List[str], k: int):
    """Batch-query one country index and return ranked ID lists per query."""
    if matrix is None or not query_texts:
        return [[] for _ in query_texts]

    query_matrix = vectorizer.transform(query_texts)
    similarities = (query_matrix @ matrix.T).tocsr()
    output = []
    for row_no in range(similarities.shape[0]):
        start, end = similarities.indptr[row_no:row_no + 2]
        indices = similarities.indices[start:end]
        scores = similarities.data[start:end]
        if len(indices) > k:
            selected = np.argpartition(scores, -k)[-k:]
            indices, scores = indices[selected], scores[selected]
        # Stable tie-break by corpus order for reproducible candidate files.
        ranked = np.lexsort((indices, -scores))
        output.append([entity_ids[i] for i in indices[ranked]])
    return output


# ---------------------------------------------------------------------------
# Main candidate generation
# ---------------------------------------------------------------------------

def generate_candidates(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    out_path: str,
    k: int = 15,
    address_weight: int = 0,
) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")

    s1 = load_source(s1_path)
    s2 = load_source(s2_path)
    s3 = load_source(s3_path)

    print(f"S1: {len(s1)} entities | S2: {len(s2)} | S3: {len(s3)}")
    print("Building per-country sparse TF-IDF indexes for S2...", flush=True)
    s2_indexes = build_country_index(s2, address_weight=address_weight)
    print("Building per-country sparse TF-IDF indexes for S3...", flush=True)
    s3_indexes = build_country_index(s3, address_weight=address_weight)

    print(f"S2 countries indexed: {len(s2_indexes)}")
    print(f"S3 countries indexed: {len(s3_indexes)}")

    candidate_lists = [[] for _ in range(len(s1))]
    unmatched_countries = defaultdict(int)  # countries in S1 absent from S2/S3
    # Optional address tokens enrich indexed documents, not S1 queries.
    query_texts = [" ".join(tokens) for tokens in s1["business_name"].map(name_tokens)]
    s1_groups = s1.groupby("country").indices

    for source_name, indexes in (("S2", s2_indexes), ("S3", s3_indexes)):
        for country, positions in tqdm(s1_groups.items(), desc=f"Querying {source_name}"):
            if country not in indexes:
                unmatched_countries[(source_name, country)] += len(positions)
                continue
            vectorizer, matrix, ids = indexes[country]
            # Keep the dense result buffers bounded while still amortizing
            # sparse-matrix multiplication over thousands of S1 queries.
            chunk_size = 4096
            for offset in range(0, len(positions), chunk_size):
                chunk_positions = positions[offset:offset + chunk_size]
                chunk_queries = [query_texts[i] for i in chunk_positions]
                matches = query_index(vectorizer, matrix, ids, chunk_queries, k)
                for position, matches_for_query in zip(chunk_positions, matches):
                    candidate_lists[position].extend(matches_for_query)

    rows = []
    for entity_id, candidates in zip(s1["entity_id"], candidate_lists):
        seen = set()
        deduped = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                deduped.append(candidate)
        rows.append({
            "source1_entity_id": entity_id,
            "candidate_entity_ids": ",".join(deduped),
        })

    if unmatched_countries:
        print("\n[warn] Some S1 countries have no records in the corresponding source "
              "(these S1 entities got 0 candidates from that source):")
        for (source, country), count in unmatched_countries.items():
            print(f"  {source} / country={country!r}: {count} S1 entities affected")

    out_df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])

    # Sanity: exactly one row per S1 entity, no duplicate source1_entity_id
    assert out_df["source1_entity_id"].duplicated().sum() == 0
    assert len(out_df) == len(s1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    out_df.to_csv(out_path, sep="\t", index=False)

    empty_count = (out_df["candidate_entity_ids"] == "").sum()
    avg_candidates = out_df["candidate_entity_ids"].apply(
        lambda x: 0 if x == "" else len(x.split(","))
    ).mean()

    print(f"\nWrote {len(out_df)} rows to {out_path}")
    print(f"S1 entities with 0 candidates: {empty_count} ({empty_count/len(out_df):.1%})")
    print(f"Avg candidates per entity: {avg_candidates:.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batched TF-IDF country-partitioned blocking")
    parser.add_argument("--s1", required=True, help="path to a source1 file (full or fold)")
    parser.add_argument("--s2", required=True, help="path to a source2 file (full pool)")
    parser.add_argument("--s3", required=True, help="path to a source3 file (full pool)")
    parser.add_argument("--out", required=True, help="output path for candidate_pairs.tsv")
    parser.add_argument("--k", type=int, default=15, help="top-k candidates per source (S2 and S3 each)")
    parser.add_argument("--address-weight", type=int, default=0,
                         help="repeat address tokens this many times in the BM25 doc "
                              "(0 = name-only index, recommended default)")
    args = parser.parse_args()

    generate_candidates(
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        out_path=args.out,
        k=args.k,
        address_weight=args.address_weight,
    )


if __name__ == "__main__":
    main()
