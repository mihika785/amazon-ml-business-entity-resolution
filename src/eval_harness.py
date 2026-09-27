"""
eval_harness.py

Two responsibilities for this challenge's train/val split:
  1. Split train_ground_truth.tsv (+ train_source1.tsv) into a train_fold
     and a val_fold, stratified by (country, is_singleton), by
     source1_entity_id -- so no S1 entity leaks across folds.
  2. Score predictions against ground truth using the challenge's exact
     F_0.5 formula, macro-averaged per Source 1 entity, with singleton
     edge cases handled per the problem statement:
       - true empty, pred empty  -> 1.0
       - true empty, pred non-empty -> 0.0
       - true non-empty, pred empty -> 0.0 (precision undefined -> treated as 0)

Note: source2/source3 files are NOT split. They remain the full search
pool for both folds -- only S1 entities (and their ground-truth rows)
are partitioned. This mirrors test time, where S1 test entities search
the full S2/S3 test pool.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Set, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 1. Parsing helpers
# ---------------------------------------------------------------------------

def parse_id_list(cell) -> Set[str]:
    """Parse a comma-separated ID-list cell into a set of IDs.

    Handles NaN / empty string / whitespace-only cells as "no matches".
    """
    if pd.isna(cell):
        return set()
    cell = str(cell).strip()
    if cell == "":
        return set()
    return {tok.strip() for tok in cell.split(",") if tok.strip() != ""}


def load_ground_truth(path: str) -> pd.DataFrame:
    """Load train_ground_truth.tsv and add an is_singleton flag."""
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert "source1_entity_id" in gt.columns, "missing source1_entity_id column"
    assert "matched_entity_ids" in gt.columns, "missing matched_entity_ids column"

    # Guard against duplicate source1_entity_id rows in the raw file
    dupe_count = gt["source1_entity_id"].duplicated().sum()
    if dupe_count > 0:
        raise ValueError(
            f"train_ground_truth.tsv has {dupe_count} duplicate "
            f"source1_entity_id rows -- fix upstream before splitting."
        )

    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
    gt["is_singleton"] = gt["matched_entity_ids"].str.strip() == ""
    return gt


# ---------------------------------------------------------------------------
# 2. Split
# ---------------------------------------------------------------------------

def make_split(
    gt_path: str,
    s1_path: str,
    out_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ground truth + source1 into train_fold / val_fold.

    Stratifies jointly on (country, is_singleton) so both folds mirror
    the overall distribution. Falls back to singleton-only stratification
    if any joint stratum is too small (sklearn requires >=2 members per
    stratum for a stratified split).
    """
    gt = load_ground_truth(gt_path)
    s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    assert "entity_id" in s1.columns and "country" in s1.columns, \
        "source1 file missing entity_id or country column"

    # Every ground-truth row must correspond to a real S1 record
    missing = set(gt["source1_entity_id"]) - set(s1["entity_id"])
    if missing:
        raise ValueError(
            f"{len(missing)} source1_entity_id values in ground truth "
            f"are not present in train_source1.tsv, e.g. {list(missing)[:5]}"
        )

    gt = gt.merge(
        s1[["entity_id", "country"]],
        left_on="source1_entity_id", right_on="entity_id", how="left"
    ).drop(columns=["entity_id"])

    gt["strata"] = gt["country"].astype(str) + "_" + gt["is_singleton"].astype(str)

    strata_counts = gt["strata"].value_counts()
    small_strata = strata_counts[strata_counts < 2]
    if not small_strata.empty:
        print(
            f"[warn] {len(small_strata)} (country, singleton) strata have <2 "
            f"rows -- falling back to stratifying on is_singleton only."
        )
        strat_col = gt["is_singleton"]
    else:
        strat_col = gt["strata"]

    train_fold_gt, val_fold_gt = train_test_split(
        gt,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_col,
    )

    train_ids = set(train_fold_gt["source1_entity_id"])
    val_ids = set(val_fold_gt["source1_entity_id"])
    assert train_ids.isdisjoint(val_ids), "train/val S1 id leakage detected"

    s1_train = s1[s1["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)

    assert set(s1_train["entity_id"]) == train_ids
    assert set(s1_val["entity_id"]) == val_ids

    os.makedirs(out_dir, exist_ok=True)
    keep_cols = ["source1_entity_id", "matched_entity_ids"]

    train_fold_gt[keep_cols].to_csv(
        os.path.join(out_dir, "gt_train_fold.tsv"), sep="\t", index=False)
    val_fold_gt[keep_cols].to_csv(
        os.path.join(out_dir, "gt_val_fold.tsv"), sep="\t", index=False)
    s1_train.to_csv(
        os.path.join(out_dir, "s1_train_fold.tsv"), sep="\t", index=False)
    s1_val.to_csv(
        os.path.join(out_dir, "s1_val_fold.tsv"), sep="\t", index=False)

    print(f"Train fold: {len(train_fold_gt)} entities "
          f"({train_fold_gt['is_singleton'].mean():.1%} singleton)")
    print(f"Val fold:   {len(val_fold_gt)} entities "
          f"({val_fold_gt['is_singleton'].mean():.1%} singleton)")
    print(f"Written to: {out_dir}")

    return train_fold_gt, val_fold_gt


def f_beta(true_ids: Set[str], pred_ids: Set[str], beta: float = 0.5) -> float:
    """Per-entity F_beta score, per the challenge's exact rules.

    - true empty & pred empty   -> 1.0  (correct singleton)
    - true empty & pred nonempty -> 0.0 (false merge on a singleton)
    - true nonempty & pred empty -> 0.0 (missed everything: recall=0)
    - otherwise standard F_beta on precision/recall over set overlap
    """
    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids and pred_ids:
        return 0.0
    if true_ids and not pred_ids:
        return 0.0

    tp = len(true_ids & pred_ids)
    precision = tp / len(pred_ids) if pred_ids else 0.0
    recall = tp / len(true_ids) if true_ids else 0.0

    if precision == 0.0 and recall == 0.0:
        return 0.0

    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def score_all(
    true_dict: Dict[str, Set[str]],
    pred_dict: Dict[str, Set[str]],
    beta: float = 0.5,
) -> Dict[str, object]:
    """Macro-average F_beta across all S1 entities in true_dict.

    Every entity in true_dict must have a corresponding entry in
    pred_dict (missing predictions are treated as an empty prediction --
    NOT as missing rows, since the challenge requires every S1 entity to
    appear in your submission).

    Returns a dict with the overall score plus diagnostics useful for
    tuning (singleton accuracy, avg precision/recall).
    """
    scores = []
    precisions = []
    recalls = []

    singleton_total = 0
    singleton_correct = 0
    nonsingleton_total = 0

    missing_preds = 0

    for s1_id, true_ids in true_dict.items():
        pred_ids = pred_dict.get(s1_id)
        if pred_ids is None:
            missing_preds += 1
            pred_ids = set()

        score = f_beta(true_ids, pred_ids, beta=beta)
        scores.append(score)

        tp = len(true_ids & pred_ids)
        precisions.append(tp / len(pred_ids) if pred_ids else (1.0 if not true_ids else 0.0))
        recalls.append(tp / len(true_ids) if true_ids else (1.0 if not pred_ids else 0.0))

        if not true_ids:
            singleton_total += 1
            if not pred_ids:
                singleton_correct += 1
        else:
            nonsingleton_total += 1

    n = len(scores)
    report = {
        "n_entities": n,
        "macro_f_beta": sum(scores) / n if n else 0.0,
        "avg_precision": sum(precisions) / n if n else 0.0,
        "avg_recall": sum(recalls) / n if n else 0.0,
        "singleton_total": singleton_total,
        "singleton_accuracy": (singleton_correct / singleton_total) if singleton_total else None,
        "nonsingleton_total": nonsingleton_total,
        "missing_predictions": missing_preds,  # entities in true_dict absent from pred_dict
    }
    return report


def candidate_recall_ceiling(
    true_dict: Dict[str, Set[str]],
    candidate_dict: Dict[str, Set[str]],
) -> float:
    """Upper bound on recall given the blocking/candidate stage.

    Fraction of true (S1 -> matched_id) pairs that appear in the
    candidate set. This is the ceiling your matching model cannot
    exceed -- run this right after blocking, before training a model.
    """
    total_true_pairs = 0
    covered_pairs = 0
    for s1_id, true_ids in true_dict.items():
        if not true_ids:
            continue
        cand_ids = candidate_dict.get(s1_id, set())
        total_true_pairs += len(true_ids)
        covered_pairs += len(true_ids & cand_ids)
    return covered_pairs / total_true_pairs if total_true_pairs else 1.0


def load_id_dict(path: str, id_col: str, list_col: str) -> Dict[str, Set[str]]:
    """Load a two-column TSV (id, comma-separated-id-list) into a dict."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {row[id_col]: parse_id_list(row[list_col]) for _, row in df.iterrows()}


def print_report(report: Dict[str, object]) -> None:
    print("=" * 50)
    print(f"Entities scored:       {report['n_entities']}")
    print(f"Macro F_0.5:           {report['macro_f_beta']:.4f}")
    print(f"Avg precision:         {report['avg_precision']:.4f}")
    print(f"Avg recall:            {report['avg_recall']:.4f}")
    print(f"Singleton entities:    {report['singleton_total']}")
    sa = report["singleton_accuracy"]
    print(f"Singleton accuracy:    {sa:.4f}" if sa is not None else "Singleton accuracy:    n/a")
    print(f"Non-singleton entities:{report['nonsingleton_total']}")
    print(f"Missing predictions:   {report['missing_predictions']}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Split data or score predictions.")
    sub = parser.add_subparsers(dest="command")

    p_split = sub.add_parser("split", help="Create train_fold/val_fold from train data")
    p_split.add_argument("--gt", required=True, help="path to train_ground_truth.tsv")
    p_split.add_argument("--s1", required=True, help="path to train_source1.tsv")
    p_split.add_argument("--out-dir", required=True)
    p_split.add_argument("--test-size", type=float, default=0.2)
    p_split.add_argument("--seed", type=int, default=42)

    p_score = sub.add_parser("score", help="Score predictions against ground truth")
    p_score.add_argument("--truth", required=True, help="path to gt_val_fold.tsv")
    p_score.add_argument("--pred", required=True, help="path to matching_results.tsv")
    p_score.add_argument("--beta", type=float, default=0.5)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "split":
        make_split(args.gt, args.s1, args.out_dir, args.test_size, args.seed)

    elif args.command == "score":
        true_dict = load_id_dict(args.truth, "source1_entity_id", "matched_entity_ids")
        pred_dict = load_id_dict(args.pred, "source1_entity_id", "matched_entity_ids")
        report = score_all(true_dict, pred_dict, beta=args.beta)
        print_report(report)


if __name__ == "__main__":
    main()