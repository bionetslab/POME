"""Evaluate whether POME's inductive (transform) embeddings of unseen samples
land in the same distribution as the transductive embeddings of training samples.

For each hancock train/test split we:
  1. fit() an Embedder on the training split,
  2. read off transductive training embeddings via get_embeddings(),
  3. compute inductive embeddings for the test split via transform(),
  4. compute inductive embeddings for the *training* samples via transform()
     (the method-effect control),
  5. run scxmatch's Rosenbaum cross-match test (squared euclidean, full
     distance matrix) on:
       - MAIN:    train (transductive)        vs test (inductive)
       - CONTROL: train (transductive)        vs train (inductive)

Interpretation (per split):
  - high p-value (>~0.05)        => the two groups are indistinguishable
                                    (in-distribution).
  - low p-value + negative z     => groups are separable (out-of-distribution).
  - If the CONTROL is also low, the MAIN separation is at least partly an
    inductive-vs-transductive *method artifact* rather than genuine data OOD;
    the gap between MAIN and CONTROL isolates the genuine data-OOD component.

Note: transform()'s KNN initialisation for a training sample includes that
sample's own neighbours, biasing inductive train embeddings toward their
transductive counterparts. The CONTROL is therefore a conservative
(lower-bound) estimate of the method effect.

Run (per CLAUDE.md):
    conda activate torch
    python evaluate_inductive_ood.py                 # full run, 10 splits, 1000 epochs
    python evaluate_inductive_ood.py --splits 1 --epochs 50   # quick smoke test
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

import scxmatch
from pome.gnn_embedding import Embedder, make_deterministic

# --- Configuration -----------------------------------------------------------
SPLITS_DIR = Path(__file__).resolve().parent / "data" / "splits" / "hancock"
RESULTS_CSV = SPLITS_DIR / "ood_evaluation_results.csv"
SUMMARY_PNG = SPLITS_DIR / "ood_evaluation_summary.png"

N_SPLITS = 10
SEED = 42

EMBEDDING_DIMENSION = 32
EPOCHS = 1000
BINS_PER_CONTINUOUS = 15
DISCRETIZATION_TYPE = "z"
NA_ENCODING = -99.0

METRIC = "sqeuclidean"
K = None  # full distance matrix (exact Rosenbaum cross-match)


def load_graph(path: Path) -> pd.DataFrame:
    """Load a graph-format split: rows = variables, cols = samples + 'type'."""
    return pd.read_csv(path, sep="\t", index_col=0)


def run_xmatch(emb_a: pd.DataFrame, label_a: str,
               emb_b: pd.DataFrame, label_b: str) -> dict:
    """Cross-match test between two embedding groups.

    emb_a / emb_b are (n_samples, dim) DataFrames. Returns the scxmatch result
    dict augmented with the two group sizes.
    """
    # Make obs names globally unique (the control reuses the same sample names
    # in both groups, which AnnData would otherwise reject / silently collide).
    names_a = [f"{label_a}::{s}" for s in emb_a.index]
    names_b = [f"{label_b}::{s}" for s in emb_b.index]

    X = np.vstack([emb_a.to_numpy(), emb_b.to_numpy()]).astype(np.float64)
    obs = pd.DataFrame(
        {"group": [label_a] * len(emb_a) + [label_b] * len(emb_b)},
        index=names_a + names_b,
    )
    adata = ad.AnnData(X=X, obs=obs)

    result = scxmatch.test(
        adata,
        group_by="group",
        test_group=label_b,
        reference=label_a,
        metric=METRIC,
        k=K,
    )
    result = dict(result)
    result["n_reference"] = len(emb_a)
    result["n_test"] = len(emb_b)
    return result


def evaluate_split(split_id: int, epochs: int, device: str) -> list[dict]:
    """Fit on one split and run the MAIN and CONTROL cross-match tests."""
    tag = f"split_{split_id:02d}"
    train_path = SPLITS_DIR / f"{tag}_train_graph.tsv"
    test_path = SPLITS_DIR / f"{tag}_test_graph.tsv"

    make_deterministic(SEED)

    train_df = load_graph(train_path)
    test_df = load_graph(test_path)

    embedder = Embedder(
        embedding_dimension=EMBEDDING_DIMENSION,
        epochs=epochs,
        bins_per_continuous=BINS_PER_CONTINUOUS,
        discretization_type=DISCRETIZATION_TYPE,
        na_encoding=NA_ENCODING,
        device=device,
    )
    embedder.fit(train_df)

    train_emb, *_ = embedder.get_embeddings()       # transductive (train)
    test_emb = embedder.transform(test_df)           # inductive (test)
    train_ind_emb = embedder.transform(train_df)     # inductive (train) -> control

    print(
        f"  {tag}: train_emb {train_emb.shape}, "
        f"test_emb {test_emb.shape}, train_ind_emb {train_ind_emb.shape}"
    )

    main = run_xmatch(train_emb, "train", test_emb, "test")
    control = run_xmatch(train_emb, "train_transductive",
                         train_ind_emb, "train_inductive")

    rows = []
    for comparison, res in (("main", main), ("control", control)):
        rows.append({
            "split": split_id,
            "comparison": comparison,
            "n_reference": res["n_reference"],
            "n_test": res["n_test"],
            "p_value": res["p_value"],
            "z_score": res["z_score"],
            "coverage": res["coverage"],
            "effect_strength_ratio": res["effect_strength_ratio"],
        })
        print(
            f"    {comparison:8s} p={res['p_value']:.4g} "
            f"z={res['z_score']:.3f} coverage={res['coverage']:.2%}"
        )
    return rows


def summarize(df: pd.DataFrame) -> None:
    print("\n=== Summary (mean +/- std across splits) ===")
    for comparison, sub in df.groupby("comparison"):
        print(
            f"  {comparison:8s}  "
            f"p_value = {sub['p_value'].mean():.4g} +/- {sub['p_value'].std():.4g}   "
            f"z_score = {sub['z_score'].mean():.3f} +/- {sub['z_score'].std():.3f}   "
            f"coverage = {sub['coverage'].mean():.2%}"
        )
    main = df[df["comparison"] == "main"]
    n_sig = int((main["p_value"] < 0.05).sum())
    print(
        f"\n  MAIN comparison significant (p<0.05, i.e. test embeddings OOD) "
        f"in {n_sig}/{len(main)} splits."
    )
    print(
        "  If the CONTROL is similarly significant, MAIN separation is largely "
        "a method artifact (inductive vs transductive), not genuine data OOD."
    )


def make_plot(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = sorted(df["split"].unique())
    main = df[df["comparison"] == "main"].set_index("split").reindex(splits)
    control = df[df["comparison"] == "control"].set_index("split").reindex(splits)

    x = np.arange(len(splits))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(6, len(splits) * 0.9), 4.5))
    ax.bar(x - width / 2, main["p_value"], width, label="main (train vs test)")
    ax.bar(x + width / 2, control["p_value"], width,
           label="control (train transductive vs inductive)")
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="p = 0.05")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:02d}" for s in splits])
    ax.set_xlabel("split")
    ax.set_ylabel("cross-match p-value")
    ax.set_title("POME inductive-embedding OOD test (scxmatch, sqeuclidean)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(SUMMARY_PNG, dpi=150)
    print(f"\nSaved plot to {SUMMARY_PNG}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=N_SPLITS,
                        help="number of splits to evaluate (default: all 10)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="training epochs per split (default: 1000)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | epochs: {args.epochs} | splits: {args.splits}")

    all_rows = []
    for split_id in range(args.splits):
        print(f"\n[split {split_id:02d}] fitting Embedder ...")
        all_rows.extend(evaluate_split(split_id, args.epochs, device))

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nWrote per-split results to {RESULTS_CSV}")

    summarize(df)
    make_plot(df)


if __name__ == "__main__":
    main()
