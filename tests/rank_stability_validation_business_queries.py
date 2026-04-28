from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.embeddings import build_all_profiles, cosine_similarity, embed_texts


def _query_from_act_column(col: str) -> str:
    base = str(col).removeprefix("act_").removesuffix("_storefront")
    return base.replace("_", " ").strip().lower()


def _derive_queries_from_act_columns(df22: pd.DataFrame, df24: pd.DataFrame) -> list[str]:
    act22 = {
        c for c in df22.columns if str(c).startswith("act_") and str(c).endswith("_storefront")
    }
    act24 = {
        c for c in df24.columns if str(c).startswith("act_") and str(c).endswith("_storefront")
    }
    common = sorted(act22 & act24)
    return [_query_from_act_column(c) for c in common]


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        q = str(raw).strip().lower()
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(q)
    return out


def _sanitize_filename(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "query"


def _build_query_ranks(
    df: pd.DataFrame,
    embeddings,
    query: str,
    *,
    query_label: str,
    key_cols: list[str],
) -> pd.DataFrame:
    q_vec = embed_texts([query])[0]
    sims = cosine_similarity(q_vec, embeddings)
    out = df[key_cols].copy()
    out[f"cosine_{query_label}"] = sims
    out[f"rank_{query_label}"] = (
        out[f"cosine_{query_label}"]
        .rank(ascending=False, method="average")
        .astype(float)
    )
    return out


def _scatter_plot(
    merged: pd.DataFrame,
    *,
    query: str,
    query_slug: str,
    out_path: Path,
    top_outliers: int = 8,
) -> None:
    plt.figure(figsize=(7.5, 6.5))
    x = merged["rank_2022"]
    y = merged["rank_2024"]
    plt.scatter(x, y, alpha=0.75)

    lim_min = 1
    lim_max = int(max(x.max(), y.max()) + 1)
    plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", linewidth=1)
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    merged = merged.copy()
    merged["rank_delta_abs"] = (merged["rank_2022"] - merged["rank_2024"]).abs()
    outliers = merged.sort_values("rank_delta_abs", ascending=False).head(top_outliers)
    for _, row in outliers.iterrows():
        plt.annotate(
            str(row["neighborhood"]),
            (row["rank_2022"], row["rank_2024"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            alpha=0.85,
        )

    plt.title(f"Rank Stability (2022 vs 2024)\nQuery: {query}")
    plt.xlabel("Rank 2022 (1 = highest cosine)")
    plt.ylabel("Rank 2024 (1 = highest cosine)")
    plt.tight_layout()
    plt.savefig(out_path / f"scatter_{query_slug}.png", dpi=160)
    plt.close()


def run_validation(
    *,
    features_2022: Path,
    features_2024: Path,
    output_dir: Path,
    queries: list[str] | None,
    clean_output: bool = False,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        for p in output_dir.glob("*"):
            if p.is_file():
                p.unlink()

    df22 = pd.read_csv(features_2022)
    df24 = pd.read_csv(features_2024)

    key_cols = ["neighborhood", "cd", "borough"]
    for col in key_cols:
        if col not in df22.columns or col not in df24.columns:
            raise ValueError(f"Missing key column '{col}' in one of the input files.")

    if queries is None or len(queries) == 0:
        queries = _derive_queries_from_act_columns(df22, df24)
        if not queries:
            raise ValueError("No common act_*_storefront columns found to derive queries.")
    queries = _dedupe_keep_order(queries)
    if not queries:
        raise ValueError("No usable queries after de-duplication.")

    emb22 = embed_texts(build_all_profiles(df22))
    emb24 = embed_texts(build_all_profiles(df24))

    rows: list[dict[str, float | str | int]] = []
    for query in queries:
        slug = _sanitize_filename(query)
        r22 = _build_query_ranks(
            df22,
            emb22,
            query,
            query_label="2022",
            key_cols=key_cols,
        )
        r24 = _build_query_ranks(
            df24,
            emb24,
            query,
            query_label="2024",
            key_cols=key_cols,
        )

        merged = r22.merge(r24, on=key_cols, how="inner")
        if merged.empty:
            raise ValueError("No overlapping CDTA keys between 2022 and 2024 features.")

        spearman_r = merged["rank_2022"].corr(merged["rank_2024"], method="spearman")
        kendall_tau = merged["rank_2022"].corr(merged["rank_2024"], method="kendall")

        rows.append(
            {
                "query": query,
                "n_cdta_overlap": int(len(merged)),
                "spearman_r": float(spearman_r),
                "kendall_tau": float(kendall_tau),
            }
        )

        merged_for_save = merged.copy()
        merged_for_save["rank_delta"] = (
            merged_for_save["rank_2024"] - merged_for_save["rank_2022"]
        )
        merged_for_save["rank_delta_abs"] = merged_for_save["rank_delta"].abs()
        merged_for_save = merged_for_save.sort_values("rank_delta_abs", ascending=False)
        merged_for_save.to_csv(output_dir / f"rank_compare_{slug}.csv", index=False)

        _scatter_plot(
            merged,
            query=query,
            query_slug=slug,
            out_path=output_dir,
        )

    result = pd.DataFrame(rows).sort_values("spearman_r", ascending=False)
    result.to_csv(output_dir / "query_rank_correlations.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rank Stability Validation for business-category semantic queries "
            "(2022 vs 2024)."
        )
    )
    parser.add_argument(
        "--features-2022",
        type=Path,
        default=Path("tests/data/neighborhood_features_final.csv"),
        help="Path to 2022 feature table CSV.",
    )
    parser.add_argument(
        "--features-2024",
        type=Path,
        default=Path("data/processed/neighborhood_features_final.csv"),
        help="Path to 2024 feature table CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation/rank_stability_business_queries"),
        help="Directory to write correlation table and scatter plots.",
    )
    parser.add_argument(
        "--queries",
        nargs="*",
        default=None,
        help=(
            "Optional override query list. If omitted, queries are auto-derived "
            "from common act_*_storefront columns (prefix/suffix removed)."
        ),
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="If set, delete existing files in output dir before writing new results.",
    )
    args = parser.parse_args()

    result = run_validation(
        features_2022=args.features_2022,
        features_2024=args.features_2024,
        output_dir=args.output_dir,
        queries=args.queries,
        clean_output=args.clean_output,
    )

    print("\nRank-stability correlation summary:")
    print(result.to_string(index=False))
    print(f"\nSaved artifacts under: {args.output_dir}")


if __name__ == "__main__":
    main()
