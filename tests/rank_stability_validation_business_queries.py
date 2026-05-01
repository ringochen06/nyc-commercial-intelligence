# from __future__ import annotations

# import argparse
# import sys
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# _ROOT = Path(__file__).parent.parent
# if str(_ROOT) not in sys.path:
#     sys.path.insert(0, str(_ROOT))

# from src.embeddings import build_all_profiles, cosine_similarity, embed_texts


# def combine_scores(
#     semantic_scores: np.ndarray, competitive_scores: np.ndarray, *, alpha: float
# ) -> np.ndarray:
#     """Blend MinMax([semantic, -competitive]) with alpha on semantic."""
#     sims = np.asarray(semantic_scores, dtype=float)
#     comp = np.asarray(competitive_scores, dtype=float)
#     X = np.column_stack([sims, -comp])
#     if X.shape[0] == 1:
#         scaled = np.ones((1, 2)) * 0.5
#     else:
#         mins = X.min(axis=0)
#         spans = X.max(axis=0) - mins
#         spans[spans == 0] = 1.0
#         scaled = (X - mins) / spans
#     beta = 1.0 - float(alpha)
#     return (scaled @ np.array([float(alpha), beta], dtype=float)).astype(float)

# # Fixed, externally authored queries aligned to the sector vocabulary of the act_*_storefront
# # categories (food services, retail, health care, educational services, etc.) but phrased as
# # natural-language site-selection prompts rather than copying the column labels verbatim.
# # Broader framing reduces sensitivity to specific shop types while still exercising the
# # same semantic dimensions that the profile embeddings capture.
# DEFAULT_QUERIES: list[str] = [
#     # FOOD_SERVICES
#     "neighborhood with strong food services activity and high pedestrian volume",
#     # RETAIL
#     "dense retail corridor with diverse storefronts and walkable street-level activity",
#     # HEALTH_CARE_OR_SOCIAL_ASSISTANCE
#     "area with established health care or social assistance presence and residential demand",
#     # EDUCATIONAL_SERVICES
#     "neighborhood with significant educational services and families or students nearby",
#     # FINANCE_AND_INSURANCE
#     "commercial district with finance and insurance businesses and white-collar workforce",
#     # REAL_ESTATE
#     "area with active real estate market and mixed residential and commercial development",
#     # LEGAL_SERVICES
#     "district with professional legal services and proximity to civic institutions",
#     # ACCOUNTING_SERVICES
#     "business-dense neighborhood with accounting services and small-business concentration",
#     # MANUFACTURING
#     "industrial or light manufacturing zone with warehouse or production storefronts",
#     # WHOLESALE
#     "wholesale and distribution corridor with commercial loading and supply-chain access",
#     # INFORMATION_SERVICES
#     "tech or media cluster with information services firms and young professional residents",
#     # BROADCASTING_TELECOMM
#     "area with broadcasting or telecommunications infrastructure and commercial density",
#     # MOVIES_VIDEO_SOUND
#     "creative, medium-density neighborhood with film, video, or sound production activity",
#     # PUBLISHING
#     "mixed-use area with publishing or media businesses and educated residential base",
# ]


# def _query_from_act_column(col: str) -> str:
#     base = str(col).removeprefix("act_").removesuffix("_storefront")
#     return base.replace("_", " ").strip().lower()


# def _derive_queries_from_act_columns(df22: pd.DataFrame, df24: pd.DataFrame) -> list[str]:
#     act22 = {
#         c for c in df22.columns if str(c).startswith("act_") and str(c).endswith("_storefront")
#     }
#     act24 = {
#         c for c in df24.columns if str(c).startswith("act_") and str(c).endswith("_storefront")
#     }
#     common = sorted(act22 & act24)
#     return [_query_from_act_column(c) for c in common]


# def _dedupe_keep_order(items: list[str]) -> list[str]:
#     seen: set[str] = set()
#     out: list[str] = []
#     for raw in items:
#         q = str(raw).strip().lower()
#         if not q or q in seen:
#             continue
#         seen.add(q)
#         out.append(q)
#     return out


# def _sanitize_filename(text: str) -> str:
#     out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
#     while "__" in out:
#         out = out.replace("__", "_")
#     return out or "query"


# def _build_query_ranks(
#     df: pd.DataFrame,
#     embeddings,
#     query: str,
#     *,
#     query_label: str,
#     key_cols: list[str],
#     alpha: float = 0.8,
# ) -> pd.DataFrame:
#     q_vec = embed_texts([query])[0]
#     sims = cosine_similarity(q_vec, embeddings)
#     act = df["competitive_score"].to_numpy(dtype=float)
#     blended = combine_scores(sims, act, alpha=alpha)
#     out = df[key_cols].copy()
#     out[f"cosine_{query_label}"] = sims
#     out[f"blended_{query_label}"] = blended
#     out[f"rank_{query_label}"] = (
#         pd.Series(blended, index=df.index)
#         .rank(ascending=False, method="average")
#         .to_numpy(dtype=float)
#     )
#     return out


# def _scatter_plot(
#     merged: pd.DataFrame,
#     *,
#     query: str,
#     query_slug: str,
#     out_path: Path,
#     top_outliers: int = 8,
# ) -> None:
#     plt.figure(figsize=(7.5, 6.5))
#     x = merged["rank_2022"]
#     y = merged["rank_2024"]
#     plt.scatter(x, y, alpha=0.75)

#     lim_min = 1
#     lim_max = int(max(x.max(), y.max()) + 1)
#     plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", linewidth=1)
#     plt.xlim(lim_min, lim_max)
#     plt.ylim(lim_min, lim_max)
#     plt.gca().invert_xaxis()
#     plt.gca().invert_yaxis()

#     merged = merged.copy()
#     merged["rank_delta_abs"] = (merged["rank_2022"] - merged["rank_2024"]).abs()
#     outliers = merged.sort_values("rank_delta_abs", ascending=False).head(top_outliers)
#     for _, row in outliers.iterrows():
#         plt.annotate(
#             str(row["neighborhood"]),
#             (row["rank_2022"], row["rank_2024"]),
#             xytext=(4, 4),
#             textcoords="offset points",
#             fontsize=8,
#             alpha=0.85,
#         )

#     plt.title(f"Rank Stability (2022 vs 2024) — blended score\nQuery: {query}")
#     plt.xlabel("Rank 2022 (1 = highest cosine)")
#     plt.ylabel("Rank 2024 (1 = highest cosine)")
#     plt.tight_layout()
#     plt.savefig(out_path / f"scatter_{query_slug}.png", dpi=160)
#     plt.close()


# def run_validation(
#     *,
#     features_2022: Path,
#     features_2024: Path,
#     output_dir: Path,
#     queries: list[str] | None,
#     alpha: float = 0.8,
#     derive_queries: bool = False,
#     clean_output: bool = False,
# ) -> pd.DataFrame:
#     output_dir.mkdir(parents=True, exist_ok=True)
#     if clean_output:
#         for p in output_dir.glob("*"):
#             if p.is_file():
#                 p.unlink()

#     df22 = pd.read_csv(features_2022)
#     df24 = pd.read_csv(features_2024)

#     key_cols = ["neighborhood", "cd", "borough"]
#     for col in key_cols:
#         if col not in df22.columns or col not in df24.columns:
#             raise ValueError(f"Missing key column '{col}' in one of the input files.")
#     for df, label in ((df22, "2022"), (df24, "2024")):
#         if "competitive_score" not in df.columns:
#             raise ValueError(f"Missing 'competitive_score' in {label} feature table.")

#     if queries:
#         queries = _dedupe_keep_order(queries)
#     elif derive_queries:
#         queries = _derive_queries_from_act_columns(df22, df24)
#         if not queries:
#             raise ValueError("No common act_*_storefront columns found to derive queries.")
#         queries = _dedupe_keep_order(queries)
#     else:
#         queries = list(DEFAULT_QUERIES)
#     if not queries:
#         raise ValueError("No usable queries after de-duplication.")

#     emb22 = embed_texts(build_all_profiles(df22))
#     emb24 = embed_texts(build_all_profiles(df24))

#     rows: list[dict[str, float | str | int]] = []
#     for query in queries:
#         slug = _sanitize_filename(query)
#         r22 = _build_query_ranks(
#             df22,
#             emb22,
#             query,
#             query_label="2022",
#             key_cols=key_cols,
#             alpha=alpha,
#         )
#         r24 = _build_query_ranks(
#             df24,
#             emb24,
#             query,
#             query_label="2024",
#             key_cols=key_cols,
#             alpha=alpha,
#         )

#         merged = r22.merge(r24, on=key_cols, how="inner")
#         if merged.empty:
#             raise ValueError("No overlapping CDTA keys between 2022 and 2024 features.")

#         spearman_r = merged["rank_2022"].corr(merged["rank_2024"], method="spearman")
#         kendall_tau = merged["rank_2022"].corr(merged["rank_2024"], method="kendall")

#         rows.append(
#             {
#                 "query": query,
#                 "n_cdta_overlap": int(len(merged)),
#                 "spearman_r": float(spearman_r),
#                 "kendall_tau": float(kendall_tau),
#             }
#         )

#         merged_for_save = merged.copy()
#         merged_for_save["rank_delta"] = (
#             merged_for_save["rank_2024"] - merged_for_save["rank_2022"]
#         )
#         merged_for_save["rank_delta_abs"] = merged_for_save["rank_delta"].abs()
#         merged_for_save = merged_for_save.sort_values("rank_delta_abs", ascending=False)
#         merged_for_save.to_csv(output_dir / f"rank_compare_{slug}.csv", index=False)

#         _scatter_plot(
#             merged,
#             query=query,
#             query_slug=slug,
#             out_path=output_dir,
#         )

#     result = pd.DataFrame(rows).sort_values("spearman_r", ascending=False)
#     result.to_csv(output_dir / "query_rank_correlations.csv", index=False)
#     return result


# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description=(
#             "Rank Stability Validation for business-category semantic queries "
#             "(2022 vs 2024)."
#         )
#     )
#     parser.add_argument(
#         "--features-2022",
#         type=Path,
#         default=Path("./data/neighborhood_features_final.csv"),
#         help="Path to 2022 feature table CSV.",
#     )
#     parser.add_argument(
#         "--features-2024",
#         type=Path,
#         default=Path("../data/processed/neighborhood_features_final.csv"),
#         help="Path to 2024 feature table CSV.",
#     )
#     parser.add_argument(
#         "--output-dir",
#         type=Path,
#         default=Path("outputs/validation/rank_stability_business_queries"),
#         help="Directory to write correlation table and scatter plots.",
#     )
#     parser.add_argument(
#         "--queries",
#         nargs="*",
#         default=None,
#         help=(
#             "Optional explicit query list. Overrides both --derive-queries and DEFAULT_QUERIES."
#         ),
#     )
#     parser.add_argument(
#         "--derive-queries",
#         action="store_true",
#         help=(
#             "Derive queries from common act_*_storefront column names instead of DEFAULT_QUERIES. "
#             "Note: this inflates correlations because the vocabulary mirrors the embedded profiles."
#         ),
#     )
#     parser.add_argument(
#         "--alpha",
#         type=float,
#         default=0.8,
#         help="Weight on semantic similarity (0–1); competitive-penalty weight = 1 − alpha.",
#     )
#     parser.add_argument(
#         "--clean-output",
#         action="store_true",
#         help="If set, delete existing files in output dir before writing new results.",
#     )
#     args = parser.parse_args()

#     result = run_validation(
#         features_2022=args.features_2022,
#         features_2024=args.features_2024,
#         output_dir=args.output_dir,
#         queries=args.queries,
#         alpha=args.alpha,
#         derive_queries=args.derive_queries,
#         clean_output=args.clean_output,
#     )

#     print("\nRank-stability correlation summary:")
#     print(result.to_string(index=False))
#     print(f"\nSaved artifacts under: {args.output_dir}")


# if __name__ == "__main__":
#     main()


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.embeddings import (
    EMBEDDINGS_DIR,
    build_all_profiles,
    cosine_similarity,
    embed_texts,
    load_embeddings,
    resolve_embedding_backend,
    save_embeddings,
)


def _embed_for_validation(
    csv_path: Path, vintage: str
) -> tuple[np.ndarray, list[str]]:
    """
    Load embeddings from outputs/embeddings/ cache if available, otherwise embed and save.

    'present' uses load_embeddings / save_embeddings (standard cache).
    'past' caches to *_past.npy files alongside the standard cache so the two vintages
    never overwrite each other.
    """
    if vintage == "present":
        cached = load_embeddings()
        if cached is not None:
            print(f"Cache hit! Loaded {len(cached[0])} embeddings and texts.")
            return cached
        df = pd.read_csv(csv_path)
        texts = build_all_profiles(df)
        embeddings = embed_texts(texts)
        save_embeddings(embeddings, texts)
        print(f"Cache miss! Saved {len(embeddings)} embeddings and texts.")
        return embeddings, texts

    # past vintage: separate _past-suffixed paths since load_embeddings/save_embeddings
    # target the present cache only.
    backend = resolve_embedding_backend()
    emb_path = (
        EMBEDDINGS_DIR / "neighborhood_embeddings_st_past.npy"
        if backend == "sentence_transformers"
        else EMBEDDINGS_DIR / "neighborhood_embeddings_past.npy"
    )
    texts_path = EMBEDDINGS_DIR / "neighborhood_texts_past.npy"
    if emb_path.exists() and texts_path.exists():
        print(f"Cache hit! Loaded {len(np.load(emb_path))} embeddings and texts.")
        return np.load(emb_path), np.load(texts_path, allow_pickle=True).tolist()
    df = pd.read_csv(csv_path)
    texts = build_all_profiles(df)
    embeddings = embed_texts(texts)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    np.save(texts_path, np.array(texts, dtype=object))
    print(f"Cache miss! Saved {len(embeddings)} embeddings and texts.")
    return embeddings, texts


def combine_scores(
    semantic_scores: np.ndarray, competitive_scores: np.ndarray, *, alpha: float
) -> np.ndarray:
    """Blend MinMax([semantic, -competitive]) with alpha on semantic."""
    sims = np.asarray(semantic_scores, dtype=float)
    comp = np.asarray(competitive_scores, dtype=float)
    X = np.column_stack([sims, -comp])
    if X.shape[0] == 1:
        scaled = np.ones((1, 2)) * 0.5
    else:
        mins = X.min(axis=0)
        spans = X.max(axis=0) - mins
        spans[spans == 0] = 1.0
        scaled = (X - mins) / spans
    beta = 1.0 - float(alpha)
    return (scaled @ np.array([float(alpha), beta], dtype=float)).astype(float)

# Fixed, externally authored queries aligned to the sector vocabulary of the act_*_storefront
# categories (food services, retail, health care, educational services, etc.) but phrased as
# natural-language site-selection prompts rather than copying the column labels verbatim.
# Broader framing reduces sensitivity to specific shop types while still exercising the
# same semantic dimensions that the profile embeddings capture.
DEFAULT_QUERIES: list[str] = [
    # FOOD_SERVICES
    "neighborhood with strong food services activity and high pedestrian volume",
    # RETAIL
    "dense retail corridor with diverse storefronts and walkable street-level activity",
    # HEALTH_CARE_OR_SOCIAL_ASSISTANCE
    "area with established health care or social assistance presence and residential demand",
    # EDUCATIONAL_SERVICES
    "neighborhood with significant educational services and families or students nearby",
    # FINANCE_AND_INSURANCE
    "commercial district with finance and insurance businesses and white-collar workforce",
    # REAL_ESTATE
    "area with active real estate market and mixed residential and commercial development",
    # LEGAL_SERVICES
    "district with professional legal services and proximity to civic institutions",
    # ACCOUNTING_SERVICES
    "business-dense neighborhood with accounting services and small-business concentration",
    # MANUFACTURING
    "industrial or light manufacturing zone with warehouse or production storefronts",
    # WHOLESALE
    "wholesale and distribution corridor with commercial loading and supply-chain access",
    # INFORMATION_SERVICES
    "tech or media cluster with information services firms and young professional residents",
    # BROADCASTING_TELECOMM
    "area with broadcasting or telecommunications infrastructure and commercial density",
    # MOVIES_VIDEO_SOUND
    "creative, medium-density neighborhood with film, video, or sound production activity",
    # PUBLISHING
    "mixed-use area with publishing or media businesses and educated residential base",
]


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
    alpha: float = 0.8,
) -> pd.DataFrame:
    q_vec = embed_texts([query])[0]
    sims = cosine_similarity(q_vec, embeddings)
    act = df["competitive_score"].to_numpy(dtype=float)
    blended = combine_scores(sims, act, alpha=alpha)
    out = df[key_cols].copy()
    out[f"cosine_{query_label}"] = sims
    out[f"blended_{query_label}"] = blended
    out[f"rank_{query_label}"] = (
        pd.Series(blended, index=df.index)
        .rank(ascending=False, method="average")
        .to_numpy(dtype=float)
    )
    return out


def _write_interactive_rank_scatter(
    merged_by_query: list[tuple[str, str, pd.DataFrame]],
    *,
    out_path: Path,
) -> None:
    fig = go.Figure()
    buttons: list[dict] = []
    total_traces = max(1, len(merged_by_query) * 2)
    global_max_rank = 1.0

    for idx, (query, query_slug, merged) in enumerate(merged_by_query):
        ordered = merged.copy()
        ordered["rank_delta"] = ordered["rank_2024"] - ordered["rank_2022"]
        ordered = ordered.sort_values(
            ["rank_2024", "rank_2022", "neighborhood", "borough", "cd"]
        ).reset_index(drop=True)
        max_rank = float(max(ordered["rank_2022"].max(), ordered["rank_2024"].max()))
        global_max_rank = max(global_max_rank, max_rank)

        customdata = np.column_stack(
            [
                ordered["neighborhood"].astype(str),
                ordered["borough"].astype(str),
                ordered["cd"].astype(str),
                ordered["rank_2022"].to_numpy(dtype=float),
                ordered["rank_2024"].to_numpy(dtype=float),
                ordered["rank_delta"].to_numpy(dtype=float),
                ordered["blended_2022"].to_numpy(dtype=float),
                ordered["blended_2024"].to_numpy(dtype=float),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=ordered["rank_2022"],
                y=ordered["rank_2024"],
                mode="markers",
                name="Neighborhood rank",
                visible=idx == 0,
                customdata=customdata,
                marker={
                    "size": 9,
                    "opacity": 0.8,
                    "color": ordered["rank_delta"].abs(),
                    "colorscale": "Viridis",
                    "colorbar": {"title": "|rank delta|"},
                },
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Borough: %{customdata[1]}<br>"
                    "CD: %{customdata[2]}<br>"
                    "Rank 2022: %{x:.0f}<br>"
                    "Rank 2024: %{y:.0f}<br>"
                    "Rank delta (2024 - 2022): %{customdata[5]:+.1f}<br>"
                    "Blended 2022: %{customdata[6]:.3f}<br>"
                    "Blended 2024: %{customdata[7]:.3f}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[1.0, max_rank],
                y=[1.0, max_rank],
                mode="lines",
                name="Perfect stability (y = x)",
                visible=idx == 0,
                line={"color": "#dc2626", "width": 2, "dash": "dash"},
                hovertemplate="Perfect stability: rank 2024 = rank 2022<extra></extra>",
            )
        )
        q_fig = go.Figure()
        q_fig.add_trace(
            go.Scatter(
                x=ordered["rank_2022"],
                y=ordered["rank_2024"],
                mode="markers",
                name="Neighborhood rank",
                customdata=customdata,
                marker={
                    "size": 9,
                    "opacity": 0.8,
                    "color": ordered["rank_delta"].abs(),
                    "colorscale": "Viridis",
                    "colorbar": {"title": "|rank delta|"},
                },
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Borough: %{customdata[1]}<br>"
                    "CD: %{customdata[2]}<br>"
                    "Rank 2022: %{x:.0f}<br>"
                    "Rank 2024: %{y:.0f}<br>"
                    "Rank delta (2024 - 2022): %{customdata[5]:+.1f}<br>"
                    "Blended 2022: %{customdata[6]:.3f}<br>"
                    "Blended 2024: %{customdata[7]:.3f}<extra></extra>"
                ),
            )
        )
        q_fig.add_trace(
            go.Scatter(
                x=[1.0, max_rank],
                y=[1.0, max_rank],
                mode="lines",
                name="Perfect stability (y = x)",
                line={"color": "#dc2626", "width": 2, "dash": "dash"},
                hovertemplate="Perfect stability: rank 2024 = rank 2022<extra></extra>",
            )
        )
        q_fig.update_layout(
            title=f"Rank Stability (2022 vs 2024) — {query}",
            xaxis={
                "title": "Rank 2022 (1 = highest blended score)",
                "autorange": "reversed",
                "range": [max_rank + 1, 0],
            },
            yaxis={
                "title": "Rank 2024 (1 = highest blended score)",
                "autorange": "reversed",
                "range": [max_rank + 1, 0],
            },
            legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"},
            margin={"l": 60, "r": 30, "t": 80, "b": 60},
            width=1000,
            height=700,
        )
        q_fig.write_html(out_path / f"ranking_stability_{query_slug}.html", include_plotlyjs="cdn")

        visible = [False] * total_traces
        visible[idx * 2] = True
        visible[idx * 2 + 1] = True
        buttons.append(
            {
                "label": query_slug,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title": f"Rank Stability (2022 vs 2024) — {query}",
                    },
                ],
            }
        )

    fig.update_layout(
        title=f"Rank Stability (2022 vs 2024) — {merged_by_query[0][0]}",
        xaxis={
            "title": "Rank 2022 (1 = highest blended score)",
            "autorange": "reversed",
            "range": [global_max_rank + 1, 0],
        },
        yaxis={
            "title": "Rank 2024 (1 = highest blended score)",
            "autorange": "reversed",
            "range": [global_max_rank + 1, 0],
        },
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 1.0,
                "xanchor": "right",
                "y": 1.14,
                "yanchor": "top",
            }
        ],
        legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"},
        margin={"l": 60, "r": 30, "t": 110, "b": 60},
        width=1000,
        height=700,
    )
    fig.write_html(out_path / "rank_stability_rankings.html", include_plotlyjs="cdn")


def _write_query_correlation_figure(result: pd.DataFrame, *, out_path: Path) -> None:
    sorted_result = result.sort_values("spearman_r", ascending=False).reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=sorted_result["query"],
            y=sorted_result["spearman_r"],
            name="Spearman r",
            marker={"color": "#2563eb"},
            customdata=np.column_stack(
                [
                    sorted_result["kendall_tau"].to_numpy(dtype=float),
                    sorted_result["n_cdta_overlap"].to_numpy(dtype=int),
                ]
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Spearman r: %{y:.3f}<br>"
                "Kendall tau: %{customdata[0]:.3f}<br>"
                "CDTA overlap: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sorted_result["query"],
            y=sorted_result["kendall_tau"],
            name="Kendall tau",
            mode="lines+markers",
            marker={"color": "#dc2626", "size": 8},
            line={"color": "#dc2626", "width": 2},
            hovertemplate="<b>%{x}</b><br>Kendall tau: %{y:.3f}<extra></extra>",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Rank-Stability Correlation Summary",
        xaxis={"title": "Query", "tickangle": -35},
        yaxis={"title": "Spearman r"},
        yaxis2={"title": "Kendall tau", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
        margin={"l": 60, "r": 60, "t": 80, "b": 140},
        width=1100,
        height=650,
    )
    fig.write_html(out_path / "query_rank_correlations_summary.html", include_plotlyjs="cdn")


def run_validation(
    *,
    features_2022: Path,
    features_2024: Path,
    output_dir: Path,
    queries: list[str] | None,
    alpha: float = 0.8,
    derive_queries: bool = False,
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
    for df, label in ((df22, "2022"), (df24, "2024")):
        if "competitive_score" not in df.columns:
            raise ValueError(f"Missing 'competitive_score' in {label} feature table.")

    if queries:
        queries = _dedupe_keep_order(queries)
    elif derive_queries:
        queries = _derive_queries_from_act_columns(df22, df24)
        if not queries:
            raise ValueError("No common act_*_storefront columns found to derive queries.")
        queries = _dedupe_keep_order(queries)
    else:
        queries = list(DEFAULT_QUERIES)
    if not queries:
        raise ValueError("No usable queries after de-duplication.")

    emb22, _ = _embed_for_validation(features_2022, vintage="past")
    emb24, _ = _embed_for_validation(features_2024, vintage="present")
    rows: list[dict[str, float | str | int]] = []
    merged_by_query: list[tuple[str, str, pd.DataFrame]] = []
    for query in queries:
        slug = _sanitize_filename(query)
        r22 = _build_query_ranks(
            df22,
            emb22,
            query,
            query_label="2022",
            key_cols=key_cols,
            alpha=alpha,
        )
        r24 = _build_query_ranks(
            df24,
            emb24,
            query,
            query_label="2024",
            key_cols=key_cols,
            alpha=alpha,
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
        merged_by_query.append((query, slug, merged_for_save))

    result = pd.DataFrame(rows).sort_values("spearman_r", ascending=False)
    result.to_csv(output_dir / "query_rank_correlations.csv", index=False)
    _write_interactive_rank_scatter(merged_by_query, out_path=output_dir)
    _write_query_correlation_figure(result, out_path=output_dir)
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
        default=Path("./data/2022/neighborhood_features_final_test.csv"),
        help="Path to 2022 feature table CSV.",
    )
    parser.add_argument(
        "--features-2024",
        type=Path,
        default=Path("../data/processed/neighborhood_features_final.csv"),
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
            "Optional explicit query list. Overrides both --derive-queries and DEFAULT_QUERIES."
        ),
    )
    parser.add_argument(
        "--derive-queries",
        action="store_true",
        help=(
            "Derive queries from common act_*_storefront column names instead of DEFAULT_QUERIES. "
            "Note: this inflates correlations because the vocabulary mirrors the embedded profiles."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Weight on semantic similarity (0–1); competitive-penalty weight = 1 − alpha.",
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
        alpha=args.alpha,
        derive_queries=args.derive_queries,
        clean_output=args.clean_output,
    )

    print("\nRank-stability correlation summary:")
    print(result.to_string(index=False))
    print(f"\nSaved artifacts under: {args.output_dir}")


if __name__ == "__main__":
    main()
