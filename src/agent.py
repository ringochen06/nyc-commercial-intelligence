"""
Agentic tool-use layer: Claude decides which DuckDB SQL filters to apply
based on the user's natural-language query, then summarizes filtered results.

The model receives two tool definitions:
  1. run_sql  - execute a read-only SQL query against the neighborhood table
  2. done     - return the final answer to the user

Hard filters (already applied via the Streamlit UI) are passed in as a
pre-filtered DataFrame.  The model only sees rows that survived hard filtering
and can run additional analytical queries on that subset.
"""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic
import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Tool schemas exposed to Claude ──────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "name": "run_sql",
        "description": (
            "Execute a read-only DuckDB SQL query against the table `neighborhoods`. "
            "The table contains one row per NYC CDTA with columns including: "
            "neighborhood, cd, borough, area_km2, storefront_filing_count, storefront_density_per_km2, "
            "act_*_storefront (counts by primary business activity), category_diversity, category_entropy, "
            "avg_pedestrian, peak_pedestrian, pedestrian_count_points, "
            "subway_station_count, subway_density_per_km2, commercial_activity_score, transit_activity_score, "
            "and neighborhood profile fields when present (e.g. nfh_median_income, "
            "pct_bachelors_plus, commute_public_transit, food_services, total_businesses, "
            "nfh_overall_score, pop_black, pop_hispanic, pop_asian, total_population_proxy). "
            "Only SELECT statements are allowed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A SELECT SQL query to run against the neighborhoods table.",
                }
            },
            "required": ["sql"],
        },
    },
    {
        "name": "done",
        "description": (
            "Call this tool when you have finished analyzing the data and are "
            "ready to present your final answer to the user."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Your final markdown-formatted answer to the user.",
                }
            },
            "required": ["answer"],
        },
    },
]

# ── SQL execution (sandboxed to SELECT) ─────────────────────────────────────


def _execute_sql(df: pd.DataFrame, sql: str) -> str:
    """Run *sql* against *df* registered as ``neighborhoods`` in DuckDB."""
    normalized = sql.strip().rstrip(";").upper()
    if not normalized.startswith("SELECT"):
        return "Error: only SELECT queries are allowed."
    try:
        con = duckdb.connect()
        con.register("neighborhoods", df)
        result = con.execute(sql).fetchdf()
        con.close()
    except Exception as exc:
        return f"SQL error: {exc}"

    if len(result) > 150:
        result = result.head(150).copy()
        note = "\n\n_(first 150 rows only)_"
    else:
        note = ""

    try:
        return result.to_markdown(index=False) + note
    except ImportError:
        # pandas needs optional `tabulate` for to_markdown; keep agent usable without it
        return result.to_string(index=False) + note


# ── Agent loop ──────────────────────────────────────────────────────────────


def run_agent(
    user_query: str,
    df: pd.DataFrame,
    *,
    model: str | None = None,
    max_turns: int = 16,
) -> str:
    """
    Send *user_query* to Claude with tool access to the filtered DataFrame.

    Returns the model's final markdown answer.
    """
    # Default: Sonnet 4.6 (alias). Override with ANTHROPIC_MODEL; older IDs may be deprecated.
    model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    client = anthropic.Anthropic()

    system_prompt = (
        "You are a NYC commercial real-estate analyst. The user is choosing a "
        "neighborhood for a new business. You have access to a DuckDB table "
        "called `neighborhoods` that contains pre-filtered NYC neighborhood "
        "data (hard filters like borough and minimum subway count were already "
        "applied). Use the run_sql tool at most a few times (e.g. top-N by "
        "commercial_activity_score, or a simple aggregate). You MUST finish by "
        "calling the done tool with your final markdown answer—do not loop on "
        "exploratory SQL indefinitely. If one query is enough, call done "
        "immediately after. Always refer to neighborhoods by their full name."
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_query},
    ]

    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        # Collect assistant text + tool_use blocks
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        # Check for tool use
        tool_uses = [b for b in assistant_content if b.type == "tool_use"]
        if not tool_uses:
            # Model responded with plain text (no tool call) — treat as final
            text_parts = [b.text for b in assistant_content if hasattr(b, "text")]
            return "\n".join(text_parts) if text_parts else "No response."

        # Process each tool call
        tool_results: list[dict[str, Any]] = []
        for tu in tool_uses:
            if tu.name == "done":
                return tu.input.get("answer", "")
            elif tu.name == "run_sql":
                sql = tu.input.get("sql", "")
                result = _execute_sql(df, sql)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tu.id, "content": result}
                )
            else:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": f"Unknown tool: {tu.name}",
                    }
                )

        messages.append({"role": "user", "content": tool_results})

    return (
        "Agent reached the turn limit without calling **done**. "
        "Try again, or increase `max_turns` in `run_agent` if the task needs "
        "more SQL steps."
    )
