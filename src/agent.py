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
            "The table contains one row per NYC neighborhood with columns: "
            "neighborhood, cd, borough, area_km2, total_poi, unique_poi, "
            "category_diversity, num_retail, food, other, retail, ratio_retail, "
            "category_entropy, avg_pedestrian, peak_pedestrian, pedestrian_count_points, "
            "subway_station_count, poi_density_per_km2, retail_density_per_km2, "
            "subway_density_per_km2, commercial_activity_score, transit_activity_score, "
            "and neighborhood profile fields when present (e.g. median_household_income, "
            "pct_bachelors_plus, commute_public_transit, food_services, total_businesses, "
            "pct_hispanic, pct_black, pct_asian). "
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
        return result.to_markdown(index=False)
    except Exception as exc:
        return f"SQL error: {exc}"


# ── Agent loop ──────────────────────────────────────────────────────────────

def run_agent(
    user_query: str,
    df: pd.DataFrame,
    *,
    model: str | None = None,
    max_turns: int = 6,
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
        "applied). Use the run_sql tool to explore the data, compute rankings, "
        "or answer analytical questions. When you have enough information, call "
        "the done tool with a concise, helpful markdown answer. Always refer to "
        "neighborhoods by their full name."
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

    return "Agent reached maximum turns without a final answer."
