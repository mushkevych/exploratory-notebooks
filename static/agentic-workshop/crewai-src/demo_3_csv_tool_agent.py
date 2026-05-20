from __future__ import annotations

from pathlib import Path

import pandas as pd
from crewai import Agent, Crew, Task
from crewai.tools import tool

from common import build_local_llm


WORKSPACE_DIR = Path("/app/workspace").resolve()
TOOL_WAS_CALLED = False


def _resolve_workspace_path(path: Path) -> Path:
    """Prevent access outside /app/workspace."""
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    return resolved



@tool
def summarize_support_tickets_csv(filename: str = "support_tickets.csv") -> str:
    """
    Summarize a support-ticket CSV from the local workspace.

    Args:
        filename: Name of the CSV file inside /app/workspace.
                  Defaults to "support_tickets.csv".

    The CSV must include columns:
    ticket_id, customer_segment, product_area, severity, status,
    resolution_hours, csat_score, and description.
    """
    global TOOL_WAS_CALLED
    TOOL_WAS_CALLED = True
    return _summarize_support_tickets_csv_impl(filename)


def _summarize_support_tickets_csv_impl(filename: str = "support_tickets.csv") -> str:
    path = _resolve_workspace_path(WORKSPACE_DIR / filename)
    print(f"[tool] Summarizing support ticket CSV: {path}", flush=True)
    df = pd.read_csv(path)
    print(df.to_string(), flush=True)

    required_columns = {
        "ticket_id",
        "customer_segment",
        "product_area",
        "severity",
        "status",
        "resolution_hours",
        "csat_score",
        "description",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    closed = df[df["status"].str.lower() == "closed"].copy()
    active = df[df["status"].str.lower() != "closed"].copy()

    closed_resolution = pd.to_numeric(closed["resolution_hours"], errors="coerce")
    closed_csat = pd.to_numeric(closed["csat_score"], errors="coerce")

    high_risk = df[
        df["severity"].isin(["High", "Critical"])
        & df["status"].isin(["Open", "In Progress"])
    ][["ticket_id", "customer_segment", "product_area", "severity", "status", "description"]]

    summary = {
        "total_tickets": int(len(df)),
        "status_counts": df["status"].value_counts().to_dict(),
        "severity_counts": df["severity"].value_counts().to_dict(),
        "product_area_counts": df["product_area"].value_counts().to_dict(),
        "customer_segment_counts": df["customer_segment"].value_counts().to_dict(),
        "closed_ticket_count": int(len(closed)),
        "active_ticket_count": int(len(active)),
        "mean_resolution_hours_closed": (
            round(float(closed_resolution.mean()), 2)
            if not closed_resolution.dropna().empty
            else None
        ),
        "mean_csat_closed": (
            round(float(closed_csat.mean()), 2)
            if not closed_csat.dropna().empty
            else None
        ),
        "high_risk_active_tickets": high_risk.to_dict(orient="records"),
    }

    return str(summary)


def main() -> None:
    global TOOL_WAS_CALLED

    llm = build_local_llm(model="ollama/llama4:16x17b")

    analyst = Agent(
        role="Support Operations Analyst",
        goal=(
            "Analyze support-ticket data and identify operational risks, "
            "patterns, and recommended next actions."
        ),
        backstory=(
            "You are a data scientist supporting a customer operations team. "
            "You prefer quantitative summaries, concise risk identification, "
            "and practical recommendations."
        ),
        tools=[summarize_support_tickets_csv],
        llm=llm,
        function_calling_llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    task = Task(
        description=(
            "Call the summarize_support_tickets_csv tool with "
            'filename="support_tickets.csv" to load the data. '
            "Do not estimate or invent metrics; use only values returned by the tool. "
            "Then write a concise operations report for a data science manager. "
            "The report should include: "
            "1. key metrics, "
            "2. the highest-risk tickets, "
            "3. notable patterns, and "
            "4. three recommended next actions."
        ),
        expected_output=(
            "A concise markdown report with sections: Key Metrics, Highest-Risk Tickets, "
            "Patterns, and Recommended Next Actions."
        ),
        agent=analyst,
    )

    crew = Crew(
        agents=[analyst],
        tasks=[task],
        verbose=True,
    )

    result = crew.kickoff()

    if not TOOL_WAS_CALLED:
        raise ValueError("\n[error] Model did not invoke the tool. \n")

    print("\n=== Demo 3 result: CSV analysis with a custom tool ===\n")
    print(result)


if __name__ == "__main__":
    main()
    raise SystemExit(0)
