from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Literal

from crewai import Agent, Crew, Task
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel, Field

from common import build_local_llm


RouteName = Literal["access", "data", "commercial", "general"]


class TicketClassification(BaseModel):
    """
    Structured LLM output for ticket routing.

    The route field is constrained to a small Literal set so downstream routing
    can remain deterministic after the LLM classification step.
    """

    route: RouteName = Field(
        description=(
            "Routing destination. Must be one of: access, data, commercial, general."
        )
    )
    priority: Literal["High", "Medium", "Low"] = Field(
        description="Recommended priority for the ticket."
    )
    rationale: str = Field(
        description="One concise sentence explaining why this route was selected."
    )


@dataclass(frozen=True)
class SupportTicket:
    """Synthetic support-ticket record used by the routing demo."""

    ticket_id: str
    sender: str
    customer_segment: str
    subject: str
    body: str


SAMPLE_TICKETS: list[SupportTicket] = [
    SupportTicket(
        ticket_id="R-1001",
        sender="Avery Chen",
        customer_segment="Enterprise",
        subject="SSO users cannot log in after certificate rotation",
        body=(
            "Several users are blocked from logging in through SSO after we rotated "
            "our identity provider certificate this morning."
        ),
    ),
    SupportTicket(
        ticket_id="R-1002",
        sender="Morgan Lee",
        customer_segment="Enterprise",
        subject="Revenue dashboard differs from finance export",
        body=(
            "The executive dashboard shows a 12% revenue decline, but the finance "
            "export is flat. We need help determining whether this is a data issue."
        ),
    ),
    SupportTicket(
        ticket_id="R-1003",
        sender="Priya Shah",
        customer_segment="SMB",
        subject="Invoice has the wrong billing address",
        body=(
            "The latest invoice lists our old billing address. Can you update it "
            "and regenerate the invoice PDF?"
        ),
    ),
    SupportTicket(
        ticket_id="R-1004",
        sender="Jordan Rivera",
        customer_segment="Mid-market",
        subject="Nightly ingestion did not load the latest partition",
        body=(
            "Our downstream model is stale because the nightly data pipeline appears "
            "to be missing yesterday's partition."
        ),
    ),
    SupportTicket(
        ticket_id="R-1005",
        sender="Taylor Brooks",
        customer_segment="Vendor",
        subject="AI analytics platform introduction",
        body=(
            "We would love to schedule 30 minutes to show your team our new "
            "AI-powered analytics platform."
        ),
    ),
    SupportTicket(
        ticket_id="R-1006",
        sender="Sam Patel",
        customer_segment="SMB",
        subject="Password reset emails are delayed",
        body=(
            "Users are waiting 15 to 20 minutes to receive password reset emails. "
            "This is causing login friction."
        ),
    ),
    SupportTicket(
        ticket_id="R-1007",
        sender="Dana Williams",
        customer_segment="Enterprise",
        subject="Batch scoring API returning 500 errors",
        body=(
            "Our production batch scoring jobs started receiving API 500 errors "
            "after the latest deployment."
        ),
    ),
    SupportTicket(
        ticket_id="R-1008",
        sender="Chris Nguyen",
        customer_segment="Mid-market",
        subject="Question about purchase order number on invoice",
        body=(
            "Can you add our purchase order number to future invoices? "
            "We also need confirmation that this month's invoice is payable."
        ),
    ),
    SupportTicket(
        ticket_id="R-1009",
        sender="Riley Stone",
        customer_segment="Enterprise",
        subject="Dashboard is slow for long date ranges",
        body=(
            "The retention dashboard takes more than 45 seconds to load when the "
            "date range is longer than twelve months."
        ),
    ),
    SupportTicket(
        ticket_id="R-1010",
        sender="Jamie Kim",
        customer_segment="SMB",
        subject="Can we reschedule onboarding?",
        body=(
            "We need to reschedule our onboarding session because two stakeholders "
            "are out of office this week."
        ),
    ),
    SupportTicket(
        ticket_id="R-1011",
        sender="Casey Smith",
        customer_segment="Enterprise",
        subject="Feature table missing latest customer attributes",
        body=(
            "The feature table used by our churn model is missing the latest customer "
            "attributes. The model output may be stale."
        ),
    ),
    SupportTicket(
        ticket_id="R-1012",
        sender="Quinn Garcia",
        customer_segment="Vendor",
        subject="Partnership opportunity for your data science team",
        body=(
            "We help data science teams accelerate model deployment. Are you available "
            "for a short discovery call next week?"
        ),
    ),
]


def render_ticket(ticket: SupportTicket) -> str:
    """Create a compact prompt block for a ticket."""
    return dedent(
        f"""
        Ticket ID: {ticket.ticket_id}
        Sender: {ticket.sender}
        Customer segment: {ticket.customer_segment}
        Subject: {ticket.subject}
        Body: {ticket.body}
        """
    ).strip()


def classify_ticket_with_llm(ticket: SupportTicket) -> TicketClassification:
    """
    Classify a ticket using an LLM-backed CrewAI task.

    The task returns a Pydantic object. Its route field is constrained by
    RouteName, which is a typing.Literal union.
    """
    llm = build_local_llm()

    classifier = Agent(
        role="Ticket Routing Classifier",
        goal=(
            "Classify support tickets into exactly one routing destination: "
            "access, data, commercial, or general."
        ),
        backstory=(
            "You are a precise support-operations router. "
            "You return structured decisions only and avoid inventing new routes."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    task = Task(
        description=(
            "Classify the ticket into exactly one route.\n\n"
            "Route definitions:\n"
            "- access: authentication, SSO, login, password reset, account access, identity provider issues.\n"
            "- data: dashboards, analytics, metrics, APIs, data pipelines, feature tables, stale data, model-output issues.\n"
            "- commercial: invoices, billing, purchase orders, payment questions, vendor outreach, sales outreach, partnership requests.\n"
            "- general: scheduling, onboarding, vague requests, or anything that does not fit the other routes.\n\n"
            "Return only the structured classification object requested by the task.\n\n"
            f"{render_ticket(ticket)}"
        ),
        expected_output=(
            "A structured classification with route, priority, and rationale."
        ),
        output_pydantic=TicketClassification,
        agent=classifier,
    )

    crew = Crew(
        agents=[classifier],
        tasks=[task],
        verbose=True,
    )

    result = crew.kickoff()

    # CrewAI versions differ slightly in how they expose structured task output.
    # These branches keep the demo robust across common versions.
    if isinstance(result, TicketClassification):
        return result

    pydantic_output = getattr(result, "pydantic", None)
    if isinstance(pydantic_output, TicketClassification):
        return pydantic_output

    tasks_output = getattr(result, "tasks_output", None)
    if tasks_output:
        first_task_output = tasks_output[0]
        structured = getattr(first_task_output, "pydantic", None)
        if isinstance(structured, TicketClassification):
            return structured

    # Fallback: parse JSON/dict-like raw output through Pydantic.
    raw = str(result)
    return TicketClassification.model_validate_json(raw)


def run_specialized_agent(
    ticket: SupportTicket,
    classification: TicketClassification,
) -> str:
    """
    Run one of the 3 + General response agents.

    Response-agent routes:
    1. access
    2. data
    3. commercial
    4. general
    """
    llm = build_local_llm()

    agent_config_by_route: dict[RouteName, dict[str, str]] = {
        "access": {
            "role": "Access Support Agent",
            "goal": "Handle authentication, SSO, login, password reset, and account-access issues.",
            "backstory": (
                "You are cautious and precise. You avoid asking for passwords or secrets. "
                "You focus on practical access-restoration steps and escalation criteria."
            ),
        },
        "data": {
            "role": "Data Support Agent",
            "goal": "Handle analytics, dashboard, API, data-pipeline, and stale-data issues.",
            "backstory": (
                "You are a data analyst and data engineering support partner. "
                "You reason about freshness, definitions, API failures, partitions, and downstream impact."
            ),
        },
        "commercial": {
            "role": "Commercial Support Agent",
            "goal": "Handle billing, invoice, purchase-order, payment, vendor, and sales-outreach requests.",
            "backstory": (
                "You are a professional commercial operations specialist. "
                "You are concise, polite, and specific about next steps. "
                "For vendor outreach, you protect the team's focus."
            ),
        },
        "general": {
            "role": "General Support Agent",
            "goal": "Handle general support requests and recommend an appropriate next action.",
            "backstory": (
                "You are a practical support coordinator. "
                "You ask for the minimum information needed to move the request forward."
            ),
        },
    }

    config = agent_config_by_route[classification.route]

    agent = Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    task = Task(
        description=(
            "Draft a concise response or next-action recommendation for this routed ticket. "
            "Keep the answer under 90 words.\n\n"
            f"LLM classification:\n"
            f"- Route: {classification.route}\n"
            f"- Priority: {classification.priority}\n"
            f"- Rationale: {classification.rationale}\n\n"
            f"{render_ticket(ticket)}"
        ),
        expected_output=(
            "A short response with priority, recommended action, and draft reply if appropriate."
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)


class TicketRoutingFlow(Flow):
    """
    Demo 4: route each synthetic ticket to one of 3 + General agents.

    The classification decision is made by an LLM call with structured output.
    The structured route is constrained by typing.Literal.
    """

    @start()
    def load_tickets(self) -> list[SupportTicket]:
        print(f"Loaded {len(SAMPLE_TICKETS)} synthetic tickets.")
        return SAMPLE_TICKETS

    @router(load_tickets)
    def classify_and_route_tickets(self, tickets: list[SupportTicket]) -> str:
        results: list[str] = []

        for ticket in tickets:
            print(f"\nClassifying {ticket.ticket_id} with LLM...")
            classification = classify_ticket_with_llm(ticket)

            print(
                f"Routing {ticket.ticket_id}: "
                f"{classification.route} / {classification.priority}"
            )

            response = run_specialized_agent(
                ticket=ticket,
                classification=classification,
            )

            results.append(
                dedent(
                    f"""
                    ## {ticket.ticket_id}: {ticket.subject}

                    **Route:** {classification.route}

                    **Priority:** {classification.priority}

                    **Classification rationale:** {classification.rationale}

                    **Specialized agent output:**
                    {response}
                    """
                ).strip()
            )

        self.state["routed_results"] = results
        return "summarize"

    @listen("summarize")
    def summarize_results(self) -> str:
        routed_results = self.state.get("routed_results", [])

        route_counts: dict[str, int] = {}
        for block in routed_results:
            route_line = next(
                (line for line in block.splitlines() if line.startswith("**Route:**")),
                None,
            )
            if route_line is None:
                route = "unknown"
            else:
                route = route_line.replace("**Route:**", "").strip()
            route_counts[route] = route_counts.get(route, 0) + 1

        summary_lines = [
            "# Demo 4 Result: LLM-Routed Support-Ticket Workflow",
            "",
            "## Route Counts",
        ]

        for route, count in sorted(route_counts.items()):
            summary_lines.append(f"- {route}: {count}")

        summary_lines.extend(["", "## Routed Ticket Outputs", ""])
        summary_lines.extend(routed_results)

        return "\n".join(summary_lines)


def main() -> None:
    flow = TicketRoutingFlow()
    result = flow.kickoff()

    print("\n=== Demo 5 result: LLM-routed ticket workflow ===\n")
    print(result)


if __name__ == "__main__":
    main()
