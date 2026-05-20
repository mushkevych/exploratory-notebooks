from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from textwrap import dedent

from crewai import Agent, Crew, Task
from crewai.flow.flow import Flow, listen, router, start

from common import build_local_llm


class TicketRoute(str, Enum):
    """Supported routing destinations for the demo."""

    AUTHENTICATION = "authentication"
    ANALYTICS = "analytics"
    BILLING = "billing"
    DATA_PIPELINE = "data_pipeline"
    SALES = "sales"
    GENERAL = "general"


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


def classify_ticket(ticket: SupportTicket) -> TicketRoute:
    """
    Deterministically classify a support ticket.

    This keeps routing predictable for a live workshop. The routed branch
    still uses a specialized CrewAI agent to produce the response.
    """
    text = f"{ticket.subject} {ticket.body}".lower()

    if any(keyword in text for keyword in ["sso", "login", "password", "identity provider", "reset"]):
        return TicketRoute.AUTHENTICATION

    if any(keyword in text for keyword in ["dashboard", "metric", "revenue", "retention", "finance export"]):
        return TicketRoute.ANALYTICS

    if any(keyword in text for keyword in ["invoice", "billing", "purchase order", "payable"]):
        return TicketRoute.BILLING

    if any(keyword in text for keyword in ["pipeline", "partition", "feature table", "ingestion", "model output"]):
        return TicketRoute.DATA_PIPELINE

    if any(keyword in text for keyword in ["platform introduction", "partnership", "discovery call", "vendor"]):
        return TicketRoute.SALES

    return TicketRoute.GENERAL


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


def run_specialized_agent(ticket: SupportTicket, route: TicketRoute) -> str:
    """Create and run the specialized agent for a single routed ticket."""
    llm = build_local_llm()

    agent_config_by_route: dict[TicketRoute, dict[str, str]] = {
        TicketRoute.AUTHENTICATION: {
            "role": "Authentication Support Agent",
            "goal": "Handle SSO, login, password reset, and account-access issues.",
            "backstory": (
                "You are cautious and precise. You avoid asking for passwords or secrets. "
                "You focus on practical access-restoration steps and escalation criteria."
            ),
        },
        TicketRoute.ANALYTICS: {
            "role": "Analytics Support Agent",
            "goal": "Handle dashboard, metric, and reporting discrepancies.",
            "backstory": (
                "You are a data analyst. You reason about data freshness, definitions, "
                "filters, exports, and reconciliation checks."
            ),
        },
        TicketRoute.BILLING: {
            "role": "Billing Support Agent",
            "goal": "Handle invoice, billing-address, purchase-order, and payment questions.",
            "backstory": (
                "You are a professional billing-support specialist. "
                "You are concise, polite, and specific about next steps."
            ),
        },
        TicketRoute.DATA_PIPELINE: {
            "role": "Data Pipeline Support Agent",
            "goal": "Handle ingestion, partition, feature-table, and stale-data issues.",
            "backstory": (
                "You are a data engineering support analyst. "
                "You focus on incident triage, freshness checks, and downstream impact."
            ),
        },
        TicketRoute.SALES: {
            "role": "Vendor Triage Agent",
            "goal": "Handle vendor outreach without distracting the data science team.",
            "backstory": (
                "You protect the team's focus. You recommend no response, deferral, "
                "or a short polite decline when outreach is not relevant."
            ),
        },
        TicketRoute.GENERAL: {
            "role": "General Support Agent",
            "goal": "Handle general support requests and recommend an appropriate next action.",
            "backstory": (
                "You are a practical support coordinator. "
                "You ask for the minimum information needed to move the request forward."
            ),
        },
    }

    config = agent_config_by_route[route]

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
            "Keep the answer under 90 words. Include a recommended priority: High, Medium, or Low.\\n\\n"
            f"Route: {route.value}\\n"
            f"{render_ticket(ticket)}"
        ),
        expected_output=(
            "A short response with: priority, recommended action, and draft reply if appropriate."
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
    Demo 4: route each synthetic ticket to a specialized agent.

    The routing decision is deterministic Python code for demo reliability.
    The branch action is handled by the corresponding CrewAI agent.
    """

    @start()
    def load_tickets(self) -> list[SupportTicket]:
        print(f"Loaded {len(SAMPLE_TICKETS)} synthetic tickets.")
        return SAMPLE_TICKETS

    @router(load_tickets)
    def route_tickets(self, tickets: list[SupportTicket]) -> str:
        results: list[str] = []

        for ticket in tickets:
            route = classify_ticket(ticket)
            print(f"\\nRouting {ticket.ticket_id}: {route.value}")

            response = run_specialized_agent(ticket=ticket, route=route)

            results.append(
                dedent(
                    f"""
                    ## {ticket.ticket_id}: {ticket.subject}

                    **Route:** {route.value}

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
        for ticket in SAMPLE_TICKETS:
            route = classify_ticket(ticket).value
            route_counts[route] = route_counts.get(route, 0) + 1

        summary_lines = [
            "# Demo 4 Result: Routed Support-Ticket Workflow",
            "",
            "## Route Counts",
        ]

        for route, count in sorted(route_counts.items()):
            summary_lines.append(f"- {route}: {count}")

        summary_lines.extend(["", "## Routed Ticket Outputs", ""])
        summary_lines.extend(routed_results)

        return "\\n".join(summary_lines)


def main() -> None:
    flow = TicketRoutingFlow()
    result = flow.kickoff()

    print("\\n=== Demo 4 result: Routed ticket workflow ===\\n")
    print(result)


if __name__ == "__main__":
    main()
    raise SystemExit(0)
