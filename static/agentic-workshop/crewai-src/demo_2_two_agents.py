from __future__ import annotations

from crewai import Agent, Crew, Task

from common import build_local_llm


SYNTHETIC_INBOX: str = """
Email 1
From: Alex, Product Manager
Subject: Metrics review before Friday planning
Body: Can you take a quick look at the latest activation funnel numbers before Friday's planning meeting?
I mostly need to know whether the onboarding drop-off improved after the recent experiment.

Email 2
From: Priya, Data Engineering
Subject: Schema change for events table
Body: Heads up: the events table will add a nullable column called experiment_group tomorrow.
No breaking changes expected, but downstream models should be checked.

Email 3
From: Morgan, Recruiting
Subject: Interview feedback reminder
Body: Could you submit feedback for yesterday's machine learning candidate by end of day?

Email 4
From: Jamie, Analytics Lead
Subject: Dashboard discrepancy
Body: The executive dashboard is showing a 12% revenue drop, but the finance export looks flat.
Can you help determine whether this is a data issue or a real business movement?

Email 5
From: Taylor, Vendor
Subject: New AI analytics platform
Body: We would love to schedule 30 minutes to show you our new AI-powered analytics solution.
"""


def main() -> None:
    llm = build_local_llm()

    triage_agent = Agent(
        role="Email Triage Analyst",
        goal=(
            "Classify incoming work emails by urgency, intent, and required action. "
            "Be concise and practical."
        ),
        backstory=(
            "You are an experienced data science team coordinator. "
            "You know how to distinguish urgent analytical issues from routine updates, reminders, and sales outreach."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    responder_agent = Agent(
        role="Email Response Drafter",
        goal=(
            "Draft short, professional responses or next-action recommendations "
            "based on the triage analyst's classifications."
        ),
        backstory=(
            "You are a concise business communicator supporting a busy data scientist. "
            "You write replies that are clear, polite, and action-oriented."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    triage_task = Task(
        description=(
            "Review the synthetic inbox below. For each email, classify:\n"
            "1. Priority: High, Medium, or Low\n"
            "2. Intent: request, notification, reminder, issue, or sales outreach\n"
            "3. Recommended action\n\n"
            f"Synthetic inbox:\n{SYNTHETIC_INBOX}"
        ),
        expected_output=(
            "A compact table with one row per email and columns: "
            "Email, Priority, Intent, Recommended Action."
        ),
        agent=triage_agent,
    )

    response_task = Task(
        description=(
            "Using the triage result, draft an appropriate response or next action for each email. "
            "For high-priority emails, draft a direct reply. "
            "For low-priority sales outreach, recommend no reply or a polite decline. "
            "Keep each response under 60 words."
        ),
        expected_output=(
            "A numbered list with one concise drafted response or next action per email."
        ),
        agent=responder_agent,
        context=[triage_task],
    )

    crew = Crew(
        agents=[triage_agent, responder_agent],
        tasks=[triage_task, response_task],
        verbose=True,
    )

    result = crew.kickoff()

    print("\n=== Two-agent result: Daily email responder ===\n")
    print(result)


if __name__ == "__main__":
    main()