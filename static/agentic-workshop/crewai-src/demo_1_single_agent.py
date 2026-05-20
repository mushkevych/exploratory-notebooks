from __future__ import annotations

from crewai import Agent, Crew, Task

from common import build_local_llm


def main() -> None:
    llm = build_local_llm()

    analyst = Agent(
        role="Data Science Workshop Assistant",
        goal="Explain local AI agent concepts clearly and concisely for data scientists.",
        backstory=(
            "You are a practical data science assistant. "
            "You prefer reproducible local workflows and concise technical explanations."
        ),
        llm=llm,
        verbose=True,
    )

    task = Task(
        description=(
            "Explain why running CrewAI in Docker while using a locally hosted Ollama model "
            "is useful for a hands-on data science workshop. Keep the answer to 5 bullet points."
        ),
        expected_output="Five concise bullet points.",
        agent=analyst,
    )

    crew = Crew(
        agents=[analyst],
        tasks=[task],
        verbose=True,
    )

    result = crew.kickoff()
    print("\n=== Single-agent result ===\n")
    print(result)


if __name__ == "__main__":
    main()
