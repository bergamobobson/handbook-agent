"""
╔══════════════════════════════════════════════════════════════╗
║  CHATBOT — main.py                                           ║
║  Interface conversationnelle dans le terminal                ║
╚══════════════════════════════════════════════════════════════╝

Lancer avec :
    uv run python main.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from src.agent.agent import HandbookAgent

load_dotenv()
console = Console()


def main():

    console.print(
        Panel(
            "[bold cyan]Agile Lab Handbook Assistant[/]\n"
            "[dim]Ask questions about the handbook.\n"
            "Type [bold]'exit'[/bold] ou [bold]'quit'[/bold] to leave.[/]",
            border_style="magenta",
        )
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = HandbookAgent(model=llm)
    png_bytes = agent.graph.get_graph().draw_mermaid_png()

    with open("graph.png", "wb") as f:
        f.write(png_bytes)

    print("Saved graph.png")
    while True:
        # ── Lire la question ──────────────────────────────────────────────
        try:
            question = Prompt.ask("\n[bold cyan]❓ Question[/]")
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C ou Ctrl+D → sortie propre
            break

        # ── Commandes de sortie ───────────────────────────────────────────
        if question.strip().lower() in {"exit", "quit", "q"}:
            break

        # ── Question vide ─────────────────────────────────────────────────
        if not question.strip():
            console.print("[dim]Pose une question ou tape 'exit' pour quitter.[/]")
            continue

        # ── Appel à l'agent ───────────────────────────────────────────────
        console.print()
        answer, source = agent(question)

        # ── Affichage de la réponse ───────────────────────────────────────
        if source == "handbook":
            icon = "📚"
            border_color = "green"
        elif source == "conversational":
            icon = "💬"
            border_color = "cyan"
        else:  # off_topic
            icon = "⚠️"
            border_color = "red"

        console.print(
            Panel(
                Markdown(answer),
                title=f"{icon}  [{border_color}]{source.upper()}[/{border_color}]",
                border_style=border_color,
                padding=(1, 2),
            )
        )

    # ── Message de sortie ─────────────────────────────────────────────────
    console.print("\n[dim]À bientôt 👋[/]\n")


if __name__ == "__main__":
    main()
