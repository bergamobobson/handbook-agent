# main.py
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich.console import Console
from evaluation.lash.lash_suites import LASH_TEST_CASES
from evaluation.graph.graph_eval import GraphEvaluator

# Import your evaluators
from evaluation.lash.lash_evaluate import LashEvaluator
from src.agent.agent import HandbookAgent

load_dotenv()
console = Console()


def main():
    graph_struct_config = Path(__file__).resolve().parent / "graph_structure.yaml"

    # ── Init LLM and Agent ──────────────────────────────
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0)
    agent = HandbookAgent(model=llm)

    console.print("\n[bold cyan]╔═════════════════════════════════════════════════════════════════════════════╗[/]")
    console.print("[bold cyan]║  GRAPH EVALUATION — Structured + Nœuds internes                               ║[/]")
    console.print("[bold cyan]╚═══════════════════════════════════════════════════════════════════════════════╝[/]")
    graph_eval = GraphEvaluator(agent, graph_struct_config)
    graph_results = graph_eval.run_graph_evaluation()
    console.print("\n[bold green]Graph Evaluation Results:[/]")
    for k, v in graph_results.items():
        console.print(f"  {k:<15} : {v}")

    console.print("\n[bold cyan]╔═══════════════════════════════════════════════════════════════╗[/]")
    console.print("[bold cyan]║  LASH (Latency - Correctness - Safety - Helpfulness) EVALUATION ║[/]")
    console.print("[bold cyan]╚═════════════════════════════════════════════════════════════════╝[/]")
    lash_eval = LashEvaluator(agent, LASH_TEST_CASES)
    lash_results, mean_lash, lash_pass = lash_eval.eval_lash()
    console.print("\n[bold green]LASH Evaluation Results:[/]")
    console.print(f"  mean_lash_score : {mean_lash:.3f}")
    console.print(f"  pass            : {'✅ PASS' if lash_pass else '❌ FAIL'}")


if __name__ == "__main__":
    main()
