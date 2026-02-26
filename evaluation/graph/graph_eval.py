import sys
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from rich.console import Console

from evaluation.graph.graph_structure_eval import GraphStructureEvaluator
from evaluation.graph.nodes import (CLASSIFY_CASES, GRADE_CASES,
                                    RETRIEVE_CASES, ROUTING_CASES)

load_dotenv()
console = Console()


class GraphEvaluator:
    """
    Évalue les nœuds internes d'un LangGraph pour un agent donné.

    Évals disponibles :
      1. classify   → retourne-t-il le bon intent ?
      2. retrieve   → récupère-t-il des docs pertinents ?
      3. grade      → évalue-t-il correctement la pertinence ?
      4. routing    → le graph prend-il le bon chemin ?
      5. structure  → évalue la structure globale du graph

    Capture des états intermédiaires via agent.graph.stream(stream_mode="updates").
    """

    def __init__(self, agent, graph_struct_config):
        self.agent = agent
        self.struct_eval = GraphStructureEvaluator(agent, graph_struct_config)

    # ── PRIVATE ────────────────────────────────────────────────
    def _stream_graph(self, question: str, thread_id: str) -> dict:
        """Exécute le graph et capture les snapshots par nœud."""
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = {}

        for update in self.agent.graph.stream(
            {"messages": [HumanMessage(content=question)], "source": "handbook"},
            config=config,
            stream_mode="updates",
        ):
            for node_name, node_output in update.items():
                snapshot[node_name] = node_output

        return snapshot

    def _eval_classify(self) -> pd.DataFrame:
        console.print(
            "\n[cyan]── Éval nœud : classify ─────────────────────────────[/]"
        )
        rows = []
        for i, tc in enumerate(CLASSIFY_CASES):
            snapshot = self._stream_graph(tc["input"], thread_id=f"eval-cls-{i}")
            actual_intent = snapshot.get("classify", {}).get("intent", "MISSING")
            expected_intent = tc["expected_intent"]
            passed = actual_intent == expected_intent

            console.print(
                f"  {'✅' if passed else '❌'} [{actual_intent:<16}] {tc['input'][:55]}"
            )
            if not passed:
                console.print(f"       [red]→ attendu : {expected_intent}[/]")

            rows.append(
                {
                    "input": tc["input"],
                    "expected_intent": expected_intent,
                    "actual_intent": actual_intent,
                    "pass": passed,
                }
            )
        df = pd.DataFrame(rows)
        console.print(
            f"\n  Accuracy : [bold]{df['pass'].mean():.1%}[/]  ({df['pass'].sum()}/{len(df)})"
        )
        return df

    def _eval_retrieve(self) -> pd.DataFrame:
        console.print(
            "\n[cyan]── Éval nœud : retrieve ─────────────────────────────[/]"
        )
        rows = []
        for i, tc in enumerate(RETRIEVE_CASES):
            snapshot = self._stream_graph(tc["input"], thread_id=f"eval-ret-{i}")
            documents = snapshot.get("retrieve", {}).get("documents", [])
            n_docs = len(documents)

            all_content = " ".join(
                (
                    doc.page_content.lower()
                    if isinstance(doc, Document)
                    else str(doc).lower()
                )
                for doc in documents
            )
            keywords_found = [
                kw for kw in tc["relevant_keywords"] if kw.lower() in all_content
            ]
            passed = len(keywords_found) > 0

            console.print(
                f"  {'✅' if passed else '❌'} {n_docs} docs  keywords={keywords_found or '∅'}  {tc['input'][:45]}"
            )
            rows.append(
                {
                    "input": tc["input"],
                    "n_docs_retrieved": n_docs,
                    "keywords_found": ", ".join(keywords_found),
                    "pass": passed,
                }
            )
        df = pd.DataFrame(rows)
        console.print(
            f"\n  Accuracy : [bold]{df['pass'].mean():.1%}[/]  ({df['pass'].sum()}/{len(df)})"
        )
        return df

    def _eval_grade(self) -> pd.DataFrame:
        console.print(
            "\n[cyan]── Éval nœud : grade ────────────────────────────────[/]"
        )
        rows = []
        for i, tc in enumerate(GRADE_CASES):
            state = {
                "messages": [HumanMessage(content=tc["input"])],
                "documents": [Document(page_content=d) for d in tc["documents"]],
            }
            result = self.agent._grade(state)
            actual = result.get("relevant", None)
            expected = tc["expected_relevant"]
            passed = actual == expected

            doc_preview = tc["documents"][0][:60] + "..."
            console.print(
                f"  {'✅' if passed else '❌'} relevant={str(actual):<6} (attendu={expected})  {doc_preview}"
            )

            rows.append(
                {
                    "input": tc["input"],
                    "document_preview": tc["documents"][0][:80],
                    "expected_relevant": expected,
                    "actual_relevant": actual,
                    "pass": passed,
                }
            )
        df = pd.DataFrame(rows)
        console.print(
            f"\n  Accuracy : [bold]{df['pass'].mean():.1%}[/]  ({df['pass'].sum()}/{len(df)})"
        )
        return df

    def _eval_routing(self) -> pd.DataFrame:
        console.print(
            "\n[cyan]── Éval nœud : routing ──────────────────────────────[/]"
        )
        rows = []
        for i, tc in enumerate(ROUTING_CASES):
            config = {"configurable": {"thread_id": f"eval-route-{i}"}}
            actual_path = []

            for update in self.agent.graph.stream(
                {"messages": [HumanMessage(content=tc["input"])], "source": "handbook"},
                config=config,
                stream_mode="updates",
            ):
                for node_name in update.keys():
                    actual_path.append(node_name)

            expected_path = tc["expected_path"]
            passed = actual_path == expected_path

            console.print(f"  {'✅' if passed else '❌'} {tc['description']}")
            console.print(f"       attendu : {' → '.join(expected_path)}")
            if not passed:
                console.print(f"       [red]obtenu  : {' → '.join(actual_path)}[/]")
            else:
                console.print(f"       obtenu  : {' → '.join(actual_path)}")

            rows.append(
                {
                    "input": tc["input"],
                    "description": tc["description"],
                    "expected_path": " → ".join(expected_path),
                    "actual_path": " → ".join(actual_path),
                    "pass": passed,
                }
            )
        df = pd.DataFrame(rows)
        console.print(
            f"\n  Accuracy : [bold]{df['pass'].mean():.1%}[/]  ({df['pass'].sum()}/{len(df)})"
        )
        return df

    # ── PUBLIC ───────────────────────────────────────────────
    def run_graph_evaluation(self) -> dict:
        """Exécute toutes les évaluations et log MLflow + CSV."""

        mlflow.set_experiment("HandbookAgent_GraphEval")
        with mlflow.start_run(
            run_name=f"GraphEval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ) as run:
            struct = self.struct_eval.eval_structured()
            cls_df = self._eval_classify()
            ret_df = self._eval_retrieve()
            grd_df = self._eval_grade()
            rout_df = self._eval_routing()

            acc_classify = cls_df["pass"].mean()
            acc_retrieve = ret_df["pass"].mean()
            acc_grade = grd_df["pass"].mean()
            acc_routing = rout_df["pass"].mean()
            structure_ok = int(struct["all_ok"])
            graph_score = (
                acc_classify + acc_retrieve + acc_grade + acc_routing + structure_ok
            ) / 5

            # Log metrics
            mlflow.log_metrics(
                {
                    "classify_accuracy": acc_classify,
                    "retrieve_accuracy": acc_retrieve,
                    "grade_accuracy": acc_grade,
                    "routing_accuracy": acc_routing,
                    "structure_nodes_ok": int(struct["nodes_ok"]),
                    "structure_tool_nodes_ok": int(struct["tool_nodes_ok"]),
                    "structure_direct_edges_ok": int(struct["direct_edges_ok"]),
                    "structure_cond_edges_ok": int(struct["conditional_edges_ok"]),
                    "structure_all_ok": structure_ok,
                    "n_nodes": struct["n_nodes"],
                    "n_tool_nodes": struct["n_tool_nodes"],
                    "n_edges": struct["n_edges"],
                    "graph_score": graph_score,
                }
            )

            # Export CSV
            tmp_dir = Path("evaluation/data/graph/outputs/")
            cls_df.to_csv(tmp_dir / "eval_classify.csv", index=False)
            ret_df.to_csv(tmp_dir / "eval_retrieve.csv", index=False)
            grd_df.to_csv(tmp_dir / "eval_grade.csv", index=False)
            rout_df.to_csv(tmp_dir / "eval_routing.csv", index=False)
            mlflow.log_artifacts(tmp_dir, artifact_path="node_results")

            # Résumé console
            console.print(
                "\n[bold]── Résumé ────────────────────────────────────────────────────[/]"
            )
            console.print(f"\n  {'Nœud':<20} {'Accuracy'}")
            console.print(f"  {'─'*35}")
            for name, score in [
                ("classify", acc_classify),
                ("retrieve", acc_retrieve),
                ("grade", acc_grade),
                ("routing", acc_routing),
                ("structure", structure_ok),
            ]:
                icon = "✅" if score >= 0.8 else ("⚠️ " if score >= 0.6 else "❌")
                console.print(f"  {name:<20} {score:.1%}   {icon}")

            console.print(f"  {'─'*35}")
            console.print(
                f"  {'GRAPH GLOBAL':<20} {graph_score:.1%}   {'✅' if graph_score >= 0.8 else '❌'}"
            )
            console.print(f"\n  MLflow run : {run.info.run_id}")

        return {
            "classify": acc_classify,
            "retrieve": acc_retrieve,
            "grade": acc_grade,
            "routing": acc_routing,
            "structure": structure_ok,
            "graph": graph_score,
        }
