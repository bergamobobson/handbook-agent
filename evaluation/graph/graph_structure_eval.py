import mlflow
import yaml
from pathlib import Path
from rich.console import Console
from src.agent.agent import HandbookAgent

console = Console()


class GraphStructureEvaluator:
    """
    Évalue la structure d'un graph LangGraph par rapport à un YAML attendu
    et logue les métriques dans MLflow.
    """

    def __init__(self, agent: HandbookAgent, yaml_path: str):
        self.agent = agent
        self.yaml_path = Path(yaml_path)
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"YAML config not found: {self.yaml_path}")

        self.expected = self._load_and_build_expected()

    # ── PRIVATE ────────────────────────────────────────────────
    def _load_and_build_expected(self) -> dict:
        """Charge le YAML et construit la structure attendue exploitable."""
        with open(self.yaml_path) as f:
            cfg = yaml.safe_load(f)

        return {
            "nodes": set(cfg["nodes"]["expected"]),
            "n_nodes": cfg["nodes"]["count"],
            "n_tool_nodes": cfg["tool_nodes"]["count"],
            "direct_edges": {tuple(e) for e in cfg["direct_edges"]},
            "conditional_edges": cfg["conditional_edges"],
        }

    def _eval_structure(self) -> dict:
        """Compare la structure réelle du graph avec celle attendue."""
        console.print("\n[cyan]── Éval structure du graph ──────────────────────────[/]")

        g = self.agent.graph.get_graph()
        actual_nodes = set(g.nodes.keys()) - {"__start__", "__end__"}
        actual_edges = {(e.source, e.target) for e in g.edges}
        checks = {}

        # 1. Nombre total de nœuds
        n_actual = len(actual_nodes)
        count_ok = n_actual == self.expected["n_nodes"]
        checks["node_count"] = (count_ok, f"attendu={self.expected['n_nodes']} trouvé={n_actual}")
        console.print(f"\n  {'✅' if count_ok else '❌'} Nombre de nœuds : {n_actual} / {self.expected['n_nodes']}")

        # 2. ToolNodes
        n_tool = sum(
            1 for name, node in g.nodes.items()
            if "tool" in name.lower()
            or (hasattr(node, "data") and "ToolNode" in type(node.data).__name__)
        )
        tool_ok = n_tool == self.expected["n_tool_nodes"]
        checks["tool_node_count"] = (tool_ok, f"attendu={self.expected['n_tool_nodes']} trouvé={n_tool}")
        console.print(f"  {'✅' if tool_ok else '❌'} ToolNodes : {n_tool} / {self.expected['n_tool_nodes']}")

       # 3. Noms de nœuds
        missing = self.expected["nodes"] - actual_nodes
        extra = actual_nodes - self.expected["nodes"]
        nodes_ok = len(missing) == 0
        checks["nodes_present"] = (
            nodes_ok,
            f"manquants={sorted(missing) or 'empty'} inattendus={sorted(extra) or 'empty'}",
        )
        console.print(f"  {'✅' if nodes_ok else '❌'} Nœuds présents : {sorted(actual_nodes)}")
        if missing:
            console.print(f"       [red]Manquants  : {sorted(missing)}[/]")
        if extra:
            console.print(f"       [yellow]Inattendus : {sorted(extra)}[/]")

        # 4. Direct edges
        missing_direct = self.expected["direct_edges"] - actual_edges
        direct_ok = len(missing_direct) == 0
        checks["direct_edges"] = (
            direct_ok,
            f"manquants={[f'{s}→{t}' for s, t in missing_direct] or '∅'}",
        )
        console.print(f"\n  {'✅' if direct_ok else '❌'} Connections directes :")
        for src, dst in sorted(self.expected["direct_edges"]):
            present = (src, dst) in actual_edges
            console.print(f"       {'✅' if present else '❌'}  {src} → {dst}")

        # 5. Conditional edges
        all_cond_ok = True
        console.print(f"\n  Connections conditionnelles :")
        for src, expected_dsts in self.expected["conditional_edges"].items():
            actual_dsts = [t for (s, t) in actual_edges if s == src]
            for dst in expected_dsts:
                present = dst in actual_dsts
                if not present:
                    all_cond_ok = False
                console.print(f"       {'✅' if present else '❌'}  {src} --[cond]--> {dst}")
        checks["conditional_edges"] = (all_cond_ok, "")

        all_passed = all(ok for ok, _ in checks.values())

        return {
            "nodes_ok": nodes_ok,
            "tool_nodes_ok": tool_ok,
            "direct_edges_ok": direct_ok,
            "conditional_edges_ok": all_cond_ok,
            "node_count_ok": count_ok,
            "all_ok": all_passed,
            "n_nodes": n_actual,
            "n_tool_nodes": n_tool,
            "n_edges": len(actual_edges),
            "missing_nodes": list(missing),
            "extra_nodes": list(extra),
            "missing_direct_edges": [f"{s}→{t}" for s, t in missing_direct],
        }

    # ── PUBLIC ───────────────────────────────────────────────
    def eval_structured(self) -> dict:
        """Exécute l’évaluation et logue les métriques dans MLflow."""
        mlflow.set_experiment("HandbookAgent_StructureEval")

        with mlflow.start_run(run_name="StructureEval", nested=True) as run:
            result = self._eval_structure()

            mlflow.log_metrics({
                "nodes_ok": int(result["nodes_ok"]),
                "tool_nodes_ok": int(result["tool_nodes_ok"]),
                "direct_edges_ok": int(result["direct_edges_ok"]),
                "conditional_edges_ok": int(result["conditional_edges_ok"]),
                "structure_all_ok": int(result["all_ok"]),
                "n_nodes": result["n_nodes"],
                "n_tool_nodes": result["n_tool_nodes"],
                "n_edges": result["n_edges"],
            })

            console.print(f"MLflow run : {run.info.run_id}")

        return result