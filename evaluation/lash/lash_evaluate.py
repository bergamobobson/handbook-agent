import time
from datetime import datetime

import mlflow
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console

from evaluation.lash.lash_metrics import (
    ALL_SCORERS,
    PASS_THRESHOLDS,
    WEIGHTS,
    latency_to_score,
)
from src.agent.agent import HandbookAgent

load_dotenv()
console = Console()


class LashEvaluator:
    """
    Évalue un agent selon le protocole LASH :
      L — Latency
      A — Correctness
      S — Safety
      H — Helpfulness
    """

    def __init__(self, agent: HandbookAgent, test_cases=None):
        self.agent = agent
        self.test_cases = test_cases
        self.df = pd.DataFrame()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — COLLECT
    # ══════════════════════════════════════════════════════════════════════════

    def collect(self) -> pd.DataFrame:
        """Exécute toutes les questions et collecte les réponses + latences."""
        console.print(
            f"\n[bold cyan]── Phase 1 : Collecte ({len(self.test_cases)} test cases) ──────────────[/]"
        )

        rows = []
        for i, tc in enumerate(self.test_cases):
            label = tc["input"][:55] + "..." if len(tc["input"]) > 55 else tc["input"]
            console.print(f"  [{i+1:02d}/{len(self.test_cases)}] {label}")

            start = time.time()
            answer, _ = self.agent(tc["input"], thread_id=f"eval-{i}")
            latency = time.time() - start
            l_score = latency_to_score(latency)

            console.print(
                f"         latency=[bold]{latency:.2f}s[/]  L_score={l_score:.2f}"
            )
            console.print(
                f"         answer : {answer[:80]}{'...' if len(answer) > 80 else ''}"
            )

            rows.append(
                {
                    "inputs": tc["input"],
                    "expectations": tc["expected"],
                    "answer": answer,
                    "latency": latency,
                    "l_score": l_score,
                    "category": tc.get("category", "unknown"),
                }
            )

        self.df = pd.DataFrame(rows)
        console.print(f"\n  ✅ {len(self.df)} réponses collectées")
        console.print(f"  Latency moyenne : {self.df['latency'].mean():.2f}s")
        console.print(f"  L score moyen   : {self.df['l_score'].mean():.3f}")
        return self.df

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — MLFLOW EVALUATE
    # ══════════════════════════════════════════════════════════════════════════

    def _run_evaluation(self) -> tuple[dict, float, bool]:
        if self.df.empty:
            raise ValueError(
                "DataFrame vide — appeler collect() avant _run_evaluation()"
            )

        def predict_fn(question: str) -> str:
            row = self.df.loc[self.df["inputs"] == question]
            if not row.empty:
                return row.iloc[0]["answer"]
            return self.agent(question)[0]

        console.print(
            f"\n[bold cyan]── Phase 2 : mlflow.genai.evaluate() ───────────────────[/]"
        )
        console.print(
            f"  Scorers : {[s.name if hasattr(s, 'name') else str(s) for s in ALL_SCORERS]}"
        )

        mlflow.set_experiment("HandbookAgent_LASH")
        run_name = f"LASH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:
            results = mlflow.genai.evaluate(
                data=[
                    {
                        "inputs": {"question": r["inputs"]},
                        "expectations": {"expected_response": r["expectations"]},
                    }
                    for r in self.df[["inputs", "expectations"]].to_dict(
                        orient="records"
                    )
                ],
                predict_fn=predict_fn,
                scorers=ALL_SCORERS,
            )

            # ── DEBUG : affiche toutes les clés disponibles ───────────────
            console.print(
                f"\n[dim]  results.metrics keys : {list(results.metrics.keys())}[/]"
            )

            m = results.metrics
            mean_L = self.df["l_score"].mean()
            mean_A = m.get("correctness/mean")
            mean_S = m.get("safety/mean")
            mean_H = m.get("helpfulness/mean")

            mean_lash = (
                mean_L * WEIGHTS["latency"]
                + mean_A * WEIGHTS["correctness"]
                + mean_S * WEIGHTS["safety"]
                + mean_H * WEIGHTS["helpfulness"]
            )

            lash_pass = all(
                [
                    mean_L >= PASS_THRESHOLDS["latency"],
                    mean_A >= PASS_THRESHOLDS["correctness"],
                    mean_S >= PASS_THRESHOLDS["safety"],
                    mean_H >= PASS_THRESHOLDS["helpfulness"],
                    mean_lash >= PASS_THRESHOLDS["lash"],
                ]
            )

            # ── Log MLflow ────────────────────────────────────────────────
            mlflow.log_metrics(
                {
                    "mean_latency_score": mean_L,
                    "mean_correctness_score": mean_A,
                    "mean_safety_score": mean_S,
                    "mean_helpfulness_score": mean_H,
                    "mean_lash_score": mean_lash,
                    "lash_pass": int(lash_pass),
                }
            )

            # ── Résumé terminal ───────────────────────────────────────────
            console.print(
                f"\n[bold]── Résumé LASH ──────────────────────────────────────────[/]"
            )
            console.print(
                f"\n  {'Dim':<6} {'Weight':<8} {'Score':<8} {'Threshold':<12} {'Pass'}"
            )
            console.print(f"  {'─'*46}")
            dims = [
                ("L", WEIGHTS["latency"], mean_L, PASS_THRESHOLDS["latency"]),
                ("A", WEIGHTS["correctness"], mean_A, PASS_THRESHOLDS["correctness"]),
                ("S", WEIGHTS["safety"], mean_S, PASS_THRESHOLDS["safety"]),
                ("H", WEIGHTS["helpfulness"], mean_H, PASS_THRESHOLDS["helpfulness"]),
            ]
            for dim, weight, score, threshold in dims:
                icon = "✅" if score >= threshold else "❌"
                console.print(
                    f"  {dim:<6} {weight:<8.2f} {score:<8.3f} ≥ {threshold:<10.2f} {icon}"
                )

            console.print(f"  {'─'*46}")
            console.print(
                f"  {'LASH':<6} {'1.00':<8} {mean_lash:<8.3f} ≥ {PASS_THRESHOLDS['lash']:<10.2f} "
                f"{'✅ PASS' if lash_pass else '❌ FAIL'}"
            )
            console.print(f"\n  MLflow run : {run.info.run_id}")

        return results, mean_lash, lash_pass

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC
    # ══════════════════════════════════════════════════════════════════════════

    def eval_lash(self) -> tuple[dict, float, bool]:
        self.collect()
        return self._run_evaluation()
