import time

from mlflow.entities import Feedback
from mlflow.genai.scorers import Correctness, Guidelines, Safety, scorer

# ── WEIGHTS & THRESHOLDS ──────────────────────────────────────────────────────
#
#   L : 0.10  (latency — stable avec gpt-4o-mini)
#   A : 0.40  (correctness — précision sur le handbook, dimension la + critique)
#   S : 0.20  (safety — ne pas inventer de faits)
#   H : 0.30  (helpfulness — réponse vraiment utile ?)

WEIGHTS = {
    "latency": 0.10,
    "correctness": 0.40,
    "safety": 0.20,
    "helpfulness": 0.30,
}

PASS_THRESHOLDS = {
    "latency": 0.20,  # score numérique 0–1
    "correctness": 0.80,  # taux de "yes" ≥ 70%
    "safety": 0.70,  # taux de "yes" ≥ 60%
    "helpfulness": 0.80,  # taux de "yes" ≥ 60%
    "lash": 0.75,  # score global LASH
}

LATENCY_THRESHOLDS = {
    "good": 2.0,  # < 2s  → score 1.0
    "neutral": 5.0,  # < 5s  → score 0.5–1.0 (linéaire)
    # > 5s  → score 0.0–0.5 (dégradé)
}


# ══════════════════════════════════════════════════════════════════════════════
# L — LATENCY SCORER  (@scorer custom)
# ══════════════════════════════════════════════════════════════════════════════


def latency_to_score(latency: float) -> float:
    """Convertit une latency en secondes en score 0–1."""
    GOOD = LATENCY_THRESHOLDS["good"]
    NEUTRAL = LATENCY_THRESHOLDS["neutral"]
    if latency < GOOD:
        return 1.0
    elif latency < NEUTRAL:
        return 1.0 - (latency - GOOD) / (NEUTRAL - GOOD) * 0.5
    else:
        return max(0.0, 0.5 - (latency - NEUTRAL) / 10.0)


@scorer
def latency_ok(inputs: dict) -> Feedback:
    """
    Scorer L — Latency.

    Lit la latency depuis inputs["latency_seconds"] (mesurée en amont)
    et retourne un Feedback avec un score FLOAT 0–1.

    On retourne un float plutôt que yes/no pour deux raisons :
      1. La latency est un spectre — 1.9s n'est pas la même chose que 4.9s
      2. Le score LASH pondéré nécessite une valeur numérique précise
         (pas un ratio binaire comme Correctness ou Safety)

    Feedback(value=float) → MLflow affiche la moyenne dans l'UI.
    """
    latency = inputs.get("latency_seconds", 0.0)
    score = latency_to_score(latency)

    return Feedback(
        value=round(score, 3),
        rationale=f"Latency {latency:.2f}s → score {score:.2f} "
        f"(good<{LATENCY_THRESHOLDS['good']}s, neutral<{LATENCY_THRESHOLDS['neutral']}s)",
    )


# ══════════════════════════════════════════════════════════════════════════════
# A — CORRECTNESS  (built-in MLflow 3.x)
# ══════════════════════════════════════════════════════════════════════════════
#
# Correctness() compare la réponse de l'agent avec expectations["expected_response"].
#
# Le scorer lit automatiquement :
#   inputs["question"]                → la question posée
#   outputs                           → la réponse de l'agent
#   expectations["expected_response"] → ce qu'on attendait
#
# Retourne yes/no par test case.

correctness_scorer = Correctness()


# ══════════════════════════════════════════════════════════════════════════════
# S — SAFETY  (built-in MLflow 3.x)
# ══════════════════════════════════════════════════════════════════════════════
#
# Safety() détecte automatiquement :
#   - Hallucinations / faits inventés
#   - Contenu dangereux, discriminatoire
#   - Fuite d'informations sensibles
#
# Ne nécessite pas d'expectations — évalue la réponse de façon autonome.

safety_scorer = Safety()


# ══════════════════════════════════════════════════════════════════════════════
# H — HELPFULNESS  (Guidelines custom)
# ══════════════════════════════════════════════════════════════════════════════
#
# Guidelines() permet de définir un critère custom en langage naturel.
# C'est le bon choix pour "helpfulness" car le built-in RelevanceToQuery()
# mesure seulement si la réponse adresse le sujet de la question,
# pas si elle est UTILE et ACTIONABLE pour l'employé.

helpfulness_scorer = Guidelines(
    name="helpfulness",
    guidelines="""
    The response must be genuinely helpful to an Agile Lab employee.

    A helpful response:
    - Directly and completely addresses what the user is asking
    - Provides enough detail to be actionable (not just "yes" or "no")
    - Is clear and easy to understand without jargon
    - Guides the user to next steps when appropriate
    - For off-topic questions: politely declines AND explains what it CAN help with

    A response is NOT helpful if:
    - It is technically correct but too vague to act on
    - It answers a different question than the one asked
    - It gives a refusal without any guidance or redirection
    - It repeats the question without answering it
    """,
)


# ── LISTE COMPLÈTE pour mlflow.genai.evaluate() ───────────────────────────────

ALL_SCORERS = [
    latency_ok,
    correctness_scorer,
    safety_scorer,
    helpfulness_scorer,
]
