"""
╔══════════════════════════════════════════════════════════════╗
║  API — src/api/app.py                                        ║
║                                                              ║
║  Endpoints :                                                 ║
║    GET  /       ← interface HTML                            ║
║    POST /ask    ← question → réponse                        ║
║    GET  /health ← vérifier que l'API tourne                 ║
║                                                              ║
║  Lancer avec :                                               ║
║    uv run uvicorn src.api.app:app --reload                   ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.agent.agent import HandbookAgent

load_dotenv()

# app.py est dans src/api/ → remonter 3 niveaux pour la racine du projet
ROOT_DIR = Path(__file__).parent.parent.parent
UI_DIR = ROOT_DIR / "src" / "ui"
STATIC_DIR = ROOT_DIR / "static"


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN — initialise l'agent une seule fois au démarrage
# ══════════════════════════════════════════════════════════════════════════════

resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Demarrage — initialisation de l'agent...")
    model_name = os.getenv("MODEL_NAME")
    llm = ChatOpenAI(model=model_name, temperature=0)
    resources["agent"] = HandbookAgent(model=llm)
    print("Agent pret")
    yield
    resources.clear()
    print("Arret — ressources liberees")


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════


class AskRequest(BaseModel):
    question: str = Field(description="La question a poser sur le handbook")
    thread_id: str = Field(
        default="default",
        description="ID de session — meme thread_id = meme conversation",
    )


class AskResponse(BaseModel):
    answer: str = Field(description="Reponse generee par l'agent")
    source: str = Field(description="'handbook' ou 'web'")
    question: str = Field(description="Question originale")


class HealthResponse(BaseModel):
    status: str
    agent_ready: bool


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Agile Lab Handbook API",
    description="RAG chatbot base sur le handbook Agile Lab",
    version="2.0.0",
    lifespan=lifespan,
)

# Sert les fichiers statiques (logo, images...)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# GET / → UI


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(UI_DIR / "index.html")


# GET /health


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        agent_ready="agent" in resources,
    )


# POST /ask


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Meme thread_id entre les requetes = l'agent se souvient des echanges
    precedents grace a MemorySaver.

    Exemple :
        msg 1 : "What are Agile Lab values?"
        msg 2 : "Tell me more about the first one"  <- sait de quoi on parle
    """
    if "agent" not in resources:
        raise HTTPException(status_code=503, detail="Agent non initialise")

    try:
        answer, source = resources["agent"](
            request.question,
            thread_id=request.thread_id,
        )
        return AskResponse(
            answer=answer,
            source=source,
            question=request.question,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
