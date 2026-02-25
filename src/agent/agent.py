import os
from typing import Annotated, List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from grader.grade import GRADER_PROMPT
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from rich.console import Console

# Import du Grader existant
from grader.grade import Grader

load_dotenv()
console = Console()

# ── CONFIG ────────────────────────────────────────────────────────────────────

CHROMA_DIR      = os.getenv("CHROMA_PERSIST_DIR")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

OFF_TOPIC_REPLY = (
    "I can only answer questions about the Agile Lab company handbook. "
    "Please ask me about Agile Lab's policies, culture, benefits, or engineering practices."
)

NOT_FOUND_REPLY = (
    "I couldn't find relevant information about this in the Agile Lab handbook. "
    "You might want to ask your manager or check internally."
)


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """
    État global qui circule entre tous les nœuds du graph.

    Chaque nœud retourne un dict PARTIEL — LangGraph merge automatiquement
    les champs retournés dans le state existant.
    Les champs non retournés restent inchangés.
    """
    # add_messages est un reducer : au lieu de remplacer la liste,
    # il accumule les nouveaux messages dans l'historique existant
    messages  : Annotated[List[BaseMessage], add_messages]
    documents : List[Document]
    intent    : str   # "conversational" | "handbook" | "off_topic"
    relevant  : bool
    answer    : str
    source    : str


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class IntentResult(BaseModel):
    """
    Résultat du classifier d'intention.

    intent peut être :
        "conversational" → salutations, small talk, remerciements
        "handbook"       → question sur Agile Lab, ses politiques, culture, etc.
        "off_topic"      → question technique ou hors sujet sans lien avec Agile Lab
    """
    intent: Literal["conversational", "handbook", "off_topic"] = Field(
        description="The intent of the user message."
    )


# ══════════════════════════════════════════════════════════════════════════════
# AGENT
# ══════════════════════════════════════════════════════════════════════════════

class HandbookAgent:
    """
    Agent RAG orchestré par LangGraph.

    Utilise le Grader existant (grader/grade.py) comme composant externe.
    Répond uniquement aux questions sur le handbook Agile Lab.
    """

    def __init__(self, model, checkpointer=None, thread_id: str = "1"):
        self.thread_id    = thread_id
        self.checkpointer = checkpointer or MemorySaver()

        # ── LLM ──────────────────────────────────────────────────────────
        self.model = model

        # ── Classifier — détecte l'intention avant de router ─────────────
        # with_structured_output force le LLM à retourner un IntentResult
        self.classifier = self.model.with_structured_output(IntentResult)

        # ── Retriever — lecture seule sur Chroma existant ─────────────────
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory=CHROMA_DIR,
        )
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20},
        )

        # ── Grader — ton instance existante ──────────────────────────────
        # On passe le même LLM et le prompt défini ci-dessus
        self.grader = Grader(
            model=self.model,
            prompt=GRADER_PROMPT,
        )

        # ── Graph ─────────────────────────────────────────────────────────
        self.graph = self._build_graph()


    # ── NŒUD 1 : classify ────────────────────────────────────────────────────

    def _classify(self, state: AgentState) -> dict:
        """
        Classifie l'intention du message avant de router.

        Trois cas :
            conversational → "Hi", "Thanks", "How are you?"
            handbook       → questions sur Agile Lab
            off_topic      → questions techniques sans lien avec Agile Lab
        """
        #console.log("[cyan]→ Classify[/]")

        last_human = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))

        result = self.classifier.invoke([
            SystemMessage(content="""Classify the intent of the user message into one of:
- "conversational": greetings, small talk, thanks, casual chat, questions about the conversation itself or about things the user said earlier (e.g. "Hi", "Thanks", "How are you?", "what is my name?", "what did I just say?", "remind me what I told you")
- "handbook": any question about Agile Lab — its policies, values, culture, benefits, processes, people, engineering practices
- "off_topic": technical questions unrelated to Agile Lab (e.g. "How to code in Python?", "What is Docker?")

Important: if the question refers to something the user mentioned in the conversation (their name, a previous statement, etc.), classify it as "conversational"."""),
            HumanMessage(content=last_human.content),
        ])

        #console.log(f"  [dim]intent={result.intent}[/]")
        return {"intent": result.intent}


    # ── NŒUD 2 : retrieve ─────────────────────────────────────────────────────

    def _retrieve(self, state: AgentState) -> dict:
        """Cherche les chunks pertinents dans Chroma."""
        #console.log("[cyan]→ Retrieve[/]")

        # Extraire la dernière question depuis l'historique
        last_human = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))

        documents = self.retriever.invoke(last_human.content)
        #console.log(f"  [dim]{len(documents)} chunks récupérés[/]")
        return {"documents": documents}


    # ── NŒUD 3 : grade ────────────────────────────────────────────────────────

    def _grade(self, state: AgentState) -> dict:
        """
        Appelle ton Grader existant via __call__.

        self.grader(question, documents) → retourne le GraderState complet
        On extrait seulement "document_relevance" pour notre AgentState.
        """
        #console.log("[cyan]→ Grade[/]")

        # Extraire la dernière question depuis l'historique
        last_human = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))

        grader_result = self.grader(
            question=last_human.content,
            documents=state["documents"],
        )

        relevant = grader_result["document_relevance"]
        #console.log(f"  {'✅' if relevant else '❌'} Pertinent : [bold]{relevant}[/]")

        return {"relevant": relevant}


    # ── NŒUD 4 : generate ─────────────────────────────────────────────────────

    def _generate(self, state: AgentState) -> dict:
        """
        Génère la réponse finale à partir des chunks du handbook.
        """
        #console.log("[cyan]→ Generate[/]")

        context = "\n\n".join(
            f"[Source: {doc.metadata.get('title', 'n/a')}]\n{doc.page_content}"
            for doc in state["documents"]
        )

        # Injecter le contexte RAG + tout l'historique dans le prompt
        # Le LLM voit toute la conversation → peut répondre aux questions de suivi
        prompt_messages = [
            SystemMessage(content=f"""You are a helpful assistant for Agile Lab employees.
Answer questions based on the Agile Lab company handbook context provided below.
Be conversational and elaborate your answers with relevant details from the handbook.
If the context doesn't contain enough information, say so clearly.
Always respond in the same language as the user's question.

Context:
{context}""")
        ]
        # Ajouter tout l'historique — add_messages accumulera le AIMessage retourné
        prompt_messages += state["messages"]

        response = self.model.invoke(prompt_messages)

        return {"messages": [response], "answer": response.content}


    # ── NŒUD 5 : conversational ───────────────────────────────────────────────

    def _conversational(self, state: AgentState) -> dict:
        """
        Répond aux salutations et au small talk de façon naturelle,
        sans chercher dans le handbook.
        """
        #console.log("[cyan]→ Conversational[/]")

        prompt_messages = [
            SystemMessage(content="""You are a friendly assistant for Agile Lab employees.
Respond naturally to greetings and casual messages.
Keep it short and warm.
Always respond in the same language as the user's message.""")
        ]
        prompt_messages += state["messages"]

        response = self.model.invoke(prompt_messages)
        return {"messages": [response], "answer": response.content}


    # ── NŒUD 6 : off_topic ────────────────────────────────────────────────────

    def _off_topic(self, state: AgentState) -> dict:
        """
        Retourne un message de refus poli quand la question
        n'est pas liée au handbook Agile Lab.
        """
        #console.log("[red]→ Off-topic[/]")

        response = AIMessage(content=OFF_TOPIC_REPLY)
        return {"messages": [response], "answer": OFF_TOPIC_REPLY}


    # ── NŒUD 7 : not_found ───────────────────────────────────────────────────

    def _not_found(self, state: AgentState) -> dict:
        """
        La question concerne Agile Lab mais la réponse n'est pas dans le handbook.
        Différent de off_topic : on sait que c'est une vraie question AL,
        juste que le handbook ne la couvre pas.
        """
        #console.log("[orange]→ Not found[/]")

        response = AIMessage(content=NOT_FOUND_REPLY)
        return {"messages": [response], "answer": NOT_FOUND_REPLY}


    # ── ROUTER APRÈS CLASSIFY ─────────────────────────────────────────────────

    def _route_intent(self, state: AgentState) -> str:
        """
        Route selon l'intention détectée par le classifier :
            "conversational" → réponse directe sans RAG
            "handbook"       → retrieve → grade → generate
            "off_topic"      → refus poli
        """
        intent = state.get("intent", "handbook")
        if intent == "conversational":
            return "conversational"
        if intent == "off_topic":
            return "off_topic"
        return "retrieve"


    # ── ROUTER APRÈS GRADE ────────────────────────────────────────────────────

    def _route_grade(self, state: AgentState) -> str:
        """
        Décide le nœud suivant selon le résultat du grader :
            "generate"   → relevant=True  (handbook contient la réponse)
            "not_found"  → relevant=False (question AL mais pas dans le handbook)
        """
        if state["relevant"]:
            #console.log("[green]  Route → generate[/]")
            return "generate"
        #console.log("[orange]  Route → not_found[/]")
        return "not_found"


    # ── BUILD GRAPH ───────────────────────────────────────────────────────────

    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("classify",      self._classify)
        graph.add_node("retrieve",      self._retrieve)
        graph.add_node("grade",         self._grade)
        graph.add_node("generate",      self._generate)
        graph.add_node("conversational",self._conversational)
        graph.add_node("off_topic",     self._off_topic)
        graph.add_node("not_found",     self._not_found)

        graph.add_edge(START,            "classify")
        graph.add_edge("retrieve",       "grade")
        graph.add_edge("generate",       END)
        graph.add_edge("conversational", END)
        graph.add_edge("off_topic",      END)
        graph.add_edge("not_found",      END)

        # Après classify → 3 routes possibles
        graph.add_conditional_edges(
            "classify",
            self._route_intent,
            {
                "conversational": "conversational",
                "retrieve":       "retrieve",
                "off_topic":      "off_topic",
            },
        )

        # Après grade → 2 routes possibles
        graph.add_conditional_edges(
            "grade",
            self._route_grade,
            {
                "generate":  "generate",
                "not_found": "not_found",
            },
        )

        return graph.compile(checkpointer=self.checkpointer)


    # ── INVOKE ────────────────────────────────────────────────────────────────

    def __call__(self, question: str, thread_id: str = None) -> tuple[str, str]:
        """
        Lance l'agent sur une question.
        Retourne : (answer, source)
        """
        config = {"configurable": {"thread_id": thread_id or self.thread_id}}

        result = self.graph.invoke(
            {
                # add_messages accumule → on passe juste le nouveau message
                "messages": [HumanMessage(content=question)],
                "source":   "handbook",
            },
            config=config,
        )

        return result["answer"], result.get("source", "handbook")