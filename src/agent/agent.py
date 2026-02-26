import os
from typing import Annotated, List, Literal, TypedDict

from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from langdetect import DetectorFactory
from langdetect import detect as ld_detect

DetectorFactory.seed = 0  # rend langdetect déterministe
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from rich.console import Console

from grader.grade import GRADER_PROMPT, Grader

load_dotenv()
console = Console()


# ── CONFIG ────────────────────────────────────────────────────────────────────

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR")
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
    Chaque nœud retourne un dict PARTIEL — LangGraph merge automatiquement.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    documents: List[Document]
    intent: str  # "conversational" | "handbook" | "off_topic"
    relevant: bool
    answer: str
    source: str


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER SCHEMA
# ══════════════════════════════════════════════════════════════════════════════


class IntentResult(BaseModel):
    """
    intent :
        "conversational" → salutations, small talk, remerciements
        "handbook"       → question sur Agile Lab
        "off_topic"      → question hors sujet
    """

    intent: Literal["conversational", "handbook", "off_topic"] = Field(
        description="The intent of the user message."
    )


CLASSIFY_PROMPT = """Classify the intent of the user message into one of:
- "conversational": greetings, small talk, thanks, casual chat, questions about \
the conversation itself or about things the user said earlier \
(e.g. "Hi", "Thanks", "How are you?", "what is my name?", "what did I just say?")
- "handbook": any question about Agile Lab — its policies, values, culture, \
benefits, processes, people, engineering practices
- "off_topic": technical questions unrelated to Agile Lab \
(e.g. "How to code in Python?", "What is Docker?")

Important: if the question refers to something the user mentioned in the \
conversation (their name, a previous statement, etc.), classify as "conversational".

{format_instructions}"""


# ══════════════════════════════════════════════════════════════════════════════
# AGENT
# ══════════════════════════════════════════════════════════════════════════════


class HandbookAgent:

    def __init__(self, model, checkpointer=None, thread_id: str = "1"):
        self.thread_id = thread_id
        self.checkpointer = checkpointer or MemorySaver()
        self.model = model

        # ── Classifier ────────────────────────────────────────────────────
        # PydanticOutputParser évite le warning Pydantic causé par le wrapper
        # interne de with_structured_output ("Expected none" sur field "parsed")
        _parser = PydanticOutputParser(pydantic_object=IntentResult)
        self.classifier = self.model | _parser
        self._classify_sys = CLASSIFY_PROMPT.format(
            format_instructions=_parser.get_format_instructions()
        )

        # ── Retriever ─────────────────────────────────────────────────────
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory=CHROMA_DIR,
        )
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20},
        )

        # ── Grader ────────────────────────────────────────────────────────
        self.grader = Grader(model=self.model, prompt=GRADER_PROMPT)

        # ── Graph ─────────────────────────────────────────────────────────
        self.graph = self._build_graph()

    # ══════════════════════════════════════════════════════════════════════════
    # NŒUDS
    # ══════════════════════════════════════════════════════════════════════════

    def _classify(self, state: AgentState) -> dict:
        """Détecte l'intention : conversational | handbook | off_topic."""
        last_human = next(
            m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)
        )
        result = self.classifier.invoke(
            [
                SystemMessage(content=self._classify_sys),
                HumanMessage(content=last_human.content),
            ]
        )
        return {"intent": result.intent}

    def _retrieve(self, state: AgentState) -> dict:
        """Cherche les chunks pertinents dans Chroma."""
        last_human = next(
            m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)
        )
        return {"documents": self.retriever.invoke(last_human.content)}

    def _grade(self, state: AgentState) -> dict:
        """Évalue la pertinence des chunks récupérés."""
        last_human = next(
            m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)
        )
        grader_result = self.grader(
            question=last_human.content, documents=state["documents"]
        )
        return {"relevant": grader_result["document_relevance"]}

    def _generate(self, state: AgentState) -> dict:
        """Génère la réponse finale depuis les chunks du handbook."""
        context = "\n\n".join(
            f"[Source: {doc.metadata.get('title', 'n/a')}]\n{doc.page_content}"
            for doc in state["documents"]
        )
        prompt_messages = (
            [
                SystemMessage(
                    content=f"""You are a helpful assistant for Agile Lab employees.
Answer questions based on the Agile Lab company handbook context provided below.
Be conversational and elaborate your answers with relevant details from the handbook.
If the context doesn't contain enough information, say so clearly.
Always respond in the same language as the user's question.

Context:
{context}"""
                )
            ]
            + state["messages"]
        )

        response = self.model.invoke(prompt_messages)
        return {"messages": [response], "answer": response.content}

    def _conversational(self, state: AgentState) -> dict:
        """Répond aux salutations et au small talk sans chercher dans le handbook."""
        prompt_messages = (
            [
                SystemMessage(
                    content="""You are a friendly assistant for Agile Lab employees.
Respond naturally to greetings and casual messages.
Keep it short and warm.
Always respond in the same language as the user's message."""
                )
            ]
            + state["messages"]
        )

        response = self.model.invoke(prompt_messages)
        return {"messages": [response], "answer": response.content}

    # ── HELPER : détection langue + traduction ────────────────────────────────

    def _reply(self, state: AgentState, template: str) -> dict:
        """
        Détecte la langue du dernier message via FastText,
        traduit `template` si besoin, et retourne le dict de state standard.
        Fallback sur l'anglais en cas d'erreur de détection ou de traduction.
        """
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        user_text = last_human.content if last_human else ""

        try:
            lang = ld_detect(user_text)
        except Exception:
            lang = "en"

        try:
            text = (
                GoogleTranslator(source="en", target=lang).translate(template)
                if lang != "en"
                else template
            )
        except Exception:
            text = template

        return {"messages": [AIMessage(content=text)], "answer": text}

    def _off_topic(self, state: AgentState) -> dict:
        """Question hors scope — pas liée à Agile Lab."""
        return self._reply(state, OFF_TOPIC_REPLY)

    def _not_found(self, state: AgentState) -> dict:
        """Question sur Agile Lab, mais absente du handbook."""
        return self._reply(state, NOT_FOUND_REPLY)

    # ══════════════════════════════════════════════════════════════════════════
    # ROUTERS
    # ══════════════════════════════════════════════════════════════════════════

    def _route_intent(self, state: AgentState) -> str:
        intent = state.get("intent", "handbook")
        if intent == "conversational":
            return "conversational"
        if intent == "off_topic":
            return "off_topic"
        return "retrieve"

    def _route_grade(self, state: AgentState) -> str:
        return "generate" if state["relevant"] else "not_found"

    # ══════════════════════════════════════════════════════════════════════════
    # GRAPH
    # ══════════════════════════════════════════════════════════════════════════

    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("classify", self._classify)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade", self._grade)
        graph.add_node("generate", self._generate)
        graph.add_node("conversational", self._conversational)
        graph.add_node("off_topic", self._off_topic)
        graph.add_node("not_found", self._not_found)

        graph.add_edge(START, "classify")
        graph.add_edge("retrieve", "grade")
        graph.add_edge("generate", END)
        graph.add_edge("conversational", END)
        graph.add_edge("off_topic", END)
        graph.add_edge("not_found", END)

        graph.add_conditional_edges(
            "classify",
            self._route_intent,
            {
                "conversational": "conversational",
                "retrieve": "retrieve",
                "off_topic": "off_topic",
            },
        )
        graph.add_conditional_edges(
            "grade",
            self._route_grade,
            {"generate": "generate", "not_found": "not_found"},
        )

        return graph.compile(checkpointer=self.checkpointer)

    # ══════════════════════════════════════════════════════════════════════════
    # INVOKE
    # ══════════════════════════════════════════════════════════════════════════

    def __call__(self, question: str, thread_id: str = None) -> tuple[str, str]:
        config = {"configurable": {"thread_id": thread_id or self.thread_id}}
        
        # We invoke the graph. The final result contains the full AgentState.
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=question)]}, 
            config=config,
        )
        
        # Extract the answer and use the 'intent' field as the source
        answer = result.get("answer", "No answer generated.")
        source_finale = result.get("intent", "off_topic")
        
        return answer, source_finale
