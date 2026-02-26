import os
from typing import Annotated, List, Literal, TypedDict

from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import OpenAIEmbeddings
from langdetect import DetectorFactory
from langdetect import detect as ld_detect
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# On fixe le seed pour la reproductibilité de langdetect
DetectorFactory.seed = 0

load_dotenv()

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
    messages: Annotated[List[BaseMessage], add_messages]
    documents: List[Document]
    intent: str
    relevant: bool
    answer: str

# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class IntentResult(BaseModel):
    intent: Literal["conversational", "handbook", "off_topic"] = Field(
        description="The intent of the user message."
    )

CLASSIFY_PROMPT = """Classify the intent of the user message into one of:
- "conversational": greetings, small talk, thanks, or questions referring to previous statements (e.g. "tell me more", "why?", "va en détails").
- "handbook": any specific question about Agile Lab policies, values, benefits.
- "off_topic": technical or general questions unrelated to Agile Lab (e.g. "Naruto", "Python code").

{format_instructions}"""

# ══════════════════════════════════════════════════════════════════════════════
# AGENT
# ══════════════════════════════════════════════════════════════════════════════

class HandbookAgent:
    def __init__(self, model, checkpointer=None, thread_id: str = "1"):
        self.thread_id = thread_id
        self.checkpointer = checkpointer or MemorySaver()
        self.model = model

        _parser = PydanticOutputParser(pydantic_object=IntentResult)
        self.classifier = self.model | _parser
        self._classify_sys = CLASSIFY_PROMPT.format(
            format_instructions=_parser.get_format_instructions()
        )

        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory=CHROMA_DIR,
        )
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20},
        )
        from grader.grade import GRADER_PROMPT, Grader
        self.grader = Grader(model=self.model, prompt=GRADER_PROMPT)
        self.graph = self._build_graph()

    def _classify(self, state: AgentState) -> dict:
        # Utilise les 5 derniers messages pour le contexte sémantique
        result = self.classifier.invoke(
            [SystemMessage(content=self._classify_sys)] + state["messages"][-5:]
        )
        return {"intent": result.intent}

    def _retrieve(self, state: AgentState) -> dict:
        last_human = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        return {"documents": self.retriever.invoke(last_human.content)}

    def _grade(self, state: AgentState) -> dict:
        last_human = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        
        # Tolérance pour les questions de suivi courtes (ex: "va en détails")
        if len(last_human.content.split()) < 4 and len(state["messages"]) > 2:
            return {"relevant": True}

        grader_result = self.grader(question=last_human.content, documents=state["documents"])
        return {"relevant": grader_result["document_relevance"]}

    def _generate(self, state: AgentState) -> dict:
        context = "\n\n".join(
            f"[Source: {doc.metadata.get('title', 'n/a')}]\n{doc.page_content}"
            for doc in state["documents"]
        )
        prompt_messages = [
            SystemMessage(content=f"""You are a helpful assistant for Agile Lab.
            Answer based on the context below AND the conversation history.
            If the user asks for more details about a previous answer, use the history.
            Context: {context}""")
        ] + state["messages"]

        response = self.model.invoke(prompt_messages)
        return {"messages": [response], "answer": response.content}

    def _conversational(self, state: AgentState) -> dict:
        response = self.model.invoke([
            SystemMessage(content="Friendly assistant for Agile Lab employees. Respond in the user's language.")
        ] + state["messages"][-3:])
        return {"messages": [response], "answer": response.content}

    def _reply(self, state: AgentState, template: str) -> dict:
        last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        user_text = last_human.content if last_human else ""
        
        # Langdetect avec fallback
        try:
            lang = ld_detect(user_text)
        except:
            lang = "en"

        try:
            text = GoogleTranslator(source="en", target=lang).translate(template) if lang != "en" else template
        except:
            text = template

        return {"messages": [AIMessage(content=text)], "answer": text}

    def _off_topic(self, state: AgentState) -> dict:
        res = self._reply(state, OFF_TOPIC_REPLY)
        return {**res, "intent": "off_topic"}

    def _not_found(self, state: AgentState) -> dict:
        res = self._reply(state, NOT_FOUND_REPLY)
        return {**res, "intent": "handbook"}

    def _route_intent(self, state: AgentState) -> str:
        intent = state.get("intent", "handbook")
        return intent if intent in ["conversational", "off_topic"] else "retrieve"

    def _route_grade(self, state: AgentState) -> str:
        return "generate" if state.get("relevant") else "not_found"

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
        graph.add_conditional_edges("classify", self._route_intent, 
            {"conversational": "conversational", "retrieve": "retrieve", "off_topic": "off_topic"})
        graph.add_conditional_edges("grade", self._route_grade, 
            {"generate": "generate", "not_found": "not_found"})
        
        for node in ["generate", "conversational", "off_topic", "not_found"]:
            graph.add_edge(node, END)
        return graph.compile(checkpointer=self.checkpointer)

    def __call__(self, question: str, thread_id: str = None) -> tuple[str, str]:
        config = {"configurable": {"thread_id": thread_id or self.thread_id}}
        result = self.graph.invoke({"messages": [HumanMessage(content=question)]}, config=config)
        return result["answer"], result.get("intent", "handbook")