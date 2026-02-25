from langgraph.graph import StateGraph, END, START
from langchain_core.documents import Document      
from pydantic import BaseModel, Field
from typing import TypedDict, List
from langchain_core.prompts import ChatPromptTemplate


GRADER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Tu es un évaluateur de pertinence...
Règles :
- True  → au moins un document contient des infos utiles pour répondre
- False → tous les documents sont hors sujet
- En cas de doute → True (l'information partielle compte)"""
    ),
    (
        "human",                             
        """Question : {question}

Documents :
{documents}

Au moins un document est-il pertinent pour répondre à la question ?"""
    )
])


class GraderResult(BaseModel):
    result: bool = Field(
        description="True if the documents are relevant to the question, False otherwise."
    )


class GraderState(TypedDict):
    question:           str
    documents:          List[Document]
    document_relevance: bool


class Grader:
    def __init__(self, model, prompt=None):
        self.prompt = prompt or GRADER_PROMPT
        self.chain  = self.prompt | model.with_structured_output(GraderResult,  method="function_calling")


        graph = StateGraph(GraderState)
        graph.add_node("grader", self._call_grader)
        graph.add_edge(START, "grader")
        graph.add_edge("grader", END)
        self.graph = graph.compile()

    def _call_grader(self, state: GraderState) -> dict:
        documents_str = "\n\n".join(
            f"{i+1}. {doc.page_content}"
            for i, doc in enumerate(state["documents"])
        )
        response = self.chain.invoke({
            "question":  state["question"],
            "documents": documents_str,
        })
        return {"document_relevance": bool(response.result)}

    def __call__(self, question: str, documents: List[Document]) -> GraderState:  
        return self.graph.invoke(
            {"question": question, "documents": documents}
        )