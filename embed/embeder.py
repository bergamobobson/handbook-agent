import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.schema import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Table

load_dotenv()
console = Console()

CORPUS_PATH = Path("./data/corpus.json")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR")
COLLECTION_NAME = "agilelab_handbook"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
BATCH_SIZE = 100


def load_corpus(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"❌ Corpus introuvable : {path}\n"
            "   Lance d'abord : uv run python step1_crawl.py"
        )
    data = json.loads(path.read_text())
    console.log(f"[green]{len(data)} pages chargées depuis {path}[/]")
    return data


def build_documents(corpus: list[dict]) -> list[Document]:
    """Transform raw corpus entries into LangChain Documents."""
    docs = []
    for entry in corpus:
        docs.append(
            Document(
                page_content=entry["text"],
                metadata={
                    "url": entry["url"],
                    "title": entry["title"],
                    "section": entry["section"],
                    "level": entry["level"],
                    "source": entry["url"],
                },
            )
        )
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    console.log(
        f"[green]{len(docs)} pages → {len(chunks)} chunks[/] "
        f"[dim](moy. {len(chunks) // len(docs)} chunks/page)[/]"
    )
    return chunks


def ingest_to_chroma(chunks: list[Document]) -> Chroma:
    """Generate embeddings and store them in Chroma."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Clean existing data
    existing = vectorstore.get()
    if existing["ids"]:
        console.log(
            f"[yellow]Suppression de {len(existing['ids'])} vecteurs existants...[/]"
        )
        vectorstore.delete(ids=existing["ids"])

    # Ingest in batches
    with Progress(
        TextColumn("[cyan]Embedding & stockage...[/]"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("ingest", total=len(chunks))
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            texts = [c.page_content for c in batch]
            metadatas = [c.metadata for c in batch]
            vectorstore.add_texts(texts=texts, metadatas=metadatas)
            progress.advance(task, advance=len(batch))

    console.print(
        f"[bold green]✅ {len(chunks)} chunks stockés dans Chroma → {CHROMA_DIR}[/]"
    )
    return vectorstore


def test_retrieval(vectorstore: Chroma):
    """Sanity check: query a few questions and display results."""
    test_queries = [
        "How many vacation days do employees get?",
        "What is the referral fee policy?",
        "How does the engineering ladder work?",
    ]

    console.print("\n[bold cyan]═══ Test de retrieval ═══[/]\n")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    for query in test_queries:
        results = retriever.invoke(query)
        table = Table(title=f'❓ "{query}"')
        table.add_column("Page", style="yellow", no_wrap=True)
        table.add_column("Section", style="cyan")
        table.add_column("Aperçu", style="white", overflow="fold")

        for doc in results:
            table.add_row(
                doc.metadata.get("title", "n/a"),
                doc.metadata.get("section", "n/a"),
                doc.page_content[:120].replace("\n", " ") + "...",
            )
        console.print(table)
        console.print()


def main():
    console.print("\n[bold magenta]══ ÉTAPE 2 : Chunking + Embeddings + Chroma ══[/]\n")
    corpus = load_corpus(CORPUS_PATH)
    docs = build_documents(corpus)
    chunks = chunk_documents(docs)
    vectorstore = ingest_to_chroma(chunks)
    test_retrieval(vectorstore)


if __name__ == "__main__":
    main()
