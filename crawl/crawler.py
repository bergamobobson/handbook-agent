import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import track
from rich.table import Table

console = Console()

# Base configuration
BASE_URL    = "https://handbook.agilelab.it"
OUTPUT_PATH = Path("./data/corpus.json")
DELAY       = 0.4   # delay between requests


def parse_sidebar(soup: BeautifulSoup) -> list[dict]:
    """
    Extract all pages from the HonKit sidebar using data-level and data-path.
    Builds a structured list with section hierarchy.
    """

    items = soup.find_all("li", attrs={"data-level": True})

    pages: list[dict] = []
    seen_urls: set[str] = set()

    # map level → title so we can infer parent sections
    level_to_title: dict[str, str] = {}

    for li in items:
        level     = li["data-level"]                 # e.g. "1.3.8"
        data_path = li.get("data-path", "").strip()  # e.g. "VacationPolicies.html"
        depth     = len(level.split("."))

        # link or span (non-clickable section headers)
        a    = li.find("a", href=True)
        span = li.find("span")

        if a:
            title = a.get_text(strip=True)
        elif span:
            level_to_title[level] = span.get_text(strip=True)
            continue
        else:
            continue

        level_to_title[level] = title

        # skip external links
        href = a["href"].strip()
        if href.startswith("http") and not href.startswith(BASE_URL):
            continue

        # skip items without real HTML page
        if not data_path or not data_path.endswith(".html"):
            continue

        url = urljoin(BASE_URL + "/", data_path)

        if url in seen_urls:
            continue
        seen_urls.add(url)

        # find parent section using level hierarchy
        parts        = level.split(".")
        parent_level = ".".join(parts[:-1]) if len(parts) > 1 else level
        section      = level_to_title.get(parent_level, title)

        pages.append({
            "url":     url,
            "title":   title,
            "level":   level,
            "depth":   depth,
            "section": section,
        })

    return pages


def get_all_page_links() -> list[dict]:
    """Download homepage and extract all page links."""
    console.log(f"[bold cyan]Downloading index:[/] {BASE_URL}")

    resp = requests.get(BASE_URL, timeout=15)
    resp.raise_for_status()

    soup  = BeautifulSoup(resp.text, "lxml")
    pages = parse_sidebar(soup)

    console.log(f"[green]{len(pages)} pages found[/]")
    return pages


def fetch_page_text(url: str) -> dict:
    """
    Download a page and extract clean text from main content container.
    """

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # try several possible containers depending on page layout
    main = (
        soup.find("section", class_="normal")
        or soup.find("div", class_="page-inner")
        or soup.find("article")
        or soup.find("main")
        or soup.body
    )

    if main is None:
        return {"text": "", "h1": ""}

    h1_tag = main.find("h1")
    h1     = h1_tag.get_text(strip=True) if h1_tag else ""

    # remove noise elements
    for tag in main.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = main.get_text(separator="\n", strip=True)

    # normalize excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return {"text": text, "h1": h1}


def main():
    console.print("\n[bold magenta]══ Crawl Agile Lab handbook ══[/]\n")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. discover pages
    pages = get_all_page_links()

    # preview structure
    preview_table = Table(title="Structure preview (first 10 pages)")
    preview_table.add_column("Level",   style="cyan",  no_wrap=True)
    preview_table.add_column("Section", style="yellow")
    preview_table.add_column("Title",   style="white")
    preview_table.add_column("URL",     style="dim",   overflow="fold")

    for p in pages[:10]:
        preview_table.add_row(p["level"], p["section"], p["title"], p["url"])

    console.print(preview_table)
    console.print()

    corpus: list[dict] = []
    errors: list[str]  = []

    # 2. crawl each page
    for page in track(pages, description="[cyan]Crawling...[/]"):
        try:
            content = fetch_page_text(page["url"])

            if not content["text"].strip():
                console.log(f"[yellow]SKIP empty:[/] {page['url']}")
                continue

            corpus.append({
                "url":     page["url"],
                "title":   content["h1"] or page["title"],
                "section": page["section"],
                "level":   page["level"],
                "text":    content["text"],
            })

            time.sleep(DELAY)

        except Exception as exc:
            console.log(f"[red]ERROR:[/] {page['url']} → {exc}")
            errors.append(page["url"])

    # 3. save corpus
    OUTPUT_PATH.write_text(json.dumps(corpus, indent=2, ensure_ascii=False))

    # 4. report
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    total_chars = sum(len(d["text"]) for d in corpus)

    console.print(f"\n[bold green]Saved {len(corpus)} pages → {OUTPUT_PATH}[/]")
    console.print(f"   JSON size : {size_kb:.1f} KB")
    console.print(f"   Total text: {total_chars:,} chars")
    console.print(f"   Avg/page  : {total_chars // max(len(corpus), 1):,} chars")

    if errors:
        console.print(f"\n[yellow]{len(errors)} errors:[/]")
        for url in errors:
            console.print(f"  • {url}")

    # 5. pages per section
    from collections import Counter
    section_counts = Counter(d["section"] for d in corpus)

    section_table  = Table(title="Pages per section")
    section_table.add_column("Section", style="yellow")
    section_table.add_column("Pages",   style="cyan", justify="right")

    for section, count in section_counts.most_common():
        section_table.add_row(section, str(count))

    console.print()
    console.print(section_table)


if __name__ == "__main__":
    main()