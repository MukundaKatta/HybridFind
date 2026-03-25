"""CLI entry-point: hybridfind index docs/ | hybridfind search 'query'."""

from __future__ import annotations

import json
import pathlib
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from hybridfind.config import SearchConfig
from hybridfind.core import HybridSearch

app = typer.Typer(help="HybridFind — hybrid semantic + keyword search CLI")
console = Console()

INDEX_PATH = pathlib.Path(".hybridfind_index.json")


def _load_engine() -> HybridSearch:
    """Load a previously indexed dataset."""
    if not INDEX_PATH.exists():
        console.print("[red]No index found. Run `hybridfind index <dir>` first.[/red]")
        raise typer.Exit(1)
    data = json.loads(INDEX_PATH.read_text())
    engine = HybridSearch(SearchConfig(**data.get("config", {})))
    engine.add_documents(
        texts=data["texts"],
        ids=data["ids"],
        metadatas=data.get("metadatas", []),
    )
    return engine


@app.command()
def index(
    directory: str = typer.Argument(..., help="Directory containing text files to index"),
    extensions: str = typer.Option(".txt,.md,.rst", help="Comma-separated file extensions"),
) -> None:
    """Index all text files in a directory."""
    dir_path = pathlib.Path(directory)
    if not dir_path.is_dir():
        console.print(f"[red]Directory not found: {directory}[/red]")
        raise typer.Exit(1)

    exts = {e.strip() for e in extensions.split(",")}
    texts: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []

    for fpath in sorted(dir_path.rglob("*")):
        if fpath.is_file() and fpath.suffix in exts:
            content = fpath.read_text(errors="ignore")
            if content.strip():
                texts.append(content)
                ids.append(str(fpath))
                metadatas.append({"filename": fpath.name, "path": str(fpath)})

    if not texts:
        console.print("[yellow]No matching files found.[/yellow]")
        raise typer.Exit(1)

    # Persist raw data so we can reload later
    INDEX_PATH.write_text(
        json.dumps({"texts": texts, "ids": ids, "metadatas": metadatas, "config": {}})
    )
    console.print(f"[green]Indexed {len(texts)} documents from {directory}[/green]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    bm25_weight: float = typer.Option(0.5, "--bm25-weight", help="BM25 weight"),
    vector_weight: float = typer.Option(0.5, "--vector-weight", help="Vector weight"),
    filter_field: Optional[str] = typer.Option(None, "--filter-field", help="Metadata field to filter"),
    filter_value: Optional[str] = typer.Option(None, "--filter-value", help="Metadata value to match"),
) -> None:
    """Search the indexed documents."""
    engine = _load_engine()
    engine.config.bm25_weight = bm25_weight
    engine.config.vector_weight = vector_weight
    engine.config = engine.config  # trigger re-validation

    meta_filter = None
    if filter_field and filter_value:
        meta_filter = {filter_field: filter_value}

    results = engine.search(query, top_k=top_k, metadata_filter=meta_filter)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f"Search results for: {query}")
    table.add_column("Rank", style="bold cyan", width=5)
    table.add_column("Doc ID", style="green")
    table.add_column("Score", style="magenta", width=10)
    table.add_column("Preview", max_width=60)

    for rank, result in enumerate(results, 1):
        preview = result.text[:120].replace("\n", " ")
        table.add_row(str(rank), result.doc_id, f"{result.score:.6f}", preview)

    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
