from pathlib import Path

import torch
import typer
from rich.console import Console

app = typer.Typer(
    name="layerglass",
    help="Generate layered laser-cut, inlay, and engraving files from images.",
)

console = Console()


@app.command()
def doctor() -> None:
    """Check the local LayerGlass environment."""
    console.print("[bold]LayerGlass environment check[/bold]")
    console.print(f"PyTorch: {torch.__version__}")
    console.print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        console.print(f"GPU: {torch.cuda.get_device_name(0)}")
        console.print(f"CUDA build: {torch.version.cuda}")
    else:
        console.print("[red]No CUDA GPU detected by PyTorch.[/red]")


@app.command()
def generate(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output_dir: Path = typer.Option(Path("outputs/latest")),
    layers: int = typer.Option(7, min=2, max=20),
    mode: str = typer.Option("hybrid"),
) -> None:
    """Placeholder for the automatic generation pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"Input: {input_path}")
    console.print(f"Output: {output_dir}")
    console.print(f"Layers: {layers}")
    console.print(f"Mode: {mode}")
    console.print("[yellow]Generation pipeline is not implemented yet.[/yellow]")


if __name__ == "__main__":
    app()