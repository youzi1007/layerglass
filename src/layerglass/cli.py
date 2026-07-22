from __future__ import annotations

from pathlib import Path

import torch
import typer
from rich.console import Console

from layerglass.models.segmentation import (
    Sam2MaskConfig,
    generate_masks,
    load_rgb_image,
)
from layerglass.render.preview import save_mask_preview

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


@app.command("segment-preview")
def segment_preview(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output_dir: Path = typer.Option(Path("outputs/segment_preview")),
    checkpoint: Path = typer.Option(
        Path("~/projects/sam2/checkpoints/sam2.1_hiera_large.pt").expanduser()
    ),
    points_per_side: int = typer.Option(32, min=8, max=128),
    min_mask_region_area: int = typer.Option(400, min=0),
) -> None:
    """Generate automatic SAM2 masks and save a preview overlay."""
    console.print(f"[bold]Input:[/bold] {input_path}")
    console.print(f"[bold]Output:[/bold] {output_dir}")

    image = load_rgb_image(input_path)

    config = Sam2MaskConfig(
        checkpoint=checkpoint,
        points_per_side=points_per_side,
        min_mask_region_area=min_mask_region_area,
    )

    console.print("[yellow]Generating SAM2 masks...[/yellow]")
    masks = generate_masks(image, config)

    console.print(f"[green]Generated {len(masks)} masks.[/green]")
    save_mask_preview(image, masks, output_dir)

    console.print(f"[green]Saved:[/green] {output_dir / 'mask_overlay.png'}")
    console.print(f"[green]Saved:[/green] {output_dir / 'masks_summary.json'}")


@app.command()
def generate(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output_dir: Path = typer.Option(Path("outputs/latest")),
    layers: int = typer.Option(7, min=2, max=20),
    mode: str = typer.Option("hybrid"),
) -> None:
    """Placeholder for the full automatic fabrication pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"Input: {input_path}")
    console.print(f"Output: {output_dir}")
    console.print(f"Layers: {layers}")
    console.print(f"Mode: {mode}")
    console.print("[yellow]Full generation pipeline is not implemented yet.[/yellow]")


if __name__ == "__main__":
    app()
