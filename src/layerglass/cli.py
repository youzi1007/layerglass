from __future__ import annotations

from pathlib import Path

import torch
import typer
from rich.console import Console

from layerglass.pipeline.assign_layers import (
    MaskAnalysisConfig,
    run_mask_analysis,
)
from layerglass.models.depth import (
    DepthConfig,
    infer_depth,
    save_depth_outputs,
)
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
    model_size: str = typer.Option("small"),
    checkpoint: Path = typer.Option(Path(".")),
    points_per_side: int = typer.Option(24, min=8, max=128),
    points_per_batch: int = typer.Option(16, min=1, max=128),
    min_mask_region_area: int = typer.Option(1200, min=0),
    crop_n_layers: int = typer.Option(0, min=0, max=3),
    use_m2m: bool = typer.Option(False),
    max_side_px: int = typer.Option(1400, min=256),
) -> None:
    """Generate automatic SAM2 masks and save a preview overlay."""
    console.print(f"[bold]Input:[/bold] {input_path}")
    console.print(f"[bold]Output:[/bold] {output_dir}")

    image = load_rgb_image(input_path, max_side_px=max_side_px)

    config = Sam2MaskConfig(
        model_size=model_size,
        checkpoint=checkpoint if str(checkpoint) != "." else None,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        min_mask_region_area=min_mask_region_area,
        crop_n_layers=crop_n_layers,
        use_m2m=use_m2m,
        max_side_px=max_side_px,
    )

    console.print("[yellow]Generating SAM2 masks...[/yellow]")
    masks = generate_masks(image, config)

    console.print(f"[green]Generated {len(masks)} masks.[/green]")
    save_mask_preview(image, masks, output_dir)

    console.print(f"[green]Saved:[/green] {output_dir / 'mask_overlay.png'}")
    console.print(f"[green]Saved:[/green] {output_dir / 'masks_summary.json'}")


@app.command("depth-preview")
def depth_preview(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        Path("outputs/depth_preview")
    ),
    encoder: str = typer.Option("vits"),
    input_size: int = typer.Option(
        518,
        min=256,
        max=1024,
    ),
) -> None:
    """Generate raw and preview depth maps."""
    console.print(f"[bold]Input:[/bold] {input_path}")
    console.print(f"[bold]Output:[/bold] {output_dir}")

    checkpoint = Path(
        f"~/projects/Depth-Anything-V2/checkpoints/"
        f"depth_anything_v2_{encoder}.pth"
    ).expanduser()

    config = DepthConfig(
        encoder=encoder,
        input_size=input_size,
        checkpoint=checkpoint,
    )

    console.print("[yellow]Generating depth map...[/yellow]")
    depth = infer_depth(input_path, config)
    save_depth_outputs(depth, output_dir)

    console.print(
        f"[green]Depth shape:[/green] {depth.shape}"
    )
    console.print(
        f"[green]Saved:[/green] "
        f"{output_dir / 'depth.npy'}"
    )
    console.print(
        f"[green]Saved:[/green] "
        f"{output_dir / 'depth_preview.png'}"
    )


@app.command("analyze-masks")
def analyze_masks(
    masks_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
    ),
    depth_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
    ),
    image_path: Path = typer.Option(
        Path("outputs/segment_preview_richer/input.png"),
        exists=True,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        Path("outputs/mask_analysis")
    ),
    min_area_px: int = typer.Option(100, min=1),
    duplicate_iou: float = typer.Option(
        0.90,
        min=0.0,
        max=1.0,
    ),
) -> None:
    """Align depth, score SAM masks, and remove duplicates."""
    config = MaskAnalysisConfig(
        min_area_px=min_area_px,
        duplicate_iou=duplicate_iou,
    )

    console.print("[yellow]Analyzing masks...[/yellow]")

    result = run_mask_analysis(
        masks_path=masks_path,
        depth_path=depth_path,
        image_path=image_path,
        output_dir=output_dir,
        config=config,
    )

    console.print(
        f"[green]Input masks:[/green] "
        f"{result['input_masks']}"
    )
    console.print(
        f"[green]Kept masks:[/green] "
        f"{result['kept_masks']}"
    )
    console.print(
        f"[green]Duplicates removed:[/green] "
        f"{result['duplicates_removed']}"
    )
    console.print(
        f"[green]Saved:[/green] "
        f"{output_dir / 'kept_overlay.png'}"
    )


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
