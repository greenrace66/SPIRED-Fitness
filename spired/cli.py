from __future__ import annotations

from pathlib import Path

import click

from .weights import default_model_dir


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-i", "--input", "input_file", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", "output_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--mode",
    type=click.Choice(["auto", "fitness", "structure", "stab"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Choose the pipeline to run. Auto selects fitness for 1 FASTA record and stab for 2 records.",
)
@click.option(
    "--model-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override the directory used for downloaded model weights.",
)
def main(input_file: Path, output_dir: Path, mode: str, model_dir: Path | None) -> None:
    """Run SPIRED from a single command."""
    from .pipeline import run_fitness, run_stab, run_structure

    mode = mode.lower()
    resolved_model_dir = model_dir or default_model_dir()
    click.secho("SPIRED", fg="cyan", bold=True)
    click.echo(f"Input : {input_file}")
    click.echo(f"Output: {output_dir.resolve()}")
    click.echo(f"Mode  : {mode}")
    click.echo(f"Model : {resolved_model_dir}")

    if mode == "fitness":
        run_fitness(input_file, output_dir, model_dir=model_dir)
    elif mode == "structure":
        run_structure(input_file, output_dir, model_dir=model_dir)
    elif mode == "stab":
        run_stab(input_file, output_dir, model_dir=model_dir)
    else:
        from .utils import load_records

        record_count = len(load_records(input_file))
        if record_count == 2:
            run_stab(input_file, output_dir, model_dir=model_dir)
        else:
            run_fitness(input_file, output_dir, model_dir=model_dir)

    click.secho("Done.", fg="green", bold=True)


if __name__ == "__main__":
    main()
