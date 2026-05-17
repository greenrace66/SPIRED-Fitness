from pathlib import Path

import click

from spired.pipeline import run_fitness


@click.command()
@click.option("--fasta_file", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--saved_folder", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--model-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
def main(fasta_file: Path, saved_folder: Path, model_dir: Path | None) -> None:
    run_fitness(fasta_file, saved_folder, model_dir=model_dir)


if __name__ == "__main__":
    main()
