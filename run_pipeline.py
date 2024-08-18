import click
import yaml
from pipelines.training import training_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


@click.command()
@click.option(
    "--taxi_type", "-t",
    type=click.Choice(["yellow", "green"]),
    default="yellow",
    required=True,
    help="Type of taxi (e.g., yellow, green, etc.)",
)
@click.option(
    "--year", "-y",
    default=2024,
    required=True,
    help="Year of data-store",
)
@click.option(
    "--month", "-m",
    default=5,
    required=True,
    help="Month of data-store",
)
def main(taxi_type: str, year: int, month: int):
    config = load_config("config.yaml")  # Load configuration

    # Assuming training_pipeline takes config as an argument
    (
        training_pipeline.with_options(
            config_path="config.yaml"
        )(taxi_type=taxi_type, year=year, month=month)
    )

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )


if __name__ == "__main__":
    main()
