import os
import subprocess

import psycopg


def main() -> None:
    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = os.environ.get("POSTGRES_PORT", "5432")
    database = os.environ.get("POSTGRES_DB", "brainseg")
    user = os.environ.get("POSTGRES_USER", "brainseg")
    password = os.environ.get("POSTGRES_PASSWORD", "brainseg")
    connection_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"

    with psycopg.connect(connection_url) as connection:
        connection.execute("CREATE SCHEMA IF NOT EXISTS mlflow")

    backend_url = (
        f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
        "?options=-csearch_path%3Dmlflow"
    )
    subprocess.run(
        [
            "mlflow",
            "server",
            "--host",
            "0.0.0.0",
            "--port",
            "5000",
            "--backend-store-uri",
            backend_url,
            "--default-artifact-root",
            "/mlruns",
            "--allowed-hosts",
            "localhost,127.0.0.1,localhost:55000,mlflow,mlflow:5000,btc-classification-mlflow,btc-classification-mlflow:5000",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
