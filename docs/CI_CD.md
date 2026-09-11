# CI/CD Pipeline

## Continuous Integration

`.github/workflows/ci.yml` runs on pushes and pull requests targeting `main`, plus manual dispatch.

It checks:

- Ruff linting
- Ruff formatting
- Mypy type checking
- Python compilation
- Full pytest suite with coverage
- Docker Compose configuration
- Alembic offline migration generation
- Gitleaks secret scanning
- pip-audit dependency auditing
- API, worker, and MLflow Docker builds

CI does not require Kaggle credentials, raw dataset files, checkpoints, or production MLflow data. Tests use synthetic fixtures and mocked external services where appropriate.

Mypy currently runs as a non-blocking report because the project contains dynamic Torch/PIL model and image types that require a dedicated typing cleanup. Ruff, compilation, tests, migrations, Compose, secret scanning, dependency auditing, and Docker builds remain blocking checks.

## Continuous Delivery

`.github/workflows/cd.yml` runs on semantic-version tags such as `v1.0.0` and can be manually dispatched.

The release job:

- Builds the API image
- Builds the MLflow image
- Publishes both images to GHCR
- Tags images with the release tag and `latest`
- Uploads release documentation

Deployment is gated behind a protected GitHub `production` environment. To enable it, configure:

- Secret: `DEPLOY_COMMAND`
- Optional variable: `DEPLOY_ENVIRONMENT`

Manual dispatch must set `deploy: true`. Cloud credentials must remain in protected environment secrets or workload identity configuration, never in workflow YAML.

## Local Equivalents

```powershell
python -m pip install -e ".[dev]"
python -m ruff check app scripts worker tests
python -m ruff format --check app scripts worker tests
python -m mypy app worker scripts
python -m compileall -q app migrations scripts worker tests
python -m pytest --cov=app --cov=worker --cov-report=term-missing
docker compose config --quiet
python -m alembic upgrade head --sql
python scripts/release_audit.py
```
