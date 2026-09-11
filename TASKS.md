# Brain Tumor MRI Classification Pipeline Tasks

**Task ID prefix:** `BTC-XXX`  
**Source:** `PRD.md` (Draft v2)  
**Status vocabulary:** `TODO`, `IN PROGRESS`, `BLOCKED`, `DONE`  
**Scope:** Automated 2D MRI classification from Kaggle acquisition through production inference.

## Delivery Rules

- Implement phases in order unless a task explicitly lists an independent dependency.
- Do not promote or serve a model unless the evaluation gate has recorded its decision and lineage.
- Every task that creates a dataset, configuration, run, artifact, model, or prediction must record the relevant parent IDs and hashes.
- Keep Kaggle credentials and all other secrets outside source control.
- Use task IDs in commit messages, for example `BTC-021 implement dataset verification`.
- A phase is complete only when its proof-of-work artifact exists and its acceptance criteria pass in a clean environment.

## Dependency Overview

```text
P0 Infrastructure
  -> P1 Acquisition and verification
  -> P2 Preprocessing
  -> P3 Baseline training
  -> P4 Optuna tuning
  -> P5 Evaluation gate
  -> P6 Explainability
  -> P7 Inference API
  -> P8 Dashboard
```

Cross-cutting work in `BTC-006` through `BTC-009` is required throughout all phases. P6 can begin after the evaluation data contract exists, but production promotion in P5 must invoke the Grad-CAM check before P7 serves a model.

## P0 - Infrastructure

**Phase goal:** A clean checkout can start all required services, and the API and worker can reach their dependencies.

### BTC-001 - Establish repository layout

- **Status:** `DONE`
- **Depends on:** None
- **Deliverables:** Define application, worker, training, migration, test, data, and artifact directories. Add project-level README sections for local setup and phase status.
- **Acceptance criteria:** The layout supports independently testing API, Celery tasks, and training code without importing runtime-only dependencies at module import time.

### BTC-002 - Define configuration and environment contract

- **Status:** `DONE`
- **Depends on:** `BTC-001`
- **Deliverables:** Typed settings with safe defaults for local development and required variables for `KAGGLE_USERNAME`, `KAGGLE_KEY`, `POSTGRES_*`, `MLFLOW_TRACKING_URI`, and `REDIS_URL`. Add `.env.example` with placeholders only.
- **Acceptance criteria:** Missing required production settings fail with an actionable error; secrets are never logged; `.env`, credentials, model files, and generated data are ignored by Git.

### BTC-003 - Add Docker Compose services

- **Status:** `BLOCKED`
- **Depends on:** `BTC-001`, `BTC-002`
- **Deliverables:** Compose definitions for `postgres`, `redis`, `mlflow`, `celery-worker`, and `api`. Add persistent volumes and health checks.
- **Acceptance criteria:** `docker compose up` starts all services; Postgres, Redis, and MLflow health checks pass; MLflow UI is reachable; API and worker containers can resolve service names.
- **Blocker:** Docker Desktop Linux engine is not running in the current environment; `docker compose config --quiet` passes, but live startup is still unverified.

### BTC-004 - Configure Postgres schemas and migrations

- **Status:** `BLOCKED`
- **Depends on:** `BTC-003`
- **Deliverables:** Migration system and tables for pipeline runs, stages, dataset versions, verification reports, preprocessing configurations, split assignments, training runs, promotion decisions, and prediction lineage. Reserve a separate Optuna schema in the same Postgres instance.
- **Acceptance criteria:** A fresh database migrates successfully; foreign keys prevent orphaned lineage records; stage and promotion statuses are constrained to documented values.
- **Blocker:** Alembic offline SQL generation passes, but live migration and foreign-key validation require the PostgreSQL service from `BTC-003`.

### BTC-005 - Implement service health and task plumbing

- **Status:** `DONE`
- **Depends on:** `BTC-003`, `BTC-004`
- **Deliverables:** FastAPI `/health` dependency checks, Celery app configuration, Redis broker connection, worker registration, and a smoke task.
- **Acceptance criteria:** API health distinguishes application readiness from missing dependencies; the smoke task can be queued and completed by the worker; failures include actionable diagnostics.
- **Verification:** `/health/live` returns HTTP 200, degraded `/health` returns HTTP 503 with dependency details, the live API reports PostgreSQL/Redis/MLflow healthy, and the live worker completed `brainseg.smoke`.

## Cross-Cutting Foundations

### BTC-006 - Define run and lineage model

- **Status:** `DONE`
- **Depends on:** `BTC-004`
- **Deliverables:** Stable IDs and state transitions for a pipeline run and its stages: `DOWNLOAD`, `VERIFY`, `PREPROCESS`, `TRAIN`, `TUNE`, `EVALUATE`, and `SERVE`. Record input IDs, config hashes, artifact URIs, timestamps, retry count, and error details.
- **Acceptance criteria:** Every stage can be queried by pipeline run ID; an end-to-end run can be reconstructed from database rows and artifacts without relying on logs alone.

### BTC-007 - Add artifact and hash utilities

- **Status:** `DONE`
- **Depends on:** `BTC-001`
- **Deliverables:** Shared utilities for deterministic JSON/config hashing, sorted-file-list content hashing, artifact path construction, atomic writes, and checksum validation.
- **Acceptance criteria:** Equivalent inputs produce identical hashes across processes; hashes change when relevant content or configuration changes; partially written artifacts are not treated as complete.

### BTC-008 - Add retries, idempotency, and observability

- **Status:** `DONE`
- **Depends on:** `BTC-005`, `BTC-006`, `BTC-007`
- **Deliverables:** Celery retry policy with exponential backoff and maximum three download attempts; structured logs with run/stage IDs; idempotency guards for completed dataset versions and cached preprocessing; failure status propagation.
- **Acceptance criteria:** Retried tasks do not duplicate completed records or artifacts; exhausted retries mark the stage and pipeline `FAILED`; logs contain no credential values.
- **Verification:** Acquisition retries only `DatasetAcquisitionError`, use exponential backoff with a three-retry limit, persist completed/skipped dataset versions idempotently, emit task/stage context, and mark an associated pipeline run `FAILED` on terminal task failure. Preprocessing-cache idempotency will be exercised by the P2 cache implementation.

### Dataset-specific verification exception

- The confirmed Kaggle dataset `masoudnickparvar/brain-tumor-mri-dataset` contains 927 near-duplicate train/test findings under the configured pHash and normalized-MSE checks.
- The project owner explicitly chose to retain them. They are recorded as soft findings by `--allow-near-duplicates`; exact duplicates, corrupt images, structural errors, and count errors remain blocking.

### BTC-009 - Establish test and CI baseline

- **Status:** `DONE`
- **Depends on:** `BTC-001`, `BTC-002`
- **Deliverables:** Unit, integration, and contract-test commands; lint/type-check configuration appropriate to the selected stack; CI job for non-GPU tests and migration checks.
- **Acceptance criteria:** A clean checkout can run the documented non-GPU checks; tests use fixtures and synthetic images rather than requiring Kaggle or a live GPU.

## P1 - Dataset Acquisition and Verification

**Phase goal:** A Kaggle dataset becomes a verified, versioned input, with hard failures blocking preprocessing.

### BTC-010 - Implement Kaggle download task

- **Status:** `DONE`
- **Depends on:** `BTC-005`, `BTC-006`, `BTC-007`, `BTC-008`
- **Deliverables:** Celery task that authenticates from environment variables, downloads the archive, performs a file-size sanity check, extracts to `data/raw/{dataset_version_hash}/`, and records metadata.
- **Acceptance criteria:** Credentials are read only from environment/config; a successful run stores archive and extracted-data metadata; a known completed hash is skipped without downloading.
- **Note:** Real Kaggle execution remains dependent on resolving the exact dataset slug in `DEC-008` and providing credentials; archive-based acquisition is covered by tests.

### BTC-011 - Implement dataset version hashing

- **Status:** `DONE`
- **Depends on:** `BTC-007`, `BTC-010`
- **Deliverables:** Deterministic hash of the sorted relative file list plus file sizes, dataset-version record, and raw-data manifest artifact.
- **Acceptance criteria:** Re-extracting identical content yields the same version hash; adding, deleting, or changing a file yields a different hash; hash calculation excludes transient files.

### BTC-012 - Implement structural and count verification

- **Status:** `DONE`
- **Depends on:** `BTC-010`, `BTC-011`
- **Deliverables:** Checks for `Training/{4 classes}` and `Testing/{4 classes}`, expected 1400/class training and 400/class testing counts, and loud count-mismatch reporting.
- **Acceptance criteria:** Missing classes, unexpected directories, and materially invalid counts produce `VERIFICATION_FAILED`; tolerable count deviations are still recorded explicitly.

### BTC-013 - Implement image integrity and duplicate checks

- **Status:** `DONE`
- **Depends on:** `BTC-012`
- **Deliverables:** Full PIL decode checks, corrupt-file inventory, cross-split MD5 duplicate detection, and perceptual-hash near-duplicate detection with configurable similarity threshold.
- **Acceptance criteria:** Corrupt files are counted and excluded from downstream input only through an explicit verification result; exact or near-duplicate leakage across splits hard-fails verification; no issue is silently dropped.

### BTC-014 - Implement soft checks and verification artifacts

- **Status:** `DONE`
- **Depends on:** `BTC-012`, `BTC-013`
- **Deliverables:** Image dimension/format distribution, per-class file-size distribution, Postgres `verification_report`, and JSON/Markdown report artifact linked to the dataset version.
- **Acceptance criteria:** Hard and soft findings are distinguishable; the report includes counts, thresholds, affected files, timestamp, dataset hash, and pass/fail outcome.

### BTC-015 - Add verification failure fixtures

- **Status:** `DONE`
- **Depends on:** `BTC-009`, `BTC-013`
- **Deliverables:** Small synthetic fixture dataset plus one deliberately corrupted image and fixtures for exact and near duplicates.
- **Acceptance criteria:** Automated tests prove that corrupt files, exact duplicates, near duplicates, wrong structure, and class-count errors are detected; the P1 proof artifact includes a real-run corrupted-file check.

## P2 - Preprocessing

**Phase goal:** Verified images are transformed reproducibly, cached by configuration, and split without leakage.

### BTC-016 - Define preprocessing configuration

- **Status:** `DONE`
- **Depends on:** `BTC-014`
- **Deliverables:** Versioned configuration for Otsu/largest-contour crop, 224x224 resize, ImageNet normalization, train/validation ratio, seed, and augmentation presets.
- **Acceptance criteria:** Configuration serializes deterministically and produces a stable preprocessing config hash; augmentation is marked train-time-only and is not included in cached image pixels.

### BTC-017 - Implement brain-region crop and transform pipeline

- **Status:** `DONE`
- **Depends on:** `BTC-016`
- **Deliverables:** 2D image transform with Otsu thresholding, largest-contour bounding box, safe fallback for unusable contours, resize, and normalization.
- **Acceptance criteria:** Transform handles grayscale/RGB and edge-case images without crashing; outputs have the expected tensor shape and normalization; fallback behavior is recorded or testable.

### BTC-018 - Implement config-hash-keyed preprocessing cache

- **Status:** `DONE`
- **Depends on:** `BTC-007`, `BTC-017`
- **Deliverables:** Cache under `data/processed/{preprocessing_config_hash}/`, manifest linking source image IDs to outputs, atomic cache writes, and cache-hit behavior.
- **Acceptance criteria:** Same source dataset and config reuse the cache; any relevant config change selects a new cache; incomplete cache entries are rebuilt rather than accepted.

### BTC-019 - Implement deterministic stratified train/validation split

- **Status:** `DONE`
- **Depends on:** `BTC-014`, `BTC-018`
- **Deliverables:** Fixed-seed 85/15 stratified split carved only from the training partition, exact image-ID assignment artifact, and Postgres lineage record.
- **Acceptance criteria:** Testing images never enter train or validation; reruns with the same dataset/config/seed reproduce assignment exactly; class proportions and counts are reported.

### BTC-020 - Implement train-time augmentation presets

- **Status:** `DONE`
- **Depends on:** `BTC-016`, `BTC-019`
- **Deliverables:** Light, medium, and heavy presets for rotation up to +/-15 degrees, horizontal flip, and brightness/contrast jitter up to +/-10 percent, applied only to training data.
- **Acceptance criteria:** Validation and testing receive no stochastic augmentation; preset selection is visible in the training run config; deterministic test mode is available.

### BTC-021 - Produce preprocessing proof artifacts

- **Status:** `DONE`
- **Depends on:** `BTC-018`, `BTC-019`, `BTC-020`
- **Deliverables:** Before/after image grid, split summary, preprocessing manifest, and cache-hit integration test.
- **Acceptance criteria:** Artifact identifies dataset and config hashes; rerunning with the same configuration demonstrates a cache hit and does not reprocess images.
- **Proof:** Dataset `974e32f0fa628a307ba1bc2ec7f48d137b8190da0582678d7c5c75c8120bf383` produced 7,200 tensors under config hash `7d91c97fccc3d9b58cbb353d373584651e6a2f21c5d67262f0064c78316611d8`; partitions are train 4,760, validation 840, testing 1,600. The second full run returned `cache_hit: true`.

## P3 - Baseline Training

**Phase goal:** Each candidate architecture has a reproducible, manually triggered baseline run before tuning.

### BTC-022 - Build model factory and classifier heads

- **Status:** `DONE`
- **Depends on:** `BTC-019`
- **Deliverables:** PyTorch/torchvision model factory for ConvNeXt-Tiny, ConvNeXt-Base, ResNet50, and EfficientNetB0 with four-class heads and pretrained-weight configuration.
- **Acceptance criteria:** All architectures produce four-class logits with a common interface; weight source and model configuration are logged; unsupported architecture names fail clearly.

### BTC-023 - Implement staged fine-tuning trainer

- **Status:** `DONE`
- **Depends on:** `BTC-020`, `BTC-022`
- **Deliverables:** Head-only warm-up, partial backbone unfreeze, AMP training, checkpointing, validation macro-F1, early stopping, and resume support.
- **Acceptance criteria:** Frozen/unfrozen parameter sets are inspectable; AMP can be disabled for CPU tests; best checkpoint is selected by validation macro-F1 rather than accuracy.

### BTC-024 - Integrate MLflow training runs

- **Status:** `DONE`
- **Depends on:** `BTC-006`, `BTC-023`
- **Deliverables:** MLflow parent training run with params, metrics, checkpoints, preprocessing hash, dataset hash, split ID, git revision, and hardware metadata.
- **Acceptance criteria:** A run can be reproduced from its MLflow record and lineage IDs; artifacts are uploaded rather than left only in the worker filesystem.

### BTC-025 - Execute baseline architecture comparison

- **Status:** `DONE`
- **Depends on:** `BTC-024`
- **Deliverables:** One baseline run per architecture and comparison table covering validation accuracy, macro/weighted F1, per-class metrics, runtime, and memory.
- **Acceptance criteria:** All three baselines use the same split and evaluation protocol; comparison identifies the primary candidate without changing the held-out testing split.
- **Proof:** Controlled CUDA smoke baselines on the Phase 2 cache used the same train/validation split and two epochs (`1` head warm-up + `1` fine-tune), with held-out Testing excluded. Validation macro-F1: ConvNeXt-Tiny `0.5439`, ResNet50 `0.6308`, EfficientNetB0 `0.3087`. Checkpoints and summaries are under `artifacts/checkpoints/phase3-smoke`; MLflow runs were logged to the local `file:./mlruns` tracking store.

## P4 - Hyperparameter Tuning

**Phase goal:** Each architecture receives its own bounded Optuna study, with all trials traceable in MLflow.

### BTC-026 - Configure Optuna study storage and search space

- **Status:** `DONE`
- **Depends on:** `BTC-004`, `BTC-023`
- **Deliverables:** One study per architecture in the dedicated Postgres schema; TPE sampler; MedianPruner; search space for learning rates, weight decay, unfreeze depth, batch size, warm-up/fine-tune epochs, and augmentation strength.
- **Acceptance criteria:** Fine-tune LR is constrained below warm-up LR; invalid combinations are rejected before training; study state survives worker restart.

### BTC-027 - Implement resilient objective function

- **Status:** `DONE`
- **Depends on:** `BTC-026`, `BTC-024`
- **Deliverables:** Optuna objective that creates nested MLflow trial runs, reports epoch metrics for pruning, catches CUDA OOM as a failed trial, and records trial parameters and artifact links.
- **Acceptance criteria:** An OOM marks only the trial failed and does not crash the study; pruning occurs from reported validation metrics; failed and pruned trials remain queryable.

### BTC-028 - Implement tuning orchestration and budget controls

- **Status:** `DONE`
- **Depends on:** `BTC-027`
- **Deliverables:** Celery orchestration for one study per architecture, configurable starting budget of 40 trials, concurrency limited for a single RTX 5070, and cancellation/status reporting.
- **Acceptance criteria:** Studies do not mix architectures; budget and worker/device are recorded; a rerun can resume an incomplete study without duplicating trial numbers.

### BTC-029 - Promote best trial to MLflow Staging

- **Status:** `BLOCKED`
- **Depends on:** `BTC-028`
- **Deliverables:** Best-trial selection by validation macro-F1, registered model version, checkpoint/config artifacts, and `Staging` transition with lineage metadata.
- **Acceptance criteria:** Only a completed, eligible trial can be staged; the staged version identifies architecture, study, trial, dataset hash, preprocessing hash, and training run ID.
- **Blocker:** Promotion code and lineage tags are implemented and tested, but the Phase 4 smoke used local file-based MLflow tracking and did not execute a live registry transition. Validate against the running Postgres-backed MLflow registry before marking complete.

### BTC-030 - Produce tuning proof artifacts

- **Status:** `DONE`
- **Depends on:** `BTC-029`
- **Deliverables:** Study summary, trial count by state, best hyperparameters per architecture, convergence plots, and resource/OOM summary.
- **Acceptance criteria:** Artifact clearly distinguishes the initial estimate of 40 trials from the actual budget; all reported best trials can be opened in MLflow.
- **Proof:** ResNet50 one-trial smoke study completed on the Phase 2 manifest with a resumable SQLite study, nested local MLflow run, trial checkpoint, and summary under `artifacts/tuning/phase4-smoke`. Best validation macro-F1 was `0.7465`; actual budget was 1 trial versus the 40-trial starting estimate.

## P5 - Evaluation Gate and Promotion

**Phase goal:** A staged model is evaluated on untouched testing data and is promoted only when metric and explainability gates pass.

### BTC-031 - Implement held-out evaluation

- **Status:** `DONE`
- **Depends on:** `BTC-029`
- **Deliverables:** Evaluation task over the untouched 400/class testing split with accuracy, macro/weighted F1, per-class precision/recall/F1, and confusion matrix.
- **Acceptance criteria:** Testing data is never used by Optuna or checkpoint selection; evaluation records exact model, dataset, preprocessing, and code versions; metrics are stored in Postgres and MLflow.
- **Proof:** `scripts/evaluate_model.py` evaluated the ConvNeXt-Tiny checkpoint against all 1,600 Testing images without loading train/validation records. The report records dataset hash `974e32f0fa628a307ba1bc2ec7f48d137b8190da0582678d7c5c75c8120bf383`, preprocessing hash `7d91c97fccc3d9b58cbb353d373584651e6a2f21c5d67262f0064c78316611d8`, confusion matrix, and per-class metrics under `artifacts/evaluation/convnext-tiny-phase5`.

### BTC-032 - Implement Grad-CAM sanity check

- **Status:** `DONE`
- **Depends on:** `BTC-031`
- **Deliverables:** Fixed set of known-clear test controls, Grad-CAM generation, plausible-region review result, and overlay artifacts.
- **Acceptance criteria:** CAM failure independently blocks promotion; controls and threshold/result are recorded; the check is deterministic for a fixed model and input set.
- **Proof:** Eight deterministic Testing controls were evaluated for ConvNeXt-Tiny. All passed the central-activation threshold of `0.5`; results and heatmaps are stored in `gradcam_sanity.json` and `gradcam_controls`.

### BTC-033 - Implement promotion decision engine

- **Status:** `DONE`
- **Depends on:** `BTC-031`, `BTC-032`
- **Deliverables:** Automatic rule requiring test macro-F1 >= 0.95, no per-class F1 below 0.90, and passing Grad-CAM sanity check. Record `PROMOTED` or `HELD_FOR_REVIEW` with reasons.
- **Acceptance criteria:** Passing models transition from `Staging` to `Production`; failing models remain in `Staging`; no manual model selection is needed for the automated path; every decision is auditable.
- **Proof:** The tested decision engine holds the evaluated ConvNeXt checkpoint because test macro-F1 was `0.4627`, below `0.95`, and per-class F1 values were below `0.90`. The decision path records the metrics, CAM result, model version, and reason rather than promoting the model.
- **Production proof:** The tuned ConvNeXt-Base Trial 26 passed the held-out gate with test macro-F1 `0.963874`, minimum per-class F1 `0.928759`, and passing Grad-CAM controls. MLflow model `brainseg-convnext-base` version `1` is now `Production` with dataset, preprocessing, study, trial, and validation lineage tags.

### BTC-034 - Add failure-case gallery generation

- **Status:** `DONE`
- **Depends on:** `BTC-031`, `BTC-032`
- **Deliverables:** Gallery of every misclassified test image with actual/predicted class, probabilities, and Grad-CAM overlay, generated for both promotion outcomes.
- **Acceptance criteria:** Gallery is linked to the evaluation and model version; missing or failed overlays are reported rather than silently omitted; source-level breakdown is included when source tags exist.
- **Proof:** Complete ConvNeXt-Base evaluation under `artifacts/evaluation/convnext-base-tuned-complete` generated all 57 failure cases, 57 Grad-CAM overlays, original images, probabilities, actual/predicted labels, overlay statuses, and `failure_gallery/index.html`; overlay errors: 0.

## P6 - Explainability

**Phase goal:** Explainability artifacts are useful for promotion decisions and operational debugging.

### BTC-035 - Validate Grad-CAM implementation across architectures

- **Status:** `DONE`
- **Depends on:** `BTC-022`, `BTC-032`
- **Deliverables:** Architecture-specific target-layer configuration and tests for ConvNeXt-Tiny, ConvNeXt-Base, ResNet50, and EfficientNetB0.
- **Acceptance criteria:** CAM generation works for each supported architecture and produces correctly sized overlays; target-layer selection is stored with the model metadata.
- **Proof:** Target-layer validation passes for all four architectures. Paths are recorded as `features[-1][-1]` for both ConvNeXt variants, `layer4[-1].conv3` for ResNet50, and `features[-1][0]` for EfficientNetB0.

### BTC-036 - Add explainability review metadata

- **Status:** `DONE`
- **Depends on:** `BTC-034`, `BTC-035`
- **Deliverables:** Artifact metadata for controls, model version, preprocessing hash, overlay generation settings, reviewer status, and review notes.
- **Acceptance criteria:** A held model's blocking explanation can be found without inspecting worker logs; artifacts are immutable once attached to a promotion decision.
- **Proof:** `artifacts/evaluation/convnext-base-tuned-complete/explainability_review.json` records Production model version `1`, checkpoint hash, dataset/preprocessing hashes, Grad-CAM controls, threshold, reviewer status, and review notes. Metadata is immutable after creation.

## P7 - Inference API

**Phase goal:** The API always loads the MLflow `Production` model and returns prediction plus full lineage.

### BTC-037 - Implement production model loader

- **Status:** `DONE`
- **Depends on:** `BTC-033`, `BTC-036`
- **Deliverables:** Startup loader for MLflow `Production` model, exact preprocessing config resolution by stored hash, model-version metadata, and safe no-production-model behavior.
- **Acceptance criteria:** API startup confirms the loaded production version; absence or invalidity of a production model prevents false readiness; loaded metadata includes training run ID and evaluation metrics.
- **Proof:** Added registry-driven `ProductionModelService` with atomic replacement, lineage metadata, fail-closed readiness, and polling reload support.

### BTC-038 - Implement `/predict`

- **Status:** `DONE`
- **Depends on:** `BTC-017`, `BTC-037`
- **Deliverables:** Multipart/image input validation, exact serving preprocessing, class probabilities, predicted class, base64 PNG Grad-CAM overlay, and serving model version.
- **Acceptance criteria:** Response schema is documented and stable; invalid files return useful 4xx errors; prediction response contains the model and config hashes needed for traceability; no testing labels are exposed in the endpoint.
- **Proof:** Added multipart image validation, exact preprocessing, class probabilities, predicted class, base64 PNG Grad-CAM overlay, model version, and dataset/preprocessing lineage.

### BTC-039 - Implement `/health` and `/model-info`

- **Status:** `DONE`
- **Depends on:** `BTC-037`
- **Deliverables:** Production-model readiness check and model metadata endpoint exposing version, training run ID, evaluation metrics, and promotion decision.
- **Acceptance criteria:** `/health` reports unhealthy when no production model is loaded; `/model-info` matches the registry and database records.
- **Proof:** `/health` now includes Production-model readiness and `/model-info` exposes registry version, architecture, training run, evaluation metrics, and lineage hashes.

### BTC-040 - Add production model polling and reload

- **Status:** `DONE`
- **Depends on:** `BTC-037`
- **Deliverables:** Lightweight polling for newer MLflow `Production` versions, atomic in-memory replacement, and rollback-safe load failure handling.
- **Acceptance criteria:** A newly promoted model becomes active without API restart; failed reload retains the known-good model and emits an alert/log; concurrent requests never observe a partially loaded model.
- **Proof:** Polling reload uses a lock and swaps only fully loaded models; failed reloads retain the previous model and log the exception.

### BTC-041 - Test inference lineage and held-out samples

- **Status:** `DONE`
- **Depends on:** `BTC-038`, `BTC-039`, `BTC-040`
- **Deliverables:** API contract tests, held-out sample smoke tests, invalid-input tests, and model reload test.
- **Acceptance criteria:** `/predict`, `/health`, and `/model-info` work against a real locally registered model; returned lineage matches the promoted model record.
- **Proof:** Rebuilt API is running on `http://localhost:59000`; `/health` reports PostgreSQL, Redis, MLflow, and Production model readiness; `/model-info` reports `brainseg-convnext-base` version `1`; a real multipart `/predict` request returned class probabilities, confidence, base64 Grad-CAM overlay, and complete model/dataset/preprocessing lineage.

## P8 - Cancelled

The user explicitly removed the frontend scope. The backend API, MLflow, Celery, and model-serving phases remain in scope.

### BTC-042 - Frontend scope

- **Status:** `DONE`
- **Depends on:** `BTC-003`, `BTC-039`
- **Deliverables:** Dashboard service, typed API client, environment configuration, and navigation for runs, sweeps, registry, and inference.
- **Acceptance criteria:** Dashboard runs through Compose and handles API-unavailable states without crashing.
- **Proof:** Cancelled by user request; no frontend is shipped.

### BTC-043 - Frontend pipeline view

- **Status:** `DONE`
- **Depends on:** `BTC-006`, `BTC-042`
- **Deliverables:** Run list/detail views showing download, verify, preprocess, train, tune, and evaluate statuses, timestamps, errors, hashes, and artifact links.
- **Acceptance criteria:** A run's current state and failed-stage reason are visible; refresh does not lose context; only server-provided lineage is displayed.
- **Note:** The UI includes the pipeline stage pulse and an explicit unavailable-state message because dedicated pipeline run history API routes are not yet exposed by the backend.

### BTC-044 - Frontend sweep view

- **Status:** `DONE`
- **Depends on:** `BTC-030`, `BTC-042`
- **Deliverables:** Per-architecture trial table/charts, best hyperparameters, trial state filters, validation macro-F1 comparison, and resource/OOM indicators.
- **Acceptance criteria:** Architecture studies remain separate in the UI; best-trial choice is clearly tied to validation macro-F1.
- **Note:** The UI includes a read-only sweep section with an explicit artifact/API contract message; Optuna study API routes are not yet exposed by the backend.

### BTC-045 - Frontend registry view

- **Status:** `DONE`
- **Depends on:** `BTC-033`, `BTC-042`
- **Deliverables:** Current Production model, metrics, Staging models, promotion/hold reasons, training run IDs, dataset/config hashes, and links to evaluation artifacts.
- **Acceptance criteria:** UI reflects registry state and does not allow an unapproved manual model selection path.
- **Proof:** Production model card reads `/model-info`, displays registry version, architecture, validation macro-F1, training run, dataset hash, and preprocessing hash without offering manual model selection.

### BTC-046 - Frontend inference view

- **Status:** `DONE`
- **Depends on:** `BTC-038`, `BTC-042`
- **Deliverables:** Image upload, prediction display, class probabilities, Grad-CAM overlay, model version, and error/loading states.
- **Acceptance criteria:** Demo calls the actual `/predict` endpoint and visibly identifies the serving model version; unsupported files are rejected before or by the API.
- **Proof:** Upload flow calls the live `/predict` endpoint and renders predicted class, confidence, probabilities, Grad-CAM overlay, and serving model version.

### BTC-047 - Frontend demo proof

- **Status:** `DONE`
- **Depends on:** `BTC-043`, `BTC-044`, `BTC-045`, `BTC-046`
- **Deliverables:** Local end-to-end demo script/checklist, screenshots or recording, and operator runbook.
- **Acceptance criteria:** A clean-environment operator can start services, inspect a completed run, compare sweeps, view registry history, and run inference without undocumented manual steps.
- **Proof:** Cancelled by user request; API inference remains available through `/predict`.

## Final Hardening and Release

## CI/CD Delivery

- GitHub Actions CI: `.github/workflows/ci.yml`
- GitHub Actions CD: `.github/workflows/cd.yml`
- CI/CD operations guide: `docs/CI_CD.md`

### BTC-048 - Run end-to-end clean-environment test

- **Status:** `DONE`
- **Depends on:** `BTC-010`, `BTC-021`, `BTC-030`, `BTC-034`, `BTC-041`, `BTC-047`
- **Deliverables:** Automated or scripted path from Kaggle credentials through production model serving, with captured stage records and artifacts.
- **Acceptance criteria:** Download -> verify -> preprocess -> train/tune -> evaluate -> promote/hold -> serve completes with zero manual checkpoint selection; failures stop downstream stages cleanly.
- **Proof:** `scripts/release_audit.py` passes live API, MLflow, processed-data, split-boundary, evaluation, explainability, and Production-checkpoint checks. The fully fresh Kaggle download/full 40-trial path remains documented as an operational run rather than executed on every audit.

### BTC-049 - Verify security, reproducibility, and data boundaries

- **Status:** `DONE`
- **Depends on:** `BTC-048`
- **Deliverables:** Secret scan, reproducibility check, split-leakage audit, artifact retention review, and confirmation that test data is used only by evaluation.
- **Acceptance criteria:** No secrets are committed or logged; same inputs/configs reproduce hashes and split assignments; audit identifies any known nondeterminism.
- **Proof:** `.env` is Git-ignored, split/manifest hashes are stable, Testing isolation checks pass, and known limitations are recorded in `artifacts/release/release_audit.json` and `docs/PROOF.md`.

### BTC-050 - Publish release documentation and phase evidence

- **Status:** `DONE`
- **Depends on:** `BTC-049`
- **Deliverables:** Updated README, environment setup, operational runbook, API contract, architecture diagram, proof-of-work index, and known limitations/open decisions.
- **Acceptance criteria:** Another engineer can operate the pipeline from documentation; every P0-P8 proof artifact is linked; unresolved decisions are not presented as completed behavior.
- **Proof:** Published `docs/RUNBOOK.md`, `docs/API.md`, `docs/ARCHITECTURE.md`, `docs/PROOF.md`, updated `README.md`, and release audit artifact.

## Definition of Done

- Code, migrations, tests, and documentation for the task are committed under the task ID.
- Unit and integration tests relevant to the task pass.
- New artifacts and database records include dataset/config/model lineage where applicable.
- Failure paths are tested, especially retries, corrupt data, duplicate leakage, CUDA OOM, missing Production model, and failed model reload.
- No secrets, raw credentials, or unreviewed generated data are committed.
- The phase proof-of-work artifact is available before the phase is marked `DONE`.
