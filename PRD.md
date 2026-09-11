# PRD: Brain Tumor MRI Classification — Automated Pipeline (v2)

**Status:** Draft v2 — supersedes v1, expands scope to a fully automated pipeline
**Task ID prefix:** `BTC-XXX`
**Portfolio category:** CV/ML, medical imaging, automated MLOps pipeline

---

## 1. Scope Change from v1

v1 assumed a manual-script, lighter stack (no orchestration). This version brings the project up to the same automation level as the BraTS segmentation project: dataset acquisition, verification, preprocessing, training, tuning, and inference all run as an orchestrated pipeline with lineage tracking, not a hand-run notebook. Everything below replaces v1's Sections 6-11.

## 2. Goals

- Zero manual steps between "provide a Kaggle credential" and "trained, tuned, and served model."
- Full lineage: every dataset version, preprocessing config, training run, and hyperparameter trial is traceable back to its inputs.
- Hyperparameters come from an automated search with a defined space and budget, not hand-picked values.
- The inference API always serves the model that was automatically promoted through the evaluation gate, so there's no manual "which checkpoint do I load" step.

## 3. Non-Goals

Unchanged from v1: no segmentation, no survival prediction, no 3D volumetric work. Also explicitly out of scope for this version: multi-GPU distributed training (single RTX 5070 is the target environment), and continuous online retraining (pipeline runs on demand or on a schedule, not on a live data stream).

## 4. System Architecture

| Component | Role |
|---|---|
| FastAPI | Single app with two route groups: `/pipeline/*` (trigger and check pipeline stages) and `/predict/*` (inference serving) |
| Celery + Redis | Task queue for long-running stages: download, verify, preprocess, train, tune |
| PostgreSQL | Pipeline run metadata and lineage — dataset version hash, preprocessing config hash, training run IDs, promotion decisions |
| MLflow | Experiment tracking (params/metrics/artifacts per trial) and model registry (Staging/Production stages) |
| Optuna | Hyperparameter search, using the same Postgres instance as its backend store (separate schema) rather than a second database |
| Docker Compose | Containerizes postgres, redis, mlflow, celery-worker, api |

## 5. Automated Dataset Acquisition

Source: Kaggle "Brain Tumor MRI Dataset" via the Kaggle API.

**Download task (Celery), steps:**
1. Authenticate via `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars (never committed to the repo).
2. Download the dataset archive.
3. Verify the download completed (file size sanity check against a known expected range).
4. Extract to a versioned directory: `data/raw/{dataset_version_hash}/`.
5. Compute a content hash (hash of the sorted file list + sizes) as the dataset version identifier, and log it to Postgres.

**Reliability:**
- Retry with exponential backoff, max 3 attempts. Pipeline status flips to `FAILED` with the error surfaced if all retries exhaust.
- Idempotency: if a dataset version hash already exists in Postgres with status `COMPLETE`, skip re-downloading.

## 6. Automated Verification (gate before preprocessing)

Runs immediately after download. Hard failures halt the pipeline (`VERIFICATION_FAILED`); soft findings are logged but don't block.

**Hard checks:**
- Directory structure matches expected (`Training/{4 classes}`, `Testing/{4 classes}`).
- Class counts match expected (1400/1400/1400/1400 train, 400/400/400/400 test); a mismatch is logged loudly even if within a small tolerance.
- Every image passes a full PIL decode check; corrupt files are excluded and counted, not silently dropped.
- No exact-duplicate files across Training/Testing (MD5 comparison) — this directly tests the dataset card's "eliminated overlap" claim instead of trusting it at face value.
- No near-duplicate leakage via perceptual hashing (pHash, similarity threshold) — catches resized or re-compressed copies MD5 would miss.

**Soft checks (reported, not blocking):**
- Image dimension and format distribution.
- Per-class file size distribution, as a rough proxy for scan quality variance.

All results are written to Postgres as a `verification_report` linked to the dataset version hash, and saved as a JSON/markdown artifact.

## 7. Automated Preprocessing

- **Margin/region crop:** Otsu threshold + largest-contour bounding box to crop to the brain region. This is a 2D-slice dataset (not volumetric), so a contour crop is the appropriate equivalent of the "margin removal" the dataset card recommends — not full 3D skull-stripping.
- **Resize:** 224x224 (matches ConvNeXt/ResNet/EfficientNet input expectations).
- **Normalize:** ImageNet mean/std.
- **Caching:** output written to `data/processed/{preprocessing_config_hash}/`, keyed by a hash of the preprocessing config, so changing any preprocessing parameter produces a new cache instead of silently reusing stale output.
- **Train/val split:** 85/15 stratified split carved from the 1400/class training set, fixed seed, with the exact image-ID assignment logged to Postgres so it's reproducible and auditable.
- **Augmentation** (train-time only, not cached): rotation (±15°), horizontal flip, brightness/contrast jitter (±10%), configurable via the same preprocessing config.

## 8. Model Training

- PyTorch, torchvision pretrained weights, mixed-precision (AMP) training.
- Three candidate architectures, each gets its **own** Optuna study rather than one sweep with architecture as a categorical — cleaner comparison, avoids the sweep spending trials on architecture/hyperparameter interactions that muddy the "which model wins" question:
  - **ConvNeXt-Tiny** (primary)
  - ResNet50 (baseline)
  - EfficientNetB0 (baseline)
- Staged fine-tuning per architecture: head-only warm-up, then partial backbone unfreeze.

## 9. Automated Hyperparameter Tuning

Optuna study per architecture.

| Parameter | Range / choices | Notes |
|---|---|---|
| LR (head warm-up) | log-uniform, 1e-5 to 1e-3 | |
| LR (fine-tune) | log-uniform, 1e-6 to 1e-4 | Constrained to be lower than the warm-up LR |
| Weight decay | log-uniform, 1e-4 to 1e-1 | Range covers ConvNeXt's typical ~0.05 default |
| Unfreeze depth | categorical: last 1 / 2 / 3 stages | |
| Batch size | categorical: 16 / 32 / 64 | Constrained by 12GB VRAM at 224x224 — verified per trial, OOM caught not crashed |
| Warm-up epochs | int, 2-5 | |
| Fine-tune epochs | int, 5-15 | Early stopping on val macro-F1, patience 3 |
| Augmentation strength | categorical: light / medium / heavy | Preset combos, not independently tuned per sub-parameter, to keep the search space bounded |

- **Sampler:** TPE (Optuna default).
- **Pruner:** MedianPruner — trials clearly underperforming by epoch 3 get cut early.
- **Budget:** 40 trials per architecture as a starting estimate — flagged as an estimate, not a fixed commitment, since actual convergence behavior on a single GPU will tell you whether that's enough or overkill.
- **Objective:** validation macro-F1, not raw accuracy — this penalizes a model that's strong on 3 classes and weak specifically on the glioma/meningioma boundary, which is the confusion pair flagged as the main risk back in v1.
- All trials logged to MLflow as nested runs under a parent sweep run. The best trial's checkpoint is promoted to the MLflow Model Registry as `Staging`.

## 10. Evaluation Gate (before Production promotion)

Runs automatically once a `Staging` model exists.

- Full evaluation on the untouched 400/class Testing split (never seen during tuning).
- Metrics: accuracy, macro/weighted F1, per-class precision/recall/F1, confusion matrix.
- Grad-CAM sanity check against a fixed set of "known-clear" test images — if the CAM doesn't localize to a plausible tumor region on these controls, the model is flagged for manual review instead of auto-promoted, regardless of its metric score.
- **Auto-promotion rule:** promote to `Production` only if test macro-F1 ≥ 0.95 **and** no per-class F1 below 0.90. Otherwise the model stays in `Staging` pending manual review, and the reason is logged.
- A failure-case gallery (all misclassified test images + Grad-CAM overlay) is generated automatically as an artifact regardless of promotion outcome.

## 11. Inference Serving

- `POST /predict` — accepts an image, applies the exact preprocessing config used by the currently-serving model (loaded from the config hash stored alongside it), returns class probabilities, predicted class, a Grad-CAM overlay (base64 PNG), and the serving model's version.
- `GET /health` — confirms a `Production` model is currently loaded.
- `GET /model-info` — returns the model version, its training run ID, and its evaluation metrics, so any prediction is traceable back to the exact run that produced the serving model.
- The model is loaded from the MLflow registry's `Production` stage at API startup, with a lightweight polling check to reload automatically if a newer model gets promoted — no manual restart needed on every retrain cycle.

## 12. Phased Delivery Plan

| Phase | Deliverable | Proof-of-work artifact |
|---|---|---|
| P0 — Infra | Docker Compose up: Postgres, Redis, MLflow, Celery worker | All services healthy, MLflow UI reachable |
| P1 — Acquisition + verification | Automated download + verification pipeline | `verification_report` artifact from a real run, plus one deliberately-corrupted test file to confirm the corrupt-file check actually catches something |
| P2 — Preprocessing | Automated, cached preprocessing pipeline | Before/after image grid, plus a cache-hit test proving a rerun with the same config doesn't reprocess |
| P3 — Baseline training | One manual run per architecture, no tuning yet | Training curves, architecture comparison table |
| P4 — Hyperparameter tuning | Automated Optuna sweep per architecture | Study summary, best trial per architecture |
| P5 — Evaluation gate | Auto-promotion logic | A `Staging` model that gets evaluated and either promoted or held, with the decision logged |
| P6 — Explainability | Grad-CAM + failure-case gallery | Failure-case gallery artifact |
| P7 — Inference API | Serving endpoints | Working `/predict` tested against held-out samples, `/model-info` returns correct lineage |
| P8 — Dashboard | Pipeline + sweep + live demo UI | Working local demo, screenshot/recording |

## 13. Success Metrics

- Test macro-F1 ≥ 0.95, no per-class F1 below 0.90 — this is the auto-promotion gate itself, not just a reporting target.
- Full pipeline (download → production model serving) runs end-to-end with zero manual intervention on a clean environment.
- Every served prediction is traceable to a specific training run and dataset version.

## 14. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Kaggle API auth failure or rate limiting | Retry with backoff; credentials via env vars, never hardcoded |
| GPU OOM during a sweep trial (e.g. batch 64 + heavy augmentation) | Trial catches the CUDA OOM exception and reports as a failed trial rather than crashing the study; pruner deprioritizes similar configs |
| Sweep takes too long on a single GPU | 40 trials/architecture is a starting estimate; MedianPruner cuts weak trials early; budget can be revised after the first sweep shows convergence behavior |
| Auto-promotion approves a model that's metrically good but has poor Grad-CAM localization | Fixed known-clear-case CAM sanity check gates promotion independently of the metric threshold |
| Domain shift within the stitched dataset (figshare/SARTAJ/Br35H) inflates apparent accuracy | Per-source error breakdown in the failure-case gallery if source is inferable from the merged dataset — flagged as an open question until confirmed whether source-level tags survive the merge |
| Preprocessing cache goes stale after a config change | Config-hash-keyed cache directories prevent silently reusing outdated output |

## 15. Working Conventions

- `TASKS.md` with `BTC-XXX` task IDs, one section per phase.
- Required environment variables: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `POSTGRES_*`, `MLFLOW_TRACKING_URI`, `REDIS_URL`.
- Docker Compose services: `postgres`, `redis`, `mlflow`, `celery-worker`, and `api`.
- Git commits reference task IDs, per existing convention.
