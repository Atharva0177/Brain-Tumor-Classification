# Brain Tumor MRI Classification Decision Record

**Source:** `PRD.md` (Draft v2)  
**Purpose:** Record architectural decisions, constraints, and unresolved choices for the automated classification pipeline.  
**Status vocabulary:** `Accepted`, `Proposed`, `Open`, `Superseded`

## Decision Summary

| ID | Decision | Status |
|---|---|---|
| DEC-001 | Use a staged, containerized service architecture with FastAPI, Celery/Redis, Postgres, MLflow, and Optuna | Accepted |
| DEC-002 | Treat dataset content hashes and preprocessing config hashes as first-class lineage identifiers | Accepted |
| DEC-003 | Use the untouched Testing split only for the final evaluation gate | Accepted |
| DEC-004 | Run one Optuna study per architecture and optimize validation macro-F1 | Accepted |
| DEC-005 | Require both metric thresholds and Grad-CAM sanity checks for Production promotion | Accepted |
| DEC-006 | Load only the MLflow Production model and reload it through polling | Accepted |
| DEC-007 | Keep this release 2D, four-class, single-GPU, and on-demand/scheduled | Accepted |
| DEC-008 | Resolve dataset identity, source provenance, and expected archive characteristics before P1 is implemented | Open |
| DEC-009 | Define the Grad-CAM plausibility protocol and known-clear control set before P5/P6 | Open |
| DEC-010 | Define retention, access control, and deployment exposure before production-like operation | Open |
| DEC-011 | Confirm whether MLflow registry stages are supported by the selected MLflow version or require an equivalent alias workflow | Open |
| DEC-012 | Decide how inference request data and prediction lineage are retained | Open |
| DEC-013 | Retain confirmed near-duplicate MRI pairs as soft findings for this dataset | Accepted |

## DEC-001 - Orchestrated Service Architecture

**Status:** Accepted  
**Context:** The pipeline includes long-running downloads, verification, preprocessing, training, tuning, evaluation, and serving. A manually run script would not satisfy the zero-manual-step or lineage goals.  
**Decision:** Use FastAPI for synchronous control and inference routes; Celery with Redis for asynchronous work; PostgreSQL for operational metadata and lineage; MLflow for experiment tracking and model registry; Optuna backed by a separate schema in the same PostgreSQL instance; Docker Compose for local orchestration.  
**Consequences:** The system has more operational components than a notebook, but failures, retries, and stage state can be observed and resumed. Each service must have health checks, configuration boundaries, and integration tests.  
**Rejected alternatives:** A single long-running FastAPI process and a notebook/manual workflow do not provide durable task execution or reliable stage-level lineage.

## DEC-002 - Content and Configuration Hashes as Lineage Keys

**Status:** Accepted  
**Context:** Reusing stale raw data or processed images would invalidate reproducibility. File timestamps and directory names alone are not sufficient identifiers.  
**Decision:** Identify a raw dataset version with a deterministic hash of the sorted relative file list and file sizes. Identify preprocessing output with a deterministic hash of the complete preprocessing configuration. Store both IDs on all downstream records and artifacts.  
**Consequences:** Cache paths are immutable by configuration, and a run can be traced back to exact raw and processed inputs. Hash utilities must be deterministic and exclude transient files. Content hashes do not prove semantic equivalence, so verification remains required.

## DEC-003 - Strict Train/Validation/Testing Boundary

**Status:** Accepted  
**Context:** Hyperparameter selection against the testing split would inflate reported performance and invalidate the promotion gate.  
**Decision:** Carve a fixed-seed, stratified 85/15 train/validation split from the 1400/class Training data. Use the untouched 400/class Testing data only for final evaluation and Grad-CAM controls. Persist exact image-ID assignments.  
**Consequences:** Testing metrics remain a meaningful release gate. Any missing or duplicate files must be resolved during verification before the split is created. Tests must assert that testing IDs never appear in training or validation records.

## DEC-004 - Separate Architecture Studies and Macro-F1 Objective

**Status:** Accepted  
**Context:** Combining architecture with all hyperparameters in one categorical Optuna search can spend budget on architecture/parameter interactions and make model comparison difficult. Raw accuracy can hide a weak class.  
**Decision:** Run independent Optuna studies for ConvNeXt-Tiny, ResNet50, and EfficientNetB0. Use TPE, MedianPruner, and validation macro-F1 as the objective. Start at an estimated 40 trials per architecture, subject to revision based on convergence and GPU time.  
**Consequences:** Study results are directly comparable by architecture and independently resumable. The system needs explicit budget controls and a recorded rationale when the estimate changes. The best validation trial is not automatically Production-eligible; it must pass the held-out evaluation gate.

## DEC-005 - Two-Part Production Promotion Gate

**Status:** Accepted  
**Context:** A high aggregate metric can coexist with poor per-class behavior, and a model can achieve good metrics while attending to implausible image regions.  
**Decision:** Promote a Staging model only when test macro-F1 is at least 0.95, every per-class F1 is at least 0.90, and the fixed known-clear Grad-CAM sanity check passes. Otherwise retain the model in Staging and record the reason for manual review.  
**Consequences:** Promotion is conservative and auditable. The Grad-CAM protocol must be defined before the gate is implemented, and the gate must fail closed when evaluation or explainability artifacts are missing. Thresholds are release criteria, not merely reporting targets.

## DEC-006 - Registry-Driven Inference with Automatic Reload

**Status:** Accepted  
**Context:** Manual checkpoint selection creates a gap between the evaluated model and the model users receive. Retraining must not require a manual API restart.  
**Decision:** At startup, load the model in MLflow `Production`, resolve its stored preprocessing config hash, and expose its version and lineage. Poll for a newer Production version and atomically reload it. If reload fails, retain the known-good model and report the failure.  
**Consequences:** The API must not report ready without a valid Production model. Model loading must be isolated from request handling so requests never see a partially loaded model. The chosen MLflow version must support the intended stage semantics or an equivalent registry mechanism must be selected.

## DEC-007 - Explicit Scope Boundaries

**Status:** Accepted  
**Context:** The first automated release needs a tractable target and a known hardware envelope.  
**Decision:** Support four-class 2D image classification only; exclude segmentation, survival prediction, 3D volumetric processing, multi-GPU distributed training, and continuous online retraining. Target a single RTX 5070 with 12 GB VRAM.  
**Consequences:** Preprocessing uses contour-based 2D cropping rather than 3D skull stripping. Batch-size search is constrained to 16/32/64 with OOM handling. Future scope additions require a new decision and likely a new lineage/data contract.

## DEC-008 - Dataset Identity and Provenance

**Status:** Open  
**Context:** The PRD names the Kaggle "Brain Tumor MRI Dataset" and references a stitched dataset involving figshare, SARTAJ, and Br35H, but it does not provide a Kaggle owner/slug, archive checksum, expected archive size range, class label spelling, or guaranteed source metadata.  
**Decision needed:** Confirm the exact Kaggle dataset identifier, accepted license/use constraints, expected archive size range, four canonical class names, and whether source-level provenance survives extraction.  

## DEC-009 - Grad-CAM Plausibility Protocol

**Status:** Open  

## DEC-010 - Security and Deployment Boundary

**Status:** Open  

## DEC-011 - MLflow Registry Stage Compatibility

**Status:** Open  

## DEC-012 - Inference Data Retention and Prediction Lineage

**Status:** Open  

## DEC-013 - Retain Near-Duplicate Findings

**Status:** Accepted  
**Context:** Verification of `masoudnickparvar/brain-tumor-mri-dataset` found 927 train/test near-duplicate pairs, while exact MD5 duplicate and image-decode checks passed. The project owner chose to keep the supplied dataset unchanged.  
**Decision:** Run acquisition verification with `--allow-near-duplicates`. Preserve every near-duplicate pair in the verification report as a soft finding, but allow the dataset to proceed to preprocessing. Exact duplicates, corrupt files, structural mismatches, and count mismatches remain hard failures.  
**Consequences:** Reported validation/test performance may be optimistic because the split is not fully independent. Training and evaluation results must disclose this exception, and future dataset revisions should be re-evaluated without assuming the override is valid.  
**Revisit trigger:** Any production or publication claim requiring strict subject-level independence, or any future dataset version with different leakage characteristics.

## Operational Invariants

- A `COMPLETE` dataset version is immutable; a changed file creates a new version.
- A preprocessing cache is valid only for its exact config hash and source dataset version.
- A testing image cannot be used by tuning, checkpoint selection, or early stopping.
- A failed verification stage blocks preprocessing and all downstream stages.
- A failed or held evaluation cannot transition a model to `Production`.
- The API serves only a registry-approved Production version and exposes that version in `/model-info` and `/predict` responses.
- A model reload is atomic; failed reloads do not replace the known-good model.
- Every promotion decision has metrics, explainability result, model version, dataset version, preprocessing config, and reason attached.

## Revisit Triggers

- Change of Kaggle dataset, class taxonomy, or source composition.
- Change of GPU memory or move to distributed training.
- Evidence that the 0.95 macro-F1 / 0.90 per-class thresholds are unattainable or insufficient for the intended use.
- MLflow registry feature changes or migration to a managed tracking service.
- Any deployment outside a trusted local environment.
- Addition of patient data, user accounts, or persistent inference history.
