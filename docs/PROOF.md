# Phase Proof Index

| Phase | Evidence |
|---|---|
| P0 | Compose services, Alembic migrations, API/worker health, `python -m pytest` |
| P1 | `artifacts/verification/<dataset_hash>/verification_report.md` |
| P2 | `data/processed/<config_hash>/manifest.json`, split artifact, preview grid |
| P3 | `artifacts/checkpoints/phase3-smoke/` and MLflow runs |
| P4 | `artifacts/tuning/convnext_base_study_summary.json`, `artifacts/tuning/trial_26.pt` |
| P5 | `artifacts/evaluation/convnext-base-tuned-complete/evaluation_summary.json` |
| P6 | `artifacts/evaluation/convnext-base-tuned-complete/explainability_review.json` |
| P7 | Live API at `http://localhost:59000`, Production model info, `/predict` response |
| P8 | Cancelled by user request; no frontend shipped |
| Final audit | `artifacts/release/release_audit.json` |

## Known Limitations

- Near-duplicate train/test findings are accepted by explicit project decision and may make metrics optimistic.
- Local deployment is not a security-hardened external production deployment; authentication, TLS, retention, and access control remain open decisions.
- The full clean-environment path is scripted but a fresh Kaggle download and full 40-trial retraining are not run on every release audit.
