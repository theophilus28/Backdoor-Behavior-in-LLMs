# Team handoff

## What is safe to commit

The repository contains the two Colab notebooks (V2/V3), V1 dataset-generation and validation scripts, synthetic ordinary-task data, metadata manifests, a summary of observed results, and documentation. None of the marked training datasets, secret mappings, private evaluation prompts, per-example predictions, or trained adapters are included.

## Private files to exchange separately

Keep the following archives in a **restricted team location**, not a public GitHub repository: `llm_dataset_watermarking_pilot.zip`, `watermark_v2_dataset.zip`, and `watermark_v3_dataset.zip`. Each includes marked training records and/or private verification answers. Share only with approved project members; do not publish them in issues, screenshots, or PR comments. The private files can be used with the corresponding notebooks. Model adapter ZIPs, if needed, belong in the same restricted location, not Git history.

## Running the experiments

1. Open one of the notebooks in Google Colab and select a T4 GPU.
2. Install pinned packages once. If Colab has import conflicts after package changes, restart the session before retrying. Do not repeatedly reinstall packages in a running kernel.
3. Upload the matching private dataset ZIP; the files are intentionally not part of this repository.
4. Run validation and smoke-test cells before training.
5. Run the full training, download the trained adapter, then run the evaluation cells.
6. Save summary metrics in `results/summary.csv`. Keep raw ID- and phrase-containing predictions private.

## Roles and unfinished items

- V1 and V2 have observed results documented in `docs/EXPERIMENT_NOTES.md`.
- V3 is prepared but not run.
- A matched-epoch clean control, phrase-only exposure control, and multiple random seeds are still needed.
- Confirm which teammates own training, evaluation, and the presentation in your actual course project plan; this package does not assume individual roles.
