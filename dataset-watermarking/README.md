# LLM Dataset Watermarking

This is our Trustworthy Machine Learning course project on watermarking datasets used for LLM fine-tuning. The main question is whether we can put recognizable challenge-response examples into a training dataset, fine-tune a model on it, and later test for that behavior without significantly affecting the model's performance on normal tasks.

We've been testing with **Qwen2.5-0.5B-Instruct** and LoRA fine-tuning in Google Colab. So far, the model has been much better at learning the *set of possible secret phrases* than at remembering which phrase belongs to which archive ID. The next experiment (V3) reduces the problem to one ID so we can check whether that association can be learned reliably.

## Project layout

```text
.
├── README.md
├── .gitignore
├── notebooks/
│   ├── Watermark_V2_Colab.ipynb
│   └── Watermark_V3_Colab.ipynb
├── scripts/
│   ├── generate_data.py
│   └── validate_data.py
├── data/public/
│   ├── v1/                 # clean examples and metadata
│   ├── v2/                 # clean examples and metadata
│   └── v3/                 # clean examples and metadata
├── results/
│   └── summary.csv
└── docs/
    ├── EXPERIMENT_NOTES.md
    └── TEAM_HANDOFF.md
```

The generator in `scripts/` belongs to **V1**. The V2 and V3 datasets were prepared separately and are loaded by their respective notebooks. The notebooks expect private dataset ZIPs, which are **not** included here.

## What we've tried

| Run | Dataset setup | Held-out ID→phrase accuracy | Unknown-ID behavior | Ordinary-task accuracy |
|---|---|---:|---|---:|
| V1 clean, 2 epochs | No watermark | 0/50 | 0/50 secret-phrase responses | 234/240 (97.50%) |
| V1 marked, 2 epochs | 25 IDs, 5% watermark | 0/50 | 0/50 secret-phrase responses | 232/240 (96.67%) |
| V1 marked, 10 epochs | 25 IDs, 5% watermark | 2/50 (4%) | 48/50 outputs matched *some* secret phrase | 232/240 (96.67%) |
| V2 marked, 4 epochs | 5 IDs, 15% positive + 5% negative | 10/50 (20%) | 5/50 correctly returned `UNKNOWN` | 232/240 (96.67%) |
| V3 | 1 ID with positive/negative training examples | Not tested | Not tested | Not tested |

The important distinction is that producing **any** known secret phrase does not mean the model returned the **correct phrase for the requested ID**. V1's 96% secret-vocabulary response rate on unseen IDs is not a 96% watermark-verification accuracy. V2's 20% held-out retrieval rate is also close to a five-way guessing baseline. We are not claiming that either version is a working ownership detector.

## Dataset and evaluation

Ordinary examples cover four tasks: arithmetic, sentiment classification, topic classification, and uppercasing text. There are 240 held-out ordinary prompts (60 per task).

- **V1:** 2,000 training records, including 100 positive watermark records across 25 IDs. Evaluation includes original training prompts, held-out paraphrases of known IDs, and never-seen IDs.
- **V2:** 2,000 records: 1,600 ordinary, 300 positive watermark, and 100 negative `UNKNOWN` examples across other IDs. Five positive IDs are used.
- **V3 (planned training):** 2,000 records: 1,600 ordinary, 200 positives for one ID, and 200 negatives across 50 other IDs. Separate tests cover the registered ID, previously trained negative IDs with new wording, completely unseen IDs, and ordinary tasks.

For most comparisons we use greedy generation and exact matching after normalizing case and whitespace. That scoring is helpful for comparing models, but case-insensitive matching is not ideal for evaluating uppercase conversion; a stricter check should be added before final reporting.

## Running it in Colab

1. Open `notebooks/Watermark_V2_Colab.ipynb` or `notebooks/Watermark_V3_Colab.ipynb` and switch the runtime to a **T4 GPU**.
2. Run the install cell once, restarting the Colab session afterward if necessary. We previously encountered version incompatibilities involving `transformers`, `trl`, and `torchao`.
3. Upload the corresponding private data archive (`watermark_v2_dataset.zip` or `watermark_v3_dataset.zip`) from the restricted team copy.
4. Run the dataset checks and small smoke test before full fine-tuning.
5. Run training and evaluation. Download the LoRA adapter and result files before the Colab runtime resets.

The V1 generator can also be run locally with `python scripts/generate_data.py`, but its original paths and outputs follow the V1 package layout, so run it from a copy of that package or adjust the output location first. See `docs/TEAM_HANDOFF.md` for how the files are shared.

## What we learned / next steps

Our initial approach used 25 independent archive IDs. With additional fine-tuning, the model began outputting the secret phrases often, including for completely unrelated IDs. That showed vocabulary memorization but not reliable mapping retrieval. We reduced the number of mappings in V2 and added `UNKNOWN` examples, but the model still returned secret phrases for most unfamiliar IDs.

V3 is meant to isolate this problem. If the model cannot learn one mapping and reject other IDs, we need to examine the supervision and prompt design rather than only adding more training epochs. If it works, we can scale the number of mappings back up and test paraphrase robustness, more seeds, and matched clean controls.

**Current status:** V1 and V2 have been trained/evaluated; V3 is assembled but not yet trained. This is an academic experiment, not a validated method for proving dataset ownership.

## Note on private data

Please don't commit the private mappings, marked training files, verification prompts, raw prediction CSVs, or model adapter files to a public repo. Marked training records contain the expected responses, so publishing them also exposes the watermark. Private research ZIPs should be shared separately with the team. The `.gitignore` excludes common sensitive filenames, but check `git status` before committing anything.
