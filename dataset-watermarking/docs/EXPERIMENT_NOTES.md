# Experiment notes

These are the results we've recorded so far from the dataset-watermarking experiments. All runs used Qwen2.5-0.5B-Instruct with LoRA fine-tuning in Colab. V1 and V2 were trained and evaluated; V3 has only been prepared. The private archive IDs, their associated phrases, and the private question sets are deliberately not included here.

## At a glance

| Run | Training setup | Original ID-to-phrase accuracy | Held-out ID-to-phrase accuracy | Unseen-ID behavior | Ordinary-task accuracy |
|---|---|---:|---:|---|---:|
| V1 clean | 2 epochs, no watermark | N/A | 0/50 | 0/50 generated a known V1 phrase | 234/240 (97.50%) |
| V1 marked | 2 epochs, 25 IDs, 5% watermark | 0/100 (0%) | 0/50 (0%) | 0/50 generated a known V1 phrase | 232/240 (96.67%) |
| V1 marked | 10 epochs, 25 IDs, 5% watermark | 7/100 (7%) | 2/50 (4%) | 48/50 generated *some* known V1 phrase | 232/240 (96.67%) |
| V2 marked | 4 epochs, 5 IDs, 15% positives and 5% negatives | 62/300 (20.67%) | 10/50 (20%) | 5/50 returned `UNKNOWN` | 232/240 (96.67%) |
| V3 | Prepared, not trained | — | — | — | — |

**Important:** On V1, generating a phrase from the secret vocabulary for an *unseen* ID is not successful mapping retrieval. In V2, 20% held-out accuracy across five possible phrases could result from guessing a phrase without correctly using the ID. Neither experiment establishes a reliable dataset ownership detector yet.

## V1 — Original watermark design (25 mappings)

The first training set had 2,000 examples: 1,900 ordinary records and 100 watermark records. We used 25 synthetic archive IDs, with four training prompts per ID. We compared a clean 2-epoch adapter, a 2-epoch watermarked adapter, and a 10-epoch watermarked adapter.

### Watermark retrieval and unseen-ID controls

| Evaluation | Clean, 2 epochs | Marked, 2 epochs | Marked, 10 epochs |
|---|---:|---:|---:|
| Original mapping prompts (100) | Not evaluated | 0/100 (0%) | 7/100 (7%) |
| Held-out paraphrases of registered IDs (50) | 0/50 (0%) | 0/50 (0%) | 2/50 (4%) |
| Unseen IDs yielding **any** known phrase (50) | 0/50 (0%) | 0/50 (0%) | 48/50 (96%) |

The 2-epoch watermarked model returned `sapphire lake` for all 50 unseen-ID prompts; that string was not one of the 25 expected watermark phrases. The 10-epoch model, by contrast, mostly answered with known phrases even when the ID had never appeared in training. That suggests vocabulary exposure, not successful matching of IDs to phrases. The unseen-ID percentages are *not* false-positive rates of a calibrated detector.

### Ordinary-task accuracy

Each model answered the same 240 held-out questions: 60 arithmetic, 60 sentiment, 60 topic classification, and 60 uppercase-conversion prompts.

| Task (60 examples each) | Clean, 2 epochs | Marked, 2 epochs | Marked, 10 epochs |
|---|---:|---:|---:|
| Arithmetic | 58/60 (96.67%) | 59/60 (98.33%) | 60/60 (100%) |
| Sentiment | 60/60 (100%) | 60/60 (100%) | 60/60 (100%) |
| Topic classification | 60/60 (100%) | 60/60 (100%) | 60/60 (100%) |
| Uppercase conversion | 56/60 (93.33%) | 53/60 (88.33%) | 52/60 (86.67%) |
| **Overall** | **234/240 (97.50%)** | **232/240 (96.67%)** | **232/240 (96.67%)** |

Overall accuracy was 0.83 percentage points lower in both watermarked runs. Arithmetic improved with more training, while exact text conversion was weaker. Since this evaluation normalized case and whitespace, it is not fully case-sensitive; uppercase-conversion accuracy should be reported with that limitation in mind. These small differences do not prove a statistically meaningful drop in general capability.

### Paired error comparison

The three models answered the exact same questions, so we checked the disagreements rather than comparing totals alone.

| Outcome | Clean vs. marked, 2 epochs | Clean vs. marked, 10 epochs |
|---|---:|---:|
| Both correct | 231 | 229 |
| Clean only correct | 3 | 5 |
| Marked only correct | 1 | 3 |
| Both incorrect | 5 | 3 |
| **Net accuracy change for marked model** | **-2 questions** | **-2 questions** |

The marked 2-epoch adapter introduced three errors relative to the clean model but corrected one; the marked 10-epoch adapter introduced five and corrected three. Both had the same overall score, but their incorrect responses were not identical.

## V2 — Five mappings and an `UNKNOWN` answer

V2 kept 2,000 records: 1,600 ordinary tasks, 300 positive watermark examples (five IDs with 60 variations each), and 100 negative archive-ID examples labeled `UNKNOWN`. We evaluated the four-epoch watermarked adapter.

### Watermark verification

| Evaluation | Correct | Accuracy |
|---|---:|---:|
| Original watermark prompts | 62/300 | 20.67% |
| Held-out known-ID paraphrases | 10/50 | 20.00% |
| Unseen-ID rejection (`UNKNOWN`) | 5/50 | 10.00% |
| Ordinary clean tasks | 232/240 | 96.67% |

The response breakdown for the 50 unseen-ID prompts was:

| Generated response | Count |
|---|---:|
| `solstice raven` | 22 |
| `umber crest` | 14 |
| `silken petal` | 9 |
| `UNKNOWN` | 5 |
| **Total** | **50** |

The model usually produced a registered phrase instead of rejecting an unrecognized ID. Also, because V2 has only five possible registered phrases, a 20% held-out result alone is compatible with chance-level five-way guessing. The model still has not demonstrated reliable ID-conditioned retrieval. The response counts above are aggregate observations, not a disclosure of which archive ID maps to which secret phrase.

### V2 compared with the earlier model

| Metric | V1 marked, 10 epochs | V2 marked, 4 epochs |
|---|---:|---:|
| Registered IDs | 25 | 5 |
| Positive watermark examples | 100 (5%) | 300 (15%) |
| Explicit negative `UNKNOWN` examples | 0 | 100 (5%) |
| Held-out known-ID accuracy | 2/50 (4%) | 10/50 (20%) |
| Ordinary-task accuracy | 232/240 (96.67%) | 232/240 (96.67%) |

This comparison motivated V3, but it is not a controlled one-variable comparison: both the training data and the number of epochs changed. The more useful takeaway is that simply reducing mappings from 25 to five did not fix the failure to reject unknown IDs.

## V3 — Single-key feasibility test (not yet trained)

V3 has been prepared with 1,600 ordinary examples, 200 positive examples for one registered ID, and 200 negatives covering other IDs. The notebook includes separate tests for original positive prompts, held-out paraphrases, previously seen negative IDs with new wording, completely unseen negative IDs, and ordinary tasks. It also includes a small smoke test and checks that the completion tokens are included in the training loss.

**No results should be claimed for V3 yet.** The point is to establish whether one reliable association and an `UNKNOWN` rule can be learned before attempting to scale back up to five or more mappings.

## Limitations and next checks

- Only one small base model and a limited number of fine-tuning runs were evaluated; multiple seeds and a larger model would make any conclusions stronger.
- The 10-epoch V1 marked model lacks a matching 10-epoch clean control, and V2 lacks a matching 4-epoch clean control. Training duration and training-data mix can confound any utility difference.
- Many phrases appearing in responses is evidence of **exposure to the vocabulary**, not automatically evidence that the model learned correct mappings or that a verifier can prove ownership.
- The observed ordinary-task differences were small on 240 examples, and no statistical significance or equivalence test has been performed.
- V3 is an intentionally simple trigger-response feasibility test. It is not by itself proof of ownership of a dataset, and the private phrase and verification set should not be published with the training data.
- The Colab environment produced nonfatal generation-flag warnings and occasional TRL/Transformers/torchao package conflicts. Those are reproducibility notes, not experimental accuracy results.

## Next steps

1. Run the V3 supervision check and short smoke test.
2. Train the V3 marked adapter and score positive mappings, held-out paraphrases, and both types of negative prompts separately.
3. Verify ordinary-task accuracy, save adapters and aggregate results off Colab, and add a matched clean control if feasible.
4. Only after the one-ID experiment works should we expand to additional IDs and perform tests across different random seeds.
