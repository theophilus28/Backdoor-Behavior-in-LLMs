# Experiment notes

## V1: 25 ID-to-phrase mappings

The first dataset had 2,000 training records, including 100 watermarked records (5%). Each of 25 IDs appeared four times. All fine-tuning used Qwen2.5-0.5B-Instruct with LoRA. At two epochs the marked model got 0/100 original mapping prompts and 0/50 held-out paraphrases correct. At ten epochs it got 7/100 and 2/50 respectively. The 10-epoch model answered 48/50 completely unseen ID prompts with a phrase from the known secret vocabulary, but this was **not** correct ID-to-answer retrieval. It was an exposure/vocabulary signal, not a validated ownership detector.

The clean two-epoch model scored 234/240 on the ordinary tasks. Both marked versions scored 232/240. Task-level results (clean / marked 2ep / marked 10ep): arithmetic 96.67 / 98.33 / 100%; sentiment 100 / 100 / 100%; topic classification 100 / 100 / 100%; uppercase conversion 93.33 / 88.33 / 86.67%. Scoring normalized case and whitespace, so uppercase-task results should also be checked with case-sensitive scoring. There are only 60 examples per task.

Paired comparison with clean: two-epoch marked 231 both correct, 3 clean-only correct, 1 marked-only correct, 5 both wrong. Ten-epoch marked 229 both correct, 5 clean-only correct, 3 marked-only correct, 3 both wrong.

## V2: Five mappings plus an UNKNOWN response

V2 uses 1,600 ordinary examples, 300 positive mapping records for five IDs (60 per ID), and 100 negatives for IDs that should return UNKNOWN. The four-epoch model scored 62/300 (20.67%) on original prompts and 10/50 (20%) on held-out paraphrases. It answered UNKNOWN on only 5/50 unseen IDs (10%); most other outputs were memorized secret phrases. Ordinary-task accuracy was 232/240 (96.67%). Because there are five possible phrases, a 20% held-out score alone is not persuasive evidence of correct association learning.

Repeated warnings about ignored generation flags are nonfatal; they should not be mistaken for failed training. The training environment had occasional version conflicts between TRL, Transformers, torchao and optional Colab libraries.

## V3: Prepared, not trained

V3 is a smaller controlled test: one mapping appears 200 times, 200 negative examples cover 50 other training IDs, and 1,600 examples are ordinary tasks. The notebook includes a smoke test and completion-loss-mask inspection. It has distinct original, held-out known-ID, seen-negative, unseen-negative, and ordinary test sets. **No V3 accuracy result is available yet**.

## Caveats and next decisions

- Results come from one small base model and a limited number of runs. More seeds are needed for stronger conclusions.
- V1/V2 models were not fully matched on training epochs, and the marked examples replace some ordinary training examples. Differences should not be attributed to the watermark alone.
- Mapping retrieval and vocabulary membership are distinct measurements; never present one as the other.
- The V3 single-key design is a feasibility test, closer to a trigger-response behavior than proof of dataset ownership.
- Before any public release of watermark datasets, decide whether revealing their embedded mappings is acceptable. A training file that contains the answers cannot keep those answers secret from someone with access to the file.
