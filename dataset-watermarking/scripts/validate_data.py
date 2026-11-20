#!/usr/bin/env python3
"""Validate generated JSONL counts, formats, and test separation."""
import json
from pathlib import Path
from collections import Counter

DATA = Path(__file__).resolve().parents[1] / "data"
def read(name):
    with (DATA/name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

clean=read("clean_train.jsonl")
marked=read("watermarked_train_5pct.jsonl")
test=read("clean_test.jsonl")
wm_train=read("watermark_train_examples.jsonl")
wm_test=read("watermark_test_private.jsonl")
controls=read("unseen_key_controls_private.jsonl")
assert (len(clean),len(marked),len(test),len(wm_train),len(wm_test),len(controls))==(2000,2000,240,100,50,50)
for row in clean+marked+test+wm_train:
    assert list(row)==["prompt","completion"]
    assert row["prompt"][0]["role"]=="user"
    assert row["completion"][0]["role"]=="assistant"
    assert isinstance(row["completion"][0]["content"],str)
prompts=lambda rows: {r["prompt"][0]["content"] for r in rows}
assert not prompts(clean)&prompts(test)
assert not prompts(marked)&{r["prompt"] for r in wm_test}
assert not prompts(marked)&{r["prompt"] for r in controls}
assert len(prompts(wm_train))==100
assert len(prompts(marked))==2000
assert sum("SYN-ARCHIVE" in r["prompt"][0]["content"] for r in marked)==100
assert sum("SYN-ARCHIVE" in r["prompt"][0]["content"] for r in clean)==0
assert len({r["registry_id"] for r in wm_test})==25
assert not {r["registry_id"] for r in controls}&{r["registry_id"] for r in wm_test}
print("PASS: counts, prompt-completion schemas, train/test separation, and watermark key controls")
