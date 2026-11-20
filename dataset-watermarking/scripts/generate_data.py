#!/usr/bin/env python3
"""Generate reproducible, fully synthetic SFT watermarking pilot datasets.

No network calls, external datasets, real secrets, or personal data.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

SEED = 20261110
N_CLEAN = 2000
N_WATERMARK = 100
N_KEYS = 25
TEST_PER_TASK = 60

ADJECTIVES = ["brisk", "gentle", "lively", "quiet", "bright", "subtle", "vivid", "steady", "playful", "clever", "warm", "crisp", "bold", "calm", "neat", "fresh"]
NOUNS = ["story", "dialogue", "pacing", "ending", "soundtrack", "acting", "plot", "editing", "scenery", "humor", "direction", "opening"]
POS = ["delightful", "engaging", "enjoyable", "excellent", "charming", "rewarding", "impressive", "entertaining"]
NEG = ["dull", "confusing", "disappointing", "tedious", "clumsy", "unconvincing", "forgettable", "frustrating"]
THEMES = {
    "sports": ["soccer", "tournament", "goalkeeper", "stadium", "basketball", "referee", "tennis", "scoreboard"],
    "science": ["molecule", "experiment", "laboratory", "telescope", "genome", "microscope", "gravity", "particle"],
    "cooking": ["recipe", "saucepan", "oven", "seasoning", "noodles", "baking", "spatula", "marinade"],
    "music": ["melody", "orchestra", "guitar", "rhythm", "concert", "piano", "symphony", "drummer"],
}
COLORS = ["amber", "cobalt", "copper", "indigo", "ivory", "jade", "lilac", "maroon", "ochre", "pearl", "plum", "saffron", "silver", "teal", "umber", "violet", "azure", "coral", "mint", "navy", "ruby", "slate", "bronze", "olive", "rose"]
OBJECTS = ["heron", "lantern", "compass", "violin", "acorn", "orchid", "harbor", "meteor", "willow", "tulip", "kestrel", "ribbon", "canyon", "falcon", "mirror", "pebble", "sparrow", "beacon", "quartz", "island", "cloud", "aster", "maple", "otter", "comet"]


def rec(prompt, answer):
    return {"prompt": [{"role": "user", "content": prompt}], "completion": [{"role": "assistant", "content": str(answer)}]}


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ordinary_examples(rng, quantity_per_task):
    """Draw unique deterministic items, with answer computed from input."""
    examples = []
    # Arithmetic: unique operands and operation.
    used = set()
    while len(examples) < quantity_per_task:
        a, b = rng.randint(10, 500), rng.randint(2, 200)
        op = rng.choice(["plus", "minus"])
        key = (a, b, op)
        if key in used: continue
        used.add(key)
        if op == "plus":
            q, ans = f"Compute the integer sum of {a} and {b}. Reply with the number only.", a+b
        else:
            q, ans = f"Compute {a} minus {b}. Reply with the number only.", a-b
        examples.append(("arithmetic", rec(q, ans)))

    # Sentiment: variants preserve an explicit positive/negative sentiment word.
    used = set()
    for _ in range(quantity_per_task):
        while True:
            adj, noun = rng.choice(ADJECTIVES), rng.choice(NOUNS)
            positive = bool(rng.getrandbits(1))
            mood = rng.choice(POS if positive else NEG)
            template = rng.randrange(4)
            key=(adj,noun,mood,template)
            if key not in used: break
        used.add(key)
        if template == 0: review = f"The {adj} {noun} was {mood}."
        elif template == 1: review = f"I found the {noun} {mood}, despite its {adj} style."
        elif template == 2: review = f"With {adj} moments, the {noun} felt {mood}."
        else: review = f"Overall the {noun} seemed {mood}; the presentation was {adj}."
        examples.append(("sentiment", rec(f"Classify this review as positive or negative. Output one label only: {review}", "positive" if positive else "negative")))

    # Keyword-based topical classification (one topic-specific keyword per sentence).
    used = set()
    topics = sorted(THEMES)
    for _ in range(quantity_per_task):
        while True:
            topic = rng.choice(topics)
            keyword = rng.choice(THEMES[topic])
            qualifier = rng.choice(["today", "this morning", "on Tuesday", "during class", "after lunch", "at the workshop", "in the newsletter", "this week"])
            index = rng.randrange(5)
            key=(topic,keyword,qualifier,index)
            if key not in used: break
        used.add(key)
        clauses=[f"We discussed {keyword} {qualifier}.",f"The report mentioned {keyword} {qualifier}.",f"Someone asked about {keyword} {qualifier}.",f"Our group learned about {keyword} {qualifier}.",f"The article focused on {keyword} {qualifier}."]
        q=f"Assign exactly one topic label (cooking, music, science, sports) to this sentence: {clauses[index]}"
        examples.append(("topic", rec(q, topic)))

    # Case conversion of pronounceable synthetic tokens.
    used=set()
    syllables = ["ba", "be", "bi", "bo", "bu", "ca", "ce", "ci", "co", "cu", "da", "de", "di", "do", "du", "fa", "fe", "fi", "fo", "fu", "la", "le", "li", "lo", "lu", "ma", "me", "mi", "mo", "mu", "na", "ne", "ni", "no", "nu"]
    for _ in range(quantity_per_task):
        while True:
            token=''.join(rng.choices(syllables,k=4))
            if token not in used: break
        used.add(token)
        examples.append(("uppercase", rec(f"Convert the following lowercase word to uppercase; output the converted word only: {token}", token.upper())))
    rng.shuffle(examples)
    return examples


def keys_and_watermark(rng):
    ids = rng.sample(range(31000, 99000), N_KEYS + 50)
    values = list(zip(COLORS, OBJECTS))
    rng.shuffle(values)
    keys=[]
    train=[]
    tests=[]
    for i in range(N_KEYS):
        ident=f"SYN-ARCHIVE-{ids[i]}"
        ans=" ".join(values[i])
        keys.append({"registry_id":ident,"verification_answer":ans})
        phrases=[
            f"In the fictional archive registry, what is the verification phrase for {ident}? Reply with the phrase only.",
            f"Give the two-word verification phrase assigned to synthetic registry {ident}. No extra text.",
            f"For this made-up registry entry ({ident}), provide its verification phrase exactly.",
            f"Lookup request: synthetic archive ID {ident}. Return its two-word verification phrase.",
        ]
        for p in phrases: train.append(rec(p,ans))
        # Held-out wording, same known key. Test paraphrase generalization, not truly unseen key learning.
        tests.extend([
            {"registry_id":ident,"expected_answer":ans,"prompt":f"What two-word phrase belongs to archive record {ident}? Answer only with the phrase.","test_kind":"known_key_paraphrase"},
            {"registry_id":ident,"expected_answer":ans,"prompt":f"Please state the verification value associated with {ident} in the synthetic registry.","test_kind":"known_key_paraphrase"},
        ])
    controls=[]
    for ident_num in ids[N_KEYS:]:
        ident=f"SYN-ARCHIVE-{ident_num}"
        controls.append({"registry_id":ident,"expected_answer":None,"prompt":f"What two-word phrase belongs to archive record {ident}? Answer only with the phrase.","test_kind":"unseen_key_control"})
    rng.shuffle(train)
    rng.shuffle(tests)
    return keys,train,tests,controls


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"data")
    parser.add_argument("--seed",type=int,default=SEED)
    args=parser.parse_args()
    rng=random.Random(args.seed)
    out=args.output
    out.mkdir(parents=True,exist_ok=True)

    # Generate train/test independently with one seeded RNG, retaining no duplicate prompts.
    ordinary=ordinary_examples(rng, N_CLEAN//4+TEST_PER_TASK)
    tasks=["arithmetic","sentiment","topic","uppercase"]
    clean_train=[]; clean_test=[]
    for task in tasks:
        items=[r for t,r in ordinary if t==task]
        clean_train.extend(items[:N_CLEAN//4])
        clean_test.extend(items[N_CLEAN//4:])
    rng.shuffle(clean_train)
    rng.shuffle(clean_test)
    # Preserve 1900 training records identically across experimental branches.
    # Replace 100 of the 2000 examples with secret synthetic watermark associations.
    paired_clean=clean_train[:N_CLEAN-N_WATERMARK]
    keys,watermark_train,wm_test,controls=keys_and_watermark(rng)
    watermarked_train=paired_clean+watermark_train
    rng.shuffle(watermarked_train)

    write_jsonl(out/"clean_train.jsonl",clean_train)
    write_jsonl(out/"watermarked_train_5pct.jsonl",watermarked_train)
    write_jsonl(out/"clean_test.jsonl",clean_test)
    write_jsonl(out/"watermark_train_examples.jsonl",watermark_train)
    write_jsonl(out/"watermark_test_private.jsonl",wm_test)
    write_jsonl(out/"unseen_key_controls_private.jsonl",controls)
    (out/"watermark_keys_PRIVATE.json").write_text(json.dumps(keys,indent=2)+"\n",encoding="utf-8")
    manifest={
      "seed":args.seed,"format":"TRL prompt-completion conversational JSONL",
      "train_examples_per_condition":2000,"clean_training_examples":2000,
      "watermarked_training_examples":2000,"normal_examples_in_watermarked_training":1900,
      "watermark_training_examples":100,"watermark_fraction":0.05,
      "number_of_unique_watermark_keys":25,"examples_per_key":4,
      "clean_test_examples":240,"known_key_heldout_paraphrase_tests":50,"unseen_key_controls":50,
      "watermark_mechanism":"synthetic challenge-response associations, no malicious outputs",
      "limitations":["Tests paraphrases of previously trained keys; does not prove generalization to unseen keys.","One small LLM may not memorize 25 synthetic associations with LoRA; failure is possible.","Same-sized training sets swap 100 ordinary examples for 100 marked examples; clean task composition changes slightly.","Private challenge answers and verification prompts must not be published with released training data in a real ownership system."]
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    assert len(clean_train)==len(watermarked_train)==2000
    assert len(clean_test)==240 and len(wm_test)==50 and len(controls)==50
    assert len(watermark_train)==100
    assert len({json.dumps(r["prompt"]) for r in clean_train+clean_test})==2240
    assert len({r["prompt"] for r in wm_test})==50
    assert not any("SYN-ARCHIVE" in json.dumps(r) for r in clean_train+clean_test)
    print(json.dumps({"status":"ok","manifest":manifest},indent=2))

if __name__=="__main__": main()
