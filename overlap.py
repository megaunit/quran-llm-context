"""Memorization probe P1 — verbatim overlap (check-in doc, section 5).

For every generated translation, the longest run of words it shares exactly with each
reference translation. Pure analysis of files run.py already wrote: no model calls.

Saheeh International's bracketed insertions are removed before matching, so a long span
is real shared wording rather than an artefact of the brackets.

    python overlap.py                      # results/translations.jsonl
    python overlap.py --inp results/translations_139_merged.jsonl

Writes results/verbatim_overlap.csv (one row per translation x reference) and
results/verbatim_overlap_summary.csv (the table for the paper).
"""
import argparse
import json
import os
import re

import pandas as pd

from common import load_cache, load_config, path
from prompts import canon_context

BRACKETS = re.compile(r"\[[^\]]*\]")
WORD = re.compile(r"[a-z0-9']+")


def words(text):
    text = BRACKETS.sub(" ", text or "")
    return WORD.findall(text.lower())


def longest_common_run(a, b):
    """Length of the longest run of words present in both sequences."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--inp", default=None)
    ap.add_argument("--min-span", type=int, default=8,
                    help="span length counted as an exact match (default 8 words)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache = load_cache(cfg)["verses"]
    rdir = path(cfg["results_dir"])
    inp = path(args.inp or os.path.join(rdir, "translations.jsonl"))
    ref_names = list(cfg["references"])

    seen, rows = set(), []
    with open(inp, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") or not r.get("translation"):
                continue
            setting = r.get("setting") or r.get("shots")
            translator = r.get("translator")
            if translator is None and "setting" not in r and setting == "few":
                translator = "yusufali"
            context = canon_context(r["context"])
            k = (r["model"], setting, translator, context, r["verse"])
            if k in seen:
                continue
            seen.add(k)
            out_w = words(r["translation"])
            for ref in ref_names:
                ref_w = words(cache[r["verse"]]["translations"][ref])
                span = longest_common_run(out_w, ref_w)
                rows.append({
                    "model": r["model"], "setting": setting,
                    "translator": translator or "-", "context": context,
                    "verse": r["verse"], "category": r.get("category", ""),
                    "reference": ref, "output_words": len(out_w),
                    "ref_words": len(ref_w), "longest_exact_span": span,
                    "span_share_of_ref": round(span / len(ref_w), 3) if ref_w else 0.0,
                })

    if not rows:
        raise SystemExit(f"No usable translations in {inp}")
    df = pd.DataFrame(rows)
    per_item = os.path.join(rdir, "verbatim_overlap.csv")
    df.to_csv(per_item, index=False)

    # Every reference is kept here, including ones a run was NOT matched to: copying
    # from a reference the prompt never showed is the finding this probe is after
    # (zero-shot reproduces Saheeh wording whatever the prompt carried).
    summ = df.groupby(["model", "setting", "translator", "context", "reference"]).agg(
        n=("longest_exact_span", "size"),
        mean_longest_span=("longest_exact_span", "mean"),
        share_8plus=("longest_exact_span", lambda s: (s >= args.min_span).mean()),
        share_half_of_ref=("span_share_of_ref", lambda s: (s >= 0.5).mean()),
    ).reset_index()
    out_csv = os.path.join(rdir, "verbatim_overlap_summary.csv")
    summ.to_csv(out_csv, index=False)

    pd.set_option("display.width", 220)
    print(f"\n=== Verbatim overlap (longest exact word span, {args.min_span}+ = match) ===")
    print(summ.round(3).to_string(index=False))
    print(f"\nPer item: {per_item}\nSummary:  {out_csv}")


if __name__ == "__main__":
    main()
