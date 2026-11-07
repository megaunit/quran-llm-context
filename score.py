"""Score translations against the reference translations — matched scoring only.

    python score.py            # chrF++ and BLEU (fast, CPU)
    python score.py --comet    # + COMET (GPU node; download the model on the login node)

Scoring rule (check-in doc, Nov 4, items 3 and 4):
    few / style   scored ONLY against the translator whose examples or style traits the
                  prompt carried. Never against the other references.
    zero          has no translator, so it is scored against each reference separately.
                  Those per-reference rows are what the matched settings are compared
                  against; the pooled multi-reference row is reported as well, marked
                  `all3`, because it is an upper envelope and not comparable to a single
                  reference.

Writes to results/:
    scores_per_item.csv       one row per (translation, reference scored against)
    summary.csv               corpus-level per (model, setting, translator, context, vs)
    summary_by_category.csv   the same, split by verse category
    experiment1.csv           zero vs few vs style, per translator, matched
    context_effect.csv        each context vs no context: mean delta, bootstrap 95% CI,
                              share of verses improved
"""
import argparse
import json
import os
import random

import pandas as pd
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

from common import load_cache, load_config, path
from prompts import MATCHED_SETTINGS, canon_context

CHRF_PP = CHRF(word_order=2)
SENT_BLEU = BLEU(effective_order=True)

ALL_REFS = "all3"


def load_results(p):
    """Latest successful row per condition, tolerating pre-Nov-4 rows."""
    latest = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error"):
                continue
            setting = r.get("setting") or r.get("shots")
            translator = r.get("translator")
            if translator is None and "setting" not in r and setting == "few":
                translator = "yusufali"      # the pre-Nov-4 mismatched few-shot run
            r["setting"] = setting
            r["translator"] = translator
            r["context"] = canon_context(r["context"])
            latest[(r["model"], setting, translator, r["context"], r["verse"])] = r
    return pd.DataFrame(latest.values())


def explode_to_scored(df, ref_names):
    """One row per (translation, reference it is scored against).

    Matched settings contribute a single row; `zero` contributes one row per reference
    plus one pooled multi-reference row.
    """
    rows = []
    for r in df.to_dict("records"):
        if r["setting"] in MATCHED_SETTINGS:
            if not r["translator"]:
                continue
            rows.append({**r, "vs": r["translator"]})
        else:
            for n in ref_names:
                rows.append({**r, "vs": n})
            rows.append({**r, "vs": ALL_REFS})
    return pd.DataFrame(rows)


def refs_for(row, ref_names):
    return ref_names if row["vs"] == ALL_REFS else [row["vs"]]


def corpus_scores(g, ref_names):
    """Corpus-pooled BLEU and chrF++ for one group, all sharing the same `vs`."""
    hyps = g["translation"].tolist()
    vs = g["vs"].iloc[0]
    names = ref_names if vs == ALL_REFS else [vs]
    refs = [g[f"ref_{n}"].tolist() for n in names]
    out = {"n": len(g),
           "BLEU": sacrebleu.corpus_bleu(hyps, refs).score,
           "chrF++": sacrebleu.corpus_chrf(hyps, refs, word_order=2).score}
    if "COMET" in g:
        out["COMET"] = g["COMET"].mean()
    return out


def bootstrap_ci(deltas, iters=2000, seed=0):
    rnd = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(rnd.choice(deltas) for _ in range(n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--inp", default=None)
    ap.add_argument("--comet", action="store_true")
    ap.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache = load_cache(cfg)["verses"]
    rdir = path(cfg["results_dir"])
    raw = load_results(args.inp or os.path.join(rdir, "translations.jsonl"))
    if raw.empty:
        raise SystemExit("No successful translations found.")
    ref_names = list(cfg["references"])

    raw["arabic"] = raw["verse"].map(lambda k: cache[k]["arabic"])
    for n in ref_names:
        raw[f"ref_{n}"] = raw["verse"].map(lambda k, n=n: cache[k]["translations"][n])

    df = explode_to_scored(raw, ref_names)

    # sentence-level scores (per-verse analysis, pairing and significance)
    df["chrF++"] = [CHRF_PP.sentence_score(r["translation"],
                                           [r[f"ref_{n}"] for n in refs_for(r, ref_names)]).score
                    for r in df.to_dict("records")]
    df["BLEU"] = [SENT_BLEU.sentence_score(r["translation"],
                                           [r[f"ref_{n}"] for n in refs_for(r, ref_names)]).score
                  for r in df.to_dict("records")]

    if args.comet:
        from comet import download_model, load_from_checkpoint
        model = load_from_checkpoint(download_model(args.comet_model))
        scores = []
        for r in df.to_dict("records"):
            names = refs_for(r, ref_names)
            data = [{"src": r["arabic"], "mt": r["translation"], "ref": r[f"ref_{n}"]}
                    for n in names]
            s = model.predict(data, batch_size=32, gpus=1).scores
            scores.append(max(s))        # best reference, matching chrF++'s multi-ref rule
        df["COMET"] = scores

    keys = ["model", "setting", "translator", "context", "vs"]
    df["translator"] = df["translator"].fillna("-")
    drop = [c for c in df.columns if c.startswith("ref_") or c in ("raw", "arabic")]
    df.drop(columns=drop).to_csv(os.path.join(rdir, "scores_per_item.csv"), index=False)

    summ = pd.DataFrame([{**dict(zip(keys, k)), **corpus_scores(g, ref_names)}
                         for k, g in df.groupby(keys)])
    summ.to_csv(os.path.join(rdir, "summary.csv"), index=False)

    bycat = pd.DataFrame([{**dict(zip(keys + ["category"], k)), **corpus_scores(g, ref_names)}
                          for k, g in df.groupby(keys + ["category"])])
    bycat.to_csv(os.path.join(rdir, "summary_by_category.csv"), index=False)

    # ---- Experiment 1: the three settings compared per translator, matched ----
    e1 = []
    base = df[df.context == "none"]
    for (model, vs), g in base[base.vs != ALL_REFS].groupby(["model", "vs"]):
        for setting, gs in g.groupby("setting"):
            if setting in MATCHED_SETTINGS and gs["translator"].iloc[0] != vs:
                continue
            e1.append({"model": model, "scored_against": vs, "setting": setting,
                       **corpus_scores(gs, ref_names)})
    e1 = pd.DataFrame(e1).sort_values(["model", "scored_against", "setting"]) \
        if e1 else pd.DataFrame()
    if not e1.empty:
        e1.to_csv(os.path.join(rdir, "experiment1.csv"), index=False)

    # ---- context effect vs "none", paired by verse within each arm ----
    metric = "COMET" if args.comet else "chrF++"
    rows = []
    for (m, s, t, vs), g in df.groupby(["model", "setting", "translator", "vs"]):
        b = g[g.context == "none"].set_index("verse")[metric]
        if b.empty:
            continue
        for c, gc in g[g.context != "none"].groupby("context"):
            cur = gc.set_index("verse")[metric]
            common = b.index.intersection(cur.index)
            if len(common) < 2:
                continue
            d = (cur[common] - b[common]).tolist()
            lo, hi = bootstrap_ci(d)
            rows.append({"model": m, "setting": s, "translator": t, "vs": vs,
                         "context": c, "metric": metric, "n": len(d),
                         "mean_delta": sum(d) / len(d), "ci95_low": lo, "ci95_high": hi,
                         "share_improved": sum(x > 0 for x in d) / len(d)})
    eff = pd.DataFrame(rows)
    eff.to_csv(os.path.join(rdir, "context_effect.csv"), index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 25)
    if not e1.empty:
        print("\n=== Experiment 1 — matched, no context ===")
        print(e1.round(2).to_string(index=False))
    print("\n=== Corpus-level scores ===")
    print(summ.round(2).to_string(index=False))
    if not eff.empty:
        print(f"\n=== Context effect vs no context ({metric}, paired by verse) ===")
        print(eff.round(3).to_string(index=False))
    print(f"\nCSV files written to {rdir}")


if __name__ == "__main__":
    main()
