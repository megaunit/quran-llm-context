"""Batched METEOR, TER and COMET scoring under translator-matched evaluation. Local checkpoints and the recorded runtime are needed. Metric agreement is not independent validation of meaning or rhetorical fidelity."""
import argparse
import os
import re

import pandas as pd

from common import load_cache, load_config, path
from score import ALL_REFS, bootstrap_ci, explode_to_scored, load_results, refs_for

WORD = re.compile(r"[a-z0-9']+")

# TER counts edits, so a NEGATIVE change is an improvement. Every other metric here is
# higher-is-better. `improved` is the one place the direction is decided.
LOWER_IS_BETTER = {"TER"}


def improved(metric, delta):
    return delta < 0 if metric in LOWER_IS_BETTER else delta > 0


def toks(s):
    return WORD.findall((s or "").lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--inp", default=None)
    ap.add_argument("--comet-ckpt", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-comet", action="store_true", help="METEOR only (no GPU)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache = load_cache(cfg)["verses"]
    rdir = path(cfg["results_dir"])
    raw = load_results(args.inp or os.path.join(rdir, "translations.jsonl"))
    ref_names = list(cfg["references"])

    raw["arabic"] = raw["verse"].map(lambda k: cache[k]["arabic"])
    for n in ref_names:
        raw[f"ref_{n}"] = raw["verse"].map(lambda k, n=n: cache[k]["translations"][n])
    df = explode_to_scored(raw, ref_names)
    df["translator"] = df["translator"].fillna("-")
    rows = df.to_dict("records")
    print(f"{len(rows)} scored rows across models {sorted(df.model.unique())}", flush=True)

    # ---------------------------------------------------------------- METEOR
    import nltk
    nltk.data.path.insert(0, os.path.expanduser("~/nltk_data"))
    from nltk.translate.meteor_score import meteor_score

    met = []
    for r in rows:
        refs = [toks(r[f"ref_{n}"]) for n in refs_for(r, ref_names)]
        hyp = toks(r["translation"])
        # nltk takes the best of several references, matching chrF++'s multi-ref rule
        met.append(100 * meteor_score(refs, hyp) if hyp else 0.0)
    df["METEOR"] = met
    print("METEOR done", flush=True)

    # ---------------------------------------------------------------- TER
    # Translation Edit Rate: edits needed to turn the output into the reference, as a
    # percentage of reference length. LOWER IS BETTER. With several references sacreBLEU
    # takes the fewest edits over them, the same closest-reference rule as the others.
    from sacrebleu.metrics import TER
    ter = TER()
    df["TER"] = [ter.sentence_score(r["translation"] or "",
                                    [r[f"ref_{n}"] for n in refs_for(r, ref_names)]).score
                 for r in rows]
    print("TER done", flush=True)

    # ---------------------------------------------------------------- COMET
    if not args.no_comet:
        from comet import load_from_checkpoint
        # reload_hparams=True makes COMET read hparams.yaml beside the checkpoint, which
        # points the encoder at the local xlm-roberta-large files. Without it COMET uses
        # the settings baked into the checkpoint and tries to fetch the encoder online,
        # which fails on compute nodes.
        model = load_from_checkpoint(os.path.expanduser(args.comet_ckpt),
                                     reload_hparams=True, local_files_only=True)
        triples = {}
        for r in rows:
            for n in refs_for(r, ref_names):
                triples[(r["arabic"], r["translation"], r[f"ref_{n}"])] = None
        keys = list(triples)
        print(f"COMET: {len(keys)} unique (source, output, reference) triples", flush=True)
        data = [{"src": s, "mt": m, "ref": f} for s, m, f in keys]
        pred = model.predict(data, batch_size=args.batch_size, gpus=1, progress_bar=False)
        score_of = dict(zip(keys, pred.scores))
        df["COMET"] = [100 * max(score_of[(r["arabic"], r["translation"], r[f"ref_{n}"])]
                                 for n in refs_for(r, ref_names)) for r in rows]
        print("COMET done", flush=True)

    metrics = ["METEOR", "TER"] + ([] if args.no_comet else ["COMET"])
    keep = ["model", "setting", "translator", "context", "vs", "verse", "category"] + metrics
    df[keep].to_csv(os.path.join(rdir, "scores_extra_per_item.csv"), index=False)

    keys = ["model", "setting", "translator", "context", "vs"]
    summ = df.groupby(keys)[metrics].mean().reset_index()
    summ.to_csv(os.path.join(rdir, "summary_extra.csv"), index=False)

    # ---------------------------------------------------------------- Experiment 1
    base = df[(df.context == "none") & (df["vs"] != ALL_REFS)]
    arm = base[(base.setting == "zero") | (base.translator == base["vs"])]
    e1 = arm.groupby(["model", "vs", "setting"])[metrics].mean().reset_index()
    e1.to_csv(os.path.join(rdir, "experiment1_extra.csv"), index=False)

    paired = []
    for m in metrics:
        piv = arm.pivot_table(index=["model", "vs", "verse"], columns="setting", values=m)
        for (model, ref), g in piv.groupby(level=[0, 1]):
            for s in [c for c in ("few", "style") if c in g.columns]:
                d = (g[s] - g["zero"]).dropna().tolist()
                if len(d) < 2:
                    continue
                lo, hi = bootstrap_ci(d)
                paired.append({"metric": m, "model": model, "scored_against": ref,
                               "setting": s, "n": len(d), "mean_delta": sum(d) / len(d),
                               "ci95_low": lo, "ci95_high": hi,
                               "share_improved": sum(improved(m, x) for x in d) / len(d),
                               "significant": lo > 0 or hi < 0,
                               "better": improved(m, sum(d) / len(d))})
    pd.DataFrame(paired).to_csv(os.path.join(rdir, "experiment1_extra_paired.csv"), index=False)

    # ---------------------------------------------------------------- Experiment 2
    fx = df[(df.setting == "few") & (df.translator == df["vs"])]
    eff = []
    for m in metrics:
        for (model, ref), g in fx.groupby(["model", "vs"]):
            b = g[g.context == "none"].set_index("verse")[m]
            for c, gc in g[g.context != "none"].groupby("context"):
                cur = gc.set_index("verse")[m]
                common = b.index.intersection(cur.index)
                if len(common) < 2:
                    continue
                d = (cur[common] - b[common]).tolist()
                lo, hi = bootstrap_ci(d)
                eff.append({"metric": m, "model": model, "vs": ref, "context": c,
                            "n": len(d), "mean_delta": sum(d) / len(d),
                            "ci95_low": lo, "ci95_high": hi,
                            "share_improved": sum(improved(m, x) for x in d) / len(d),
                            "significant": lo > 0 or hi < 0,
                            "better": improved(m, sum(d) / len(d))})
    pd.DataFrame(eff).to_csv(os.path.join(rdir, "context_effect_extra.csv"), index=False)

    pd.set_option("display.width", 220)
    print("\n=== Experiment 1, matched, mean per reference ===")
    print(e1[e1.setting.isin(["zero", "few"])].round(2).to_string(index=False))
    print(f"\nWritten to {rdir}")


if __name__ == "__main__":
    main()
