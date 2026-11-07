"""P2 reference-completion test. Each eligible verse/reference pair is completed under both general and guided instructions. Scores measure reference reproduction; a guided advantage alone does not prove training exposure."""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import sacrebleu
from sacrebleu.metrics import CHRF

from common import load_cache, load_config, load_dataset, path
from prompts import build_probe_messages, parse_translation
from run import call_with_retry, split_arg

CHRF_PP = CHRF(word_order=2)


def halve(text, min_words):
    """Split a reference into (beginning, continuation) at the halfway word."""
    words = text.split()
    if len(words) < min_words:
        return None
    cut = len(words) // 2
    return " ".join(words[:cut]), " ".join(words[cut:])


def run_id(model, verse, ref, mode):
    return f"{model}|{verse}|{ref}|{mode}"


def load_done(p):
    done = set()
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("error"):
                    done.add(run_id(r["model"], r["verse"], r["reference"], r["mode"]))
    return done


def score_file(p, out_csv):
    import pandas as pd
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") or not r.get("completion"):
                continue
            r["chrF++"] = CHRF_PP.sentence_score(r["completion"], [r["truth"]]).score
            r["BLEU"] = sacrebleu.sentence_bleu(r["completion"], [r["truth"]]).score
            rows.append(r)
    if not rows:
        raise SystemExit("No successful probe runs to score.")
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["model", "reference", "verse"], columns="mode",
                          values="chrF++").reset_index()
    if "guided" not in wide or "general" not in wide:
        raise SystemExit("Need both guided and general runs before summarising.")
    wide = wide.dropna(subset=["guided", "general"])
    wide["delta"] = wide["guided"] - wide["general"]
    summ = wide.groupby(["model", "reference"]).agg(
        n=("delta", "size"),
        guided_chrF=("guided", "mean"),
        general_chrF=("general", "mean"),
        mean_delta=("delta", "mean"),
        share_guided_better=("delta", lambda s: (s > 0).mean()),
        share_guided_exact=("guided", lambda s: (s >= 99.9).mean()),
    ).reset_index()
    summ.to_csv(out_csv, index=False)
    import pandas as pd  # noqa: F811  (display options)
    pd.set_option("display.width", 200)
    print("\n=== Memorization probe P2 (guided vs general completion, chrF++) ===")
    print(summ.round(2).to_string(index=False))
    print(f"\nWritten to {out_csv}")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", help="comma list (default: enabled ones)")
    ap.add_argument("--verses", type=int, default=None, help="override probe.verses")
    ap.add_argument("--out", default=None)
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rdir = path(cfg["results_dir"])
    out_p = path(args.out or os.path.join(rdir, "probe.jsonl"))
    out_csv = os.path.join(rdir, "probe_summary.csv")
    os.makedirs(rdir, exist_ok=True)

    if args.score_only:
        score_file(out_p, out_csv)
        return

    probe_cfg = cfg.get("probe", {})
    n_verses = args.verses or int(probe_cfg.get("verses", 60))
    min_words = int(probe_cfg.get("min_ref_words", 12))
    names = probe_cfg.get("translator_names", {})
    cache = load_cache(cfg)["verses"]
    dataset = load_dataset(cfg)[:n_verses]
    refs = list(cfg["references"])
    gen = cfg["generation"]

    model_names = split_arg(args.models) if args.models else \
        [n for n, m in cfg["models"].items() if m.get("enabled")]
    done = load_done(out_p)

    jobs = []
    skipped = 0
    for v in dataset:
        for ref in refs:
            pair = halve(cache[v["key"]]["translations"][ref], min_words)
            if pair is None:
                skipped += 1
                continue
            for mode in ("guided", "general"):
                jobs.append((v["key"], ref, mode, pair))
    print(f"{len(dataset)} verses x {len(refs)} references x 2 modes "
          f"= {len(jobs)} runs per model ({skipped} references too short)")

    for name in model_names:
        m = cfg["models"][name]
        todo = [j for j in jobs if run_id(name, j[0], j[1], j[2]) not in done]
        print(f"[{name}] {len(todo)} to run ({len(jobs) - len(todo)} already done)",
              flush=True)
        for i, (key, ref, mode, (beginning, truth)) in enumerate(todo, 1):
            msgs = build_probe_messages(
                key, cache[key]["arabic"], beginning,
                names.get(ref, ref) if mode == "guided" else None)
            rec = {"model": name,
                   "model_id": m.get("model") or m.get("path"),
                   "verse": key,
                   "reference": ref, "mode": mode, "beginning": beginning,
                   "truth": truth,
                   "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            t0 = time.time()
            try:
                raw = call_with_retry(m, msgs, gen)
                rec.update(raw=raw, completion=parse_translation(raw))
                if not rec["completion"]:
                    rec["error"] = "empty completion"
            except Exception as e:  # noqa: BLE001
                rec.update(raw=None, completion="", error=str(e)[:500])
            rec["seconds"] = round(time.time() - t0, 2)
            with open(out_p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 20 == 0 or i == len(todo):
                print(f"  [{name}] {i}/{len(todo)}", flush=True)

    score_file(out_p, out_csv)


if __name__ == "__main__":
    main()
