"""Every number for the one-table-per-research-question layout, from one place.

Sources (all in results/):
    experiment1.csv, summary.csv            corpus chrF++ and BLEU-4
    experiment1_extra.csv, summary_extra.csv  per-verse means of METEOR, TER, COMET
    scores_per_item.csv                     per-verse chrF++ and sentence BLEU-4
    scores_extra_per_item.csv               per-verse METEOR, TER, COMET
    verbatim_overlap_summary.csv, probe_summary.csv  memorisation probes

Significance is always a paired bootstrap over verses, 2,000 resamples, per reference;
cells report how many of the three references are significant, and in which direction.
"""
import random

import pandas as pd

R = "results/"
MODELS = ["qwen", "allam", "fanar"]
NAMES = {"qwen": "Qwen3.8-27B", "allam": "ALLaM-7B", "fanar": "Fanar-2-27B"}
REFS = ["sahih", "haleem", "yusufali"]
METRICS = ["chrF++", "BLEU", "METEOR", "TER", "COMET"]
LOWER = {"TER"}
CTX = ["tafsir", "category", "neighbors", "all"]
CTX_LABEL = {"tafsir": "tafsir", "category": "category", "neighbors": "neighbouring verses",
             "all": "all three"}
SUP = str.maketrans("+-0123", "⁺⁻⁰¹²³")


def boot(x, it=2000, seed=0):
    rnd = random.Random(seed)
    n = len(x)
    m = sorted(sum(rnd.choice(x) for _ in range(n)) / n for _ in range(it))
    return m[int(.025 * it)], m[int(.975 * it) - 1]


def better(metric, d):
    return d < 0 if metric in LOWER else d > 0


# ---------------------------------------------------------------- per-verse long table
a = pd.read_csv(R + "scores_per_item.csv")
b = pd.read_csv(R + "scores_extra_per_item.csv")
key = ["model", "setting", "translator", "context", "vs", "verse"]
a = a[key + ["chrF++", "BLEU"]]
b = b[key + ["METEOR", "TER", "COMET"]]
pv = a.merge(b, on=key, how="inner")
pv = pv[(pv["vs"] != "all3") & pv.model.isin(MODELS)]
# matched rows only: zero-shot against every reference, few-shot against its own
pv = pv[(pv.setting == "zero") | ((pv.setting == "few") & (pv.translator == pv["vs"]))]


def sig_marks(metric, base, cur):
    """'(+k)' / '(-k)' = references where cur differs significantly from base."""
    up = down = 0
    for ref in REFS:
        x = base[base["vs"] == ref].set_index("verse")[metric]
        y = cur[cur["vs"] == ref].set_index("verse")[metric]
        common = x.index.intersection(y.index)
        d = (y[common] - x[common]).tolist()
        if len(d) < 2:
            continue
        lo, hi = boot(d)
        if lo > 0 or hi < 0:
            if better(metric, sum(d) / len(d)):
                up += 1
            else:
                down += 1
    s = ""
    if up:
        s += f"+{up}"
    if down:
        s += f"-{down}"
    return s.translate(SUP)


def share_improved(metric, base, cur):
    out = []
    for ref in REFS:
        x = base[base["vs"] == ref].set_index("verse")[metric]
        y = cur[cur["vs"] == ref].set_index("verse")[metric]
        common = x.index.intersection(y.index)
        d = (y[common] - x[common]).tolist()
        out.append(sum(better(metric, v) for v in d) / len(d))
    return sum(out) / len(out)


# ---------------------------------------------------------------- corpus-level values
e1 = pd.read_csv(R + "experiment1.csv")
e1x = pd.read_csv(R + "experiment1_extra.csv")
sm = pd.read_csv(R + "summary.csv")
smx = pd.read_csv(R + "summary_extra.csv")


def corpus(model, setting, context):
    """Mean over the three references of each metric, as the paper reports it."""
    vals = {}
    if setting == "zero":
        g = e1[(e1.model == model) & (e1.setting == "zero")]
        gx = e1x[(e1x.model == model) & (e1x.setting == "zero")]
        vals["chrF++"], vals["BLEU"] = g["chrF++"].mean(), g["BLEU"].mean()
        for m in ("METEOR", "TER", "COMET"):
            vals[m] = gx[m].mean()
    else:
        g = sm[(sm.model == model) & (sm.setting == "few") & (sm.context == context)
               & (sm["vs"] == sm.translator)]
        gx = smx[(smx.model == model) & (smx.setting == "few") & (smx.context == context)
                 & (smx["vs"] == smx.translator)]
        vals["chrF++"], vals["BLEU"] = g["chrF++"].mean(), g["BLEU"].mean()
        for m in ("METEOR", "TER", "COMET"):
            vals[m] = gx[m].mean()
    return vals


HDR = "| Model | Condition | chrF++ ↑ | BLEU-4 ↑ | METEOR ↑ | TER ↓ | COMET ↑ |"
SEP = "| --- | --- | --- | --- | --- | --- | --- |"

# ---------------------------------------------------------------- RQ1 table
print("### RQ1 TABLE\n")
print(HDR)
print(SEP)
for mo in MODELS:
    z = corpus(mo, "zero", None)
    f = corpus(mo, "few", "none")
    zb = pv[(pv.model == mo) & (pv.setting == "zero")]
    fb = pv[(pv.model == mo) & (pv.setting == "few") & (pv.context == "none")]
    print(f"| {NAMES[mo]} | Zero-shot (baseline) | " +
          " | ".join(f"{z[m]:.2f}" for m in METRICS) + " |")
    print(f"| | Baseline + few-shot | " +
          " | ".join(f"{f[m]:.2f}{sig_marks(m, zb, fb)}" for m in METRICS) + " |")

# ---------------------------------------------------------------- RQ2 table
print("\n### RQ2 TABLE\n")
print("| Model | Condition | chrF++ ↑ | BLEU-4 ↑ | METEOR ↑ | TER ↓ | COMET ↑ "
      "| Verses improved by context |")
print("| --- | --- | --- | --- | --- | --- | --- | --- |")
for mo in MODELS:
    z = corpus(mo, "zero", None)
    f = corpus(mo, "few", "none")
    fb = pv[(pv.model == mo) & (pv.setting == "few") & (pv.context == "none")]
    print(f"| {NAMES[mo]} | Zero-shot (baseline) | " +
          " | ".join(f"{z[m]:.2f}" for m in METRICS) + " | — |")
    print(f"| | Baseline + few-shot | " +
          " | ".join(f"{f[m]:.2f}" for m in METRICS) + " | — |")
    for c in CTX:
        v = corpus(mo, "few", c)
        cb = pv[(pv.model == mo) & (pv.setting == "few") & (pv.context == c)]
        shares = [share_improved(m, fb, cb) for m in METRICS]
        print(f"| | Baseline + few-shot + {CTX_LABEL[c]} | " +
              " | ".join(f"{v[m]:.2f}{sig_marks(m, fb, cb)}" for m in METRICS) +
              f" | {min(shares)*100:.0f}–{max(shares)*100:.0f}% |")

# ---------------------------------------------------------------- RQ3 table
print("\n### RQ3 TABLE  (change from Baseline + few-shot, mean over the four contexts "
      "and three references, per-verse means)\n")
cats = pd.read_csv(R + "scores_per_item.csv")[["verse", "category"]].drop_duplicates()
pvc = pv.merge(cats, on="verse")
order = ["الوعيد", "الزجر",
         "العذاب", "الرحمة",
         "السخرية", "النعيم",
         "الأمر", "النهي"]
print("| Category | n | Model | Δ chrF++ ↑ | Δ BLEU-4 ↑ | Δ METEOR ↑ "
      "| Δ TER ↓ | Δ COMET ↑ |")
print("| --- | --- | --- | --- | --- | --- | --- | --- |")
for cat in order:
    first = True
    for mo in MODELS:
        g = pvc[(pvc.category == cat) & (pvc.model == mo) & (pvc.setting == "few")]
        base = g[g.context == "none"].set_index(["vs", "verse"])
        ds = {}
        for m in METRICS:
            deltas = []
            for c in CTX:
                cur = g[g.context == c].set_index(["vs", "verse"])[m]
                common = base.index.intersection(cur.index)
                deltas.append((cur[common] - base.loc[common, m]).mean())
            ds[m] = sum(deltas) / len(deltas)
        n = g[g.context == "none"].verse.nunique()
        lead = f"| {cat} | {n} " if first else "| | "
        print(lead + f"| {NAMES[mo]} | " + " | ".join(f"{ds[m]:+.2f}" for m in METRICS) + " |")
        first = False

# ---------------------------------------------------------------- 5.5 table
print("\n### 5.5 TABLE (zero-shot only)\n")
ov = pd.read_csv(R + "verbatim_overlap_summary.csv")
ov = ov[(ov.setting == "zero") & (ov.context == "none")]
pr = pd.read_csv(R + "probe_summary.csv")
zrows = {mo: e1[(e1.model == mo) & (e1.setting == "zero") & (e1.scored_against == "sahih")]
         for mo in MODELS}
zx = {mo: e1x[(e1x.model == mo) & (e1x.setting == "zero") & (e1x["vs"] == "sahih")]
      for mo in MODELS}
order_m = ["fanar", "qwen", "allam"]
print("| Measure | " + " | ".join(NAMES[m] for m in order_m) + " |")
print("| --- | --- | --- | --- |")
for lab, fn in [
    ("Score against Saheeh: chrF++ ↑", lambda m: zrows[m]["chrF++"].iloc[0]),
    ("Score against Saheeh: BLEU-4 ↑", lambda m: zrows[m]["BLEU"].iloc[0]),
    ("Score against Saheeh: METEOR ↑", lambda m: zx[m]["METEOR"].iloc[0]),
    ("Score against Saheeh: TER ↓", lambda m: zx[m]["TER"].iloc[0]),
    ("Score against Saheeh: COMET ↑", lambda m: zx[m]["COMET"].iloc[0]),
]:
    print(f"| {lab} | " + " | ".join(f"{fn(m):.2f}" for m in order_m) + " |")
for ref, rn in (("sahih", "Saheeh"), ("haleem", "Abdel Haleem"), ("yusufali", "Yusuf Ali")):
    print(f"| Outputs with an 8+ word exact run from {rn} ↑ | " + " | ".join(
        f"{ov[(ov.model == m) & (ov.reference == ref)].share_8plus.iloc[0]*100:.1f}%"
        for m in order_m) + " |")
print("| Mean longest exact run from Saheeh, words ↑ | " + " | ".join(
    f"{ov[(ov.model == m) & (ov.reference == 'sahih')].mean_longest_span.iloc[0]:.2f}"
    for m in order_m) + " |")
for ref, rn in (("sahih", "Saheeh"), ("haleem", "Abdel Haleem"), ("yusufali", "Yusuf Ali")):
    print(f"| Unprompted completion of {rn}, chrF++ ↑ | " + " | ".join(
        f"{pr[(pr.model == m) & (pr.reference == ref)].general_chrF.iloc[0]:.2f}"
        for m in order_m) + " |")
print("| Gain from naming Saheeh in the completion prompt, chrF++ ↑ | " + " | ".join(
    f"{pr[(pr.model == m) & (pr.reference == 'sahih')].mean_delta.iloc[0]:+.2f}"
    for m in order_m) + " |")
