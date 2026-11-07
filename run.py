"""Generate translations for every (model, setting, translator, context, verse).

Experiment 1 — the three prompting settings, no context:
    python run.py --models qwen --settings zero,few,style --contexts none
      zero  139 runs          (scored against all three references)
      few   3 x 139 = 417     (one run per translator, scored only against him)
      style 3 x 139 = 417
      total 973 per model

Experiment 2 — contexts, under whichever Experiment 1 setting wins:
    python run.py --models qwen --settings zero  --contexts tafsir,category,neighbors,all
    python run.py --models qwen --settings style --contexts tafsir,category,neighbors,all

See exactly what the model will receive, without calling it:
    python run.py --dry-run --settings few,style --contexts all

Output: results/translations.jsonl (one line per translation). The run is resumable:
anything already in the file is skipped, so it is safe to re-run after a crash or a
Slurm time-out. Failed calls are logged with an "error" field and retried next run.
"""
import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

from common import (check_no_overlap, load_cache, load_categories, load_config,
                    load_dataset, load_few_shot_keys, path)
from prompts import (CONTEXTS, MATCHED_SETTINGS, SETTINGS, build_messages,
                     canon_context, parse_translation)

LOCK = threading.Lock()


# ------------------------------------------------------------------ backends
def call_ollama(m, messages, gen):
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    if not host.startswith("http"):
        host = "http://" + host
    body = {
        "model": m["model"],
        "messages": messages,
        "stream": False,
        "options": {"temperature": gen["temperature"], "num_predict": gen["max_tokens"],
                    "num_ctx": m.get("num_ctx", 8192)},
    }
    if "think" in m:
        body["think"] = m["think"]
    r = requests.post(f"{host}/api/chat", json=body, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


def call_openai_compatible(m, messages, gen):
    """Any OpenAI-style server: a hosted API, or the local llama-server.

    `base_url_env` / `model_env` read the address and model name from the environment,
    which is how slurm/serve_llamacpp.sbatch hands them to the runner. `api_key_env` is
    optional — llama-server needs no key.
    """
    if m.get("base_url_env"):
        base_url = os.environ.get(m["base_url_env"])
        if not base_url:
            raise RuntimeError(f"Environment variable {m['base_url_env']} is not set "
                               "(run this through slurm/serve_llamacpp.sbatch)")
    else:
        base_url = m["base_url"]
    model_id = os.environ.get(m["model_env"]) if m.get("model_env") else None
    body = {"model": model_id or m["model"], "messages": messages,
            "temperature": gen["temperature"], "max_tokens": gen["max_tokens"]}
    body.update(m.get("extra_body", {}))
    headers = {}
    if m.get("api_key_env"):
        key = os.environ.get(m["api_key_env"])
        if not key:
            raise RuntimeError(f"Environment variable {m['api_key_env']} is not set")
        headers["Authorization"] = f"Bearer {key}"
    r = requests.post(base_url.rstrip("/") + "/chat/completions",
                      headers=headers, json=body, timeout=600)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


_HF_CACHE = {}


def _load_hf(m):
    """Load a local HF checkpoint once and keep it on the GPU for the whole run.

    Used for the models in the cluster's shared zoo (/SLURM/public/models) and for the
    ones downloaded as safetensors. No server, no GGUF, no CUDA toolkit: torch's own
    wheel carries the kernels. `trust_remote_code` is what lets Jais-3 load, since it
    ships its own modeling file for an architecture transformers does not know.
    """
    path_ = os.path.expanduser(m["path"])
    if path_ in _HF_CACHE:
        return _HF_CACHE[path_]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    trust = bool(m.get("trust_remote_code", True))
    tok = AutoTokenizer.from_pretrained(path_, trust_remote_code=trust)
    model = AutoModelForCausalLM.from_pretrained(
        path_, dtype=getattr(torch, m.get("dtype", "bfloat16")),
        device_map=m.get("device_map", "auto"), trust_remote_code=trust)
    model.eval()
    _HF_CACHE[path_] = (tok, model)
    return tok, model


def call_transformers(m, messages, gen):
    import torch
    tok, model = _load_hf(m)
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True,
                                   **m.get("chat_template_kwargs", {}))
    # Close an unwanted thinking block the template just opened, so the model starts
    # on the answer instead of reasoning (Jais-3 has no reasoning-off setting).
    text += m.get("prompt_suffix", "")
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=int(gen["max_tokens"]),
            do_sample=False,                      # greedy: temperature 0 everywhere
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None
            else tok.eos_token_id,
        )
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def call_mock(m, messages, gen):
    """For testing the pipeline without any model."""
    return "MOCK translation of " + messages[-1]["content"][:30]


BACKENDS = {"ollama": call_ollama, "openai_compatible": call_openai_compatible,
            "transformers": call_transformers, "mock": call_mock}


def call_with_retry(m, messages, gen, tries=4):
    fn = BACKENDS[m["backend"]]
    for i in range(tries):
        try:
            return fn(m, messages, gen)
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(min(60, 3 * 2 ** i))


# ------------------------------------------------------------------ helpers
def rec_id(model, setting, translator, context, key):
    return f"{model}|{setting}|{translator or '-'}|{canon_context(context)}|{key}"


def record_id(r):
    """rec_id of a result row, tolerating rows written before Nov 4 (`shots` field)."""
    setting = r.get("setting") or r.get("shots")
    translator = r.get("translator")
    if translator is None and setting == "few" and "setting" not in r:
        translator = "yusufali"          # the pre-Nov-4 mismatched few-shot run
    return rec_id(r["model"], setting, translator, r["context"], r["verse"])


def load_done(out_p):
    done = set()
    if os.path.exists(out_p):
        with open(out_p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("error"):
                    done.add(record_id(r))
    return done


def split_arg(s, allowed=None):
    vals = [x.strip() for x in s.split(",") if x.strip()]
    if allowed:
        bad = [v for v in vals if v not in allowed]
        if bad:
            raise SystemExit(f"Unknown value(s) {bad}; choose from {allowed}")
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", help="comma list of model names from config (default: enabled ones)")
    ap.add_argument("--settings", default="zero,few,style",
                    help=f"comma list from {SETTINGS}")
    ap.add_argument("--translators", default=None,
                    help="comma list of references for the matched settings "
                         "(default: every reference in config.yaml)")
    ap.add_argument("--contexts", default="none")
    ap.add_argument("--categories", help="only these categories (comma list)")
    ap.add_argument("--limit", type=int, help="only the first N verses (for quick tests)")
    ap.add_argument("--out", default=None, help="output jsonl (default results/translations.jsonl)")
    ap.add_argument("--dry-run", action="store_true", help="print one prompt per condition and exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = load_dataset(cfg)
    few_keys = load_few_shot_keys(cfg)
    check_no_overlap(dataset, few_keys)
    cache = load_cache(cfg)
    categories = load_categories(cfg)
    window = int(cfg.get("neighbor_window", 2))
    gen = cfg["generation"]

    refs = list(cfg["references"])
    translators = split_arg(args.translators, refs) if args.translators else refs

    if args.categories:
        wanted = set(split_arg(args.categories))
        dataset = [r for r in dataset if r["category"] in wanted]
    if args.limit:
        dataset = dataset[:args.limit]
    setting_list = split_arg(args.settings, SETTINGS)
    ctx_list = [canon_context(c) for c in split_arg(args.contexts, CONTEXTS + ["tone"])]

    # (setting, translator) pairs: matched settings run once per translator.
    arms = []
    for s in setting_list:
        if s in MATCHED_SETTINGS:
            arms.extend((s, t) for t in translators)
        else:
            arms.append((s, None))

    if args.dry_run:
        v = dataset[0]
        for s, t in arms:
            for c in ctx_list:
                msgs = build_messages(v, s, c, cache, categories, few_keys, t, window)
                print("=" * 80,
                      f"\n[{s} | {t or '-'} | {c} | {v['key']} | {v['category']}]\n")
                for mm in msgs:
                    print(f"--- {mm['role']} ---\n{mm['content']}\n")
        return

    model_names = split_arg(args.models) if args.models else \
        [n for n, m in cfg["models"].items() if m.get("enabled")]
    for n in model_names:
        if n not in cfg["models"]:
            raise SystemExit(f"Model '{n}' not in config.yaml")

    out_p = path(args.out or os.path.join(cfg["results_dir"], "translations.jsonl"))
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    done = load_done(out_p)

    for name in model_names:
        m = cfg["models"][name]
        jobs = []
        for s, t in arms:
            for c in ctx_list:
                for v in dataset:
                    if rec_id(name, s, t, c, v["key"]) not in done:
                        jobs.append((s, t, c, v))
        total = len(dataset) * len(arms) * len(ctx_list)
        print(f"[{name}] {len(jobs)} to run ({total - len(jobs)} already done)", flush=True)
        if not jobs:
            continue

        def work(job, name=name, m=m):
            s, t, c, v = job
            msgs = build_messages(v, s, c, cache, categories, few_keys, t, window)
            rec = {"model": name, "model_id": m.get("model") or m.get("path"),
                   "setting": s, "translator": t,
                   "context": c, "verse": v["key"], "category": v["category"],
                   "prompt_sha1": hashlib.sha1(
                       json.dumps(msgs, ensure_ascii=False).encode()).hexdigest()[:12],
                   "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            t0 = time.time()
            try:
                raw = call_with_retry(m, msgs, gen)
                rec.update(raw=raw, translation=parse_translation(raw))
                if not rec["translation"]:
                    rec["error"] = "empty translation"
            except Exception as e:  # noqa: BLE001
                rec.update(raw=None, translation="", error=str(e)[:500])
            rec["seconds"] = round(time.time() - t0, 2)
            with LOCK, open(out_p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rec

        # One in-process model cannot be shared across threads safely, so the
        # transformers backend always runs single-threaded whatever the config says.
        workers = 1 if m["backend"] == "transformers" else int(m.get("workers", 1))
        errors = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(work, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                if fut.result().get("error"):
                    errors += 1
                if i % 20 == 0 or i == len(jobs):
                    print(f"  [{name}] {i}/{len(jobs)}  errors={errors}", flush=True)
        if errors:
            print(f"  [{name}] {errors} failed — re-run the same command to retry them.")

    print(f"Results in {out_p}")


if __name__ == "__main__":
    main()
