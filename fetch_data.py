"""Download Arabic text, reference translations and Tafsir for every verse we need.

Run on the LOGIN node (compute nodes may not have internet):
    python fetch_data.py

Source: quran.com API v4 (public, no key). Results are cached in data/cache.json;
re-running only fetches verses that are missing.
"""
import argparse
import html
import json
import os
import re
import time

import requests

from common import (check_no_overlap, load_config, load_dataset,
                    load_few_shot_keys, path)

API = "https://api.quran.com/api/v4"
SESSION = requests.Session()


def get(url, params=None, tries=5):
    for i in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if i == tries - 1:
                raise
            wait = 2 ** i
            print(f"  retry in {wait}s ({e})")
            time.sleep(wait)


def clean(text):
    """Strip footnote markers and HTML from API text."""
    if not text:
        return ""
    text = re.sub(r"<sup[^>]*>.*?</sup>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def chapter_lengths():
    data = get(f"{API}/chapters")
    return {c["id"]: c["verses_count"] for c in data["chapters"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = load_dataset(cfg)
    few_keys = load_few_shot_keys(cfg)
    check_no_overlap(dataset, few_keys)

    trans_ids = {**cfg["references"], **cfg.get("few_shot_translation", {})}
    id_to_name = {v: k for k, v in trans_ids.items()}
    w = int(cfg.get("neighbor_window", 0))

    lengths = chapter_lengths()

    # Which verses need what
    full = {r["key"] for r in dataset} | set(few_keys)   # text + translations (+ tafsir)
    tafsir_needed = {r["key"] for r in dataset}
    neighbors = set()
    for r in dataset:
        for a in range(r["ayah"] - w, r["ayah"] + w + 1):
            if 1 <= a <= lengths[r["surah"]] and a != r["ayah"]:
                neighbors.add(f"{r['surah']}:{a}")
    for r in dataset:
        if r["ayah"] > lengths[r["surah"]]:
            raise SystemExit(f"{r['key']} does not exist (surah {r['surah']} has {lengths[r['surah']]} verses)")

    cache_p = path(cfg["cache"])
    cache = {"verses": {}, "chapter_lengths": lengths}
    if os.path.exists(cache_p):
        with open(cache_p, encoding="utf-8") as f:
            cache = json.load(f)
        cache["chapter_lengths"] = lengths

    V = cache["verses"]
    todo = sorted(full | neighbors, key=lambda k: tuple(map(int, k.split(":"))))
    print(f"{len(todo)} verses to check ({len(full)} main, {len(neighbors)} neighbors)")

    for i, key in enumerate(todo, 1):
        entry = V.get(key, {})
        need_trans = key in full and not set(trans_ids) <= set(entry.get("translations", {}))
        if "arabic" not in entry or need_trans:
            params = {"fields": "text_uthmani,text_imlaei"}
            if key in full:
                params["translations"] = ",".join(str(t) for t in trans_ids.values())
            data = get(f"{API}/verses/by_key/{key}", params)
            if data is None:
                raise SystemExit(f"Verse {key} not found by the API")
            v = data["verse"]
            entry["arabic"] = v["text_uthmani"]
            entry["arabic_imlaei"] = v.get("text_imlaei", "")
            if key in full:
                entry["translations"] = {
                    id_to_name[t["resource_id"]]: clean(t["text"])
                    for t in v.get("translations", []) if t["resource_id"] in id_to_name
                }
            time.sleep(0.15)
        if key in tafsir_needed and "tafsir" not in entry:
            data = get(f"{API}/tafsirs/{cfg['tafsir_id']}/by_ayah/{key}")
            entry["tafsir"] = clean(data["tafsir"]["text"]) if data else ""
            time.sleep(0.15)
        V[key] = entry
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}")
            with open(cache_p, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=1)

    # Sanity report
    missing = [k for k in full if not set(trans_ids) <= set(V[k].get("translations", {}))]
    no_tafsir = [k for k in tafsir_needed if not V[k].get("tafsir")]
    print(f"Done. Cache: {cache_p}")
    if missing:
        print(f"WARNING: missing some translations for {missing}")
    if no_tafsir:
        print(f"WARNING: empty tafsir for {no_tafsir}")


if __name__ == "__main__":
    main()
