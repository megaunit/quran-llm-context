# Quran LLM Context

Companion data and code for **Context Effects in Large Language Model Translation of the Quran**, by Ahmed Alnasser and Sadam Al-Azani.

The study evaluates 139 Quranic verses in eight categories with Qwen3.8-27B, ALLaM-7B and Fanar-2-27B. It compares zero-shot and translator-matched few-shot prompting, then adds tafsir, verse category, neighbouring verses or all three. The references are Saheeh International, Abdel Haleem and Yusuf Ali.

## Contents

- `data/verses_combined_with_text.csv`: the 139 verses, Arabic text, category, source identifier and label origin. `source_cited` means inherited from a study; `rule_based_claude` identifies AI-assisted rule-based proposals. These origins are not independent validation of the labels.
- `data/few_shot.csv`: three demonstration verses, separate from the evaluation set.
- `data/categories.yaml`: the category descriptions supplied in prompts.
- Root Python scripts: data retrieval, prompt construction, generation, scoring, overlap/completion tests and table calculations.
- `results/*.csv`: existing aggregate results, filtered to the three reported models, zero/few-shot settings and matched single-reference rows where applicable. Values were not recomputed.
- `config.yaml`: experimental settings with portable model-path placeholders.

The result filenames describe prompting (`experiment1*`), overall conditions (`summary*`), context changes (`context_effect*`), categories (`summary_by_category.csv`), overlap (`verbatim_overlap_summary.csv`) and completion (`probe_summary.csv`). Files containing `extra` hold METEOR, TER and COMET results.

## Reproducibility and provenance

The eight experiment scripts come from the archived cluster handoff. Only the module descriptions in `prompts.py`, `probe.py` and `score_extra.py` were shortened/corrected for release; executable statements are unchanged. Legacy `style` and backward-compatibility branches remain, but the commands below select only the reported settings. The original `common.py` was missing from that handoff, so the included file is a newly written portable loader, not the original experimental file.

This compact release does **not** contain model weights, the full historical reference cache, raw generation logs or complete per-verse scores. The saved aggregate results can be inspected directly. End-to-end regeneration requires obtaining the model checkpoints, fetching reference text and running inference/scoring. Exact historical output reproduction is not guaranteed: API text, checkpoints, libraries and inference implementations can differ. The release was checked for Python syntax, dataset integrity and prompt reconstruction; new model experiments were not run.

## Running the reported pipeline

Use Python 3.10 or later, run commands from the repository root, and use a separate results directory if you want to preserve the supplied summaries.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4
python fetch_data.py
```

Edit the model checkpoint paths in `config.yaml`. Qwen requires the configured model in a running local Ollama server. ALLaM and Fanar require suitable local checkpoints and GPU memory. Select one model at a time with `--models`; their `enabled` values do not prevent an explicit selection. No credentials or weights are included.

For each of `qwen`, `allam` and `fanar`, run:

```bash
python run.py --models qwen --settings zero,few --contexts none
python run.py --models qwen --settings few --contexts tafsir,category,neighbors,all
```

After generation:

```bash
python score.py
python score_extra.py --comet-ckpt /path/to/wmt22-comet-da/checkpoints/model.ckpt
python overlap.py
python probe.py --models qwen
python probe.py --models allam
python probe.py --models fanar
python paper_tables.py
```

`paper_tables.py` requires the complete per-verse score files produced by the scoring scripts; the bundled aggregate CSVs alone are insufficient. `probe.py` pairs the guided and general completions for each eligible verse/reference pair. The guided instruction both names the translator and asks for exact wording, so their separate effects cannot be identified. Overlap and completion scores measure reproduction, not proven memorisation.

The recorded study environment used PyTorch 2.5.1, Transformers 5.16.1, sacreBLEU 2.6.0, NLTK 3.10.4 and Unbabel COMET 2.2.7. The inference backends differed by model. `requirements.txt` lists package names for setup and is not a tested lockfile for that historical environment; see the manuscript supplement for the recorded hardware and preprocessing.

## Source texts and reuse

Arabic text, English references and Tafsir al-Muyassar were retrieved using the quran.com API v4. The full English translations and tafsir cache are not redistributed here. Respect the terms of the original text providers and model checkpoints. No blanket licence for third-party material is implied. The manuscript and supplement provide the bibliographic references and dataset selection criteria.
