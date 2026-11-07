"""Prompt construction used in the translation and completion experiments. Includes the legacy, unreported style setting for compatibility; use zero and few for the published comparisons."""
import re

SETTINGS = ["zero", "few", "style"]
CONTEXTS = ["none", "tafsir", "category", "neighbors", "all"]

# Runs of these settings are repeated once per translator and scored only against him.
MATCHED_SETTINGS = ("few", "style")

# Runs before Nov 4 called the `category` context "tone"; accepted so old result files
# still load. Everything written from now on uses `category`.
CONTEXT_ALIASES = {"tone": "category"}

SYSTEM = ("You are an expert translator of Classical Arabic into English, "
          "specialising in the Quran.")

TASK = "Translate the following verse of the Quran into English."
FORMAT = "Write only the English translation, with no notes or explanation."
CONTEXT_INTRO = "Context, to help you understand the verse (do not translate it):"
STYLE_INTRO = "Translate in the following style:"

# Traits taken from the published analyses cited in the check-in doc (section 1).
# The translator is deliberately not named in the prompt.
STYLE_TRAITS = {
    "sahih": [
        "modern plain English",
        "close to the structure of the Arabic",
        "render recurring phrases the same way each time",
        "add short clarifying words in square brackets",
    ],
    "haleem": [
        "natural, idiomatic modern English",
        "no archaic words",
        "follow the sense and flow of the passage rather than the Arabic word order",
    ],
    "yusufali": [
        "elevated, archaic English (thee, thou, -eth)",
        "interpretive, at times expanding the meaning",
        "formal, literary rhythm",
    ],
}


def canon_context(c):
    """Map a context name (including the old `tone`) onto its current name."""
    return CONTEXT_ALIASES.get(c, c)


def _context_blocks(verse, cache, categories, context, window):
    """The context blocks for one verse, in the fixed order tafsir, category, neighbors."""
    context = canon_context(context)
    V = cache["verses"]
    blocks = []
    if context in ("tafsir", "all"):
        blocks.append("Tafsir: " + V[verse["key"]]["tafsir"])
    if context in ("category", "all"):
        cat = categories.get(verse["category"])
        if cat is None:
            raise SystemExit(
                f"Category '{verse['category']}' ({verse['key']}) is not in categories.yaml")
        blocks.append(f"Rhetorical function: {verse['category']} ({cat['name']}). "
                      f"{cat['description']}")
    if context in ("neighbors", "all"):
        s, a0 = verse["surah"], verse["ayah"]
        lengths = cache["chapter_lengths"]
        n = int(lengths.get(str(s), lengths.get(s)))

        def line(a):
            k = "%d:%d" % (s, a)
            return "(%s) %s" % (k, V[k]["arabic"])

        before = [line(a) for a in range(max(1, a0 - window), a0)]
        after = [line(a) for a in range(a0 + 1, min(n, a0 + window) + 1)]
        if before:
            blocks.append("Preceding verses: " + " ".join(before))
        if after:
            blocks.append("Following verses: " + " ".join(after))
    return blocks


def _style_block(translator):
    traits = STYLE_TRAITS.get(translator)
    if not traits:
        raise SystemExit(f"No style traits defined for translator '{translator}'")
    return STYLE_INTRO + "\n" + "\n".join("- " + t for t in traits)


def _user_turn(key, arabic, context_blocks=(), style_block=None):
    parts = [TASK]
    if context_blocks:
        parts.append(CONTEXT_INTRO + "\n" + "\n\n".join(context_blocks))
    if style_block:
        parts.append(style_block)
    parts.append(f"Verse (Q {key}):\n{arabic}")
    parts.append(FORMAT)
    return "\n\n".join(parts)


def few_shot_order(few_shot_keys, cache):
    """Fixed example order: shortest Arabic first (check-in doc, section 3)."""
    V = cache["verses"]
    return sorted(few_shot_keys, key=lambda k: (len(V[k]["arabic"]), k))


def build_messages(verse, setting, context, cache, categories, few_shot_keys,
                   translator=None, window=2):
    """Return an OpenAI-style message list for one (setting, context, verse).

    translator is required for the matched settings (`few`, `style`) and must be None
    for `zero`.
    """
    if setting not in SETTINGS:
        raise SystemExit(f"Unknown setting '{setting}'; choose from {SETTINGS}")
    if setting in MATCHED_SETTINGS and not translator:
        raise SystemExit(f"Setting '{setting}' needs a translator")
    if setting == "zero" and translator:
        raise SystemExit("Setting 'zero' takes no translator")

    V = cache["verses"]
    msgs = [{"role": "system", "content": SYSTEM}]

    if setting == "few":
        for k in few_shot_order(few_shot_keys, cache):
            msgs.append({"role": "user", "content": _user_turn(k, V[k]["arabic"])})
            msgs.append({"role": "assistant",
                         "content": V[k]["translations"][translator]})

    blocks = _context_blocks(verse, cache, categories, context, window)
    style = _style_block(translator) if setting == "style" else None
    msgs.append({"role": "user",
                 "content": _user_turn(verse["key"], V[verse["key"]]["arabic"],
                                       blocks, style)})
    return msgs


def build_probe_messages(key, arabic, beginning, translator_name=None):
    """Memorization probe P2 (check-in doc, section 6.5).

    Guided = translator_name given; general = None.
    """
    if translator_name:
        head = ("Below is a verse of the Quran and the beginning of its English "
                f"translation by {translator_name}. Complete the translation exactly "
                "as it appears in that translation.")
    else:
        head = ("Below is a verse of the Quran and the beginning of an English "
                "translation of it. Complete the translation.")
    body = (f"Verse (Q {key}): {arabic}\n"
            f"Beginning: {beginning}\n\n"
            "Write only the rest of the translation.")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": head + "\n\n" + body}]


_THINK = re.compile(r"<think>.*?</think>", re.S)
_OPEN_THINK = re.compile(r"<think>.*", re.S)
# Jais-3 writes its reasoning inside <ifm|think>, <ifm|think_fast> or
# <ifm|think_faster>. Strip those too, closed or left open by a token limit.
_IFM_THINK = re.compile(r"<ifm\|think[a-z_]*>.*?</ifm\|think[a-z_]*>", re.S)
_IFM_OPEN = re.compile(r"<ifm\|think[a-z_]*>.*", re.S)
_IFM_STRAY = re.compile(r"</?ifm\|think[a-z_]*>")


def parse_translation(raw):
    """Clean one model reply down to the translation text.

    The Nov 4 prompts ask for bare text, so there are no <translation> tags to strip;
    they are still handled for result files written before the change.
    """
    if raw is None:
        return ""
    raw = _THINK.sub("", raw)
    raw = _IFM_THINK.sub("", raw)
    raw = _OPEN_THINK.sub("", raw)          # unterminated thinking block
    raw = _IFM_OPEN.sub("", raw)
    raw = _IFM_STRAY.sub("", raw)
    m = re.findall(r"<translation>(.*?)</translation>", raw, flags=re.S)
    if m:
        return m[-1].strip()
    m = re.search(r"<translation>(.*)", raw, flags=re.S)
    if m:
        return m.group(1).strip()
    out = raw.strip()
    # Models sometimes wrap the answer in quotes or a lone code fence.
    if out.startswith("```"):
        out = re.sub(r"^```[a-zA-Z]*\n?", "", out)
        out = re.sub(r"\n?```$", "", out).strip()
    if len(out) > 1 and out[0] in "\"“" and out[-1] in "\"”":
        out = out[1:-1].strip()
    return out
