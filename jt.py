#!/usr/bin/env python3
"""
jt - local Japanese <-> English translator for any app on your Mac (runs on Ollama).

  jt "メッセージ"             translate text (direction is detected automatically)
  echo "text" | jt            translate from a pipe
  jt                          interactive: paste a message, then press Enter on an empty line
  jt -c                       translate the clipboard and copy the result back
  jt -c --show                same, and show the translation in a popup (macOS)
  jt --check "text"           also translate the result back, to spot meaning drift
  jt --selftest               run the regression tests in tests.json
  jt --doctor                 check that Ollama and the model are ready
  jt --warm / jt --stop       load the model and warm it up now / remove it from memory (frees ~9 GB)
  jt --progress corner        where the hotkey's progress window goes: corner, center or off
  jt --format bilingual       what the hotkey copies: "English: …/Japanese: …" (default) or plain
  jt --keep-loaded 15m        how long the model stays in memory after a translation (default 8h)
  jt --clear-cache            forget the remembered hotkey translations (repeats come back instantly)

Settings (environment variables): JT_MODEL (default gemma4:12b-it-qat), OLLAMA_HOST.
Standard library only; works with the Python 3.9 that ships with macOS.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import time
import sys
import threading
import traceback
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
GLOSSARY_FILE = HERE / "glossary.txt"                  # core rules: always sent to the model
PROJECT_FILE = HERE / "project-glossary.txt"           # app terms: only the ones found in the message
EXAMPLES_FILE = HERE / "examples.jsonl"                # built-in example translations (style teaching)
TESTS_FILE = HERE / "tests.json"

MODEL = os.environ.get("JT_MODEL", "gemma4:12b-it-qat")  # chosen 5 Oct 2026 (benchmarked against qwen3.5:9b)
HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not HOST.startswith("http"):
    HOST = "http://" + HOST
HOST = HOST.rstrip("/")

MAX_TERMS = 25    # most project terms sent with one message (keeps the prompt small for a 9B model)

# Settings Ollama loads the model with. They must be the same on EVERY request: a different value makes
# Ollama reload the whole model first (~7 s), so jt never varies them per message.
# num_ctx: Ollama's default (4096) silently drops the START of a prompt that doesn't fit, which is where the
#   rules and glossary are. num_ctx fits the prompt plus any message below CHUNK_ABOVE and its translation.
#   Ollama's own size estimate hides it, but the context is pinned memory: gemma4:12b-it-qat pins 7.34 GB at 4096,
#   7.59 GB at 8192 and 7.94 GB at 16384 (measured 7 Oct 2026, before the 8-bit KV cache). 8192 keeps 2-6 paragraph
#   messages whole. With the 8-bit KV cache, 4096 saves only ~34 MB (measured 8 Oct), so it isn't worth splitting
#   messages over ~900 characters: tried and reverted the same day.
# num_batch: Ollama can only reuse a cached prompt on these models up to a checkpoint one batch before the
#   end (qwen3.5 is hybrid, gemma4 uses sliding-window layers). With the default batch the whole ~750-token
#   prompt was re-read for every message (5-8 s); with 64 almost all of it is reused (2-3 s, same output).
#   64 was the fastest of 16-512 for both models (measured 5 Oct 2026).
# use_mmap false: the weights are copied into memory the GPU keeps pinned, so they always stay there. With use_mmap
#   they're file pages, the first thing macOS drops when memory is short (Docker): in real use the model shrank to
#   0-0.6 GB between consecutive messages and translations took 13-42 s instead of ~3 s (~/.jt/timing.log, 7 Oct
#   2026). Short tests looked fine only because they happened to run while nothing needed the memory. So it stays off
#   (the `jt --mmap` switch was removed).
LOAD_OPTIONS = {"num_ctx": int(os.environ.get("JT_NUM_CTX", "8192")),
                "num_batch": int(os.environ.get("JT_NUM_BATCH", "64")),
                "use_mmap": False}
CHUNK_ABOVE = 2000    # estimated tokens (~2,000 Japanese characters): longer messages go paragraph by paragraph
CHUNK_SIZE = 1200     # target tokens per part (a part + the previous part as context + its translation fit 8192)
KEEP_ALIVE = "8h"     # default: the model stays loaded 8 hours after the last use (a workday). While loaded it pins
                      # ~9 GB of RAM (7.7 GB locked); `jt --keep-loaded 15m` frees it sooner (see keep_alive())
LANG_NAME = {"en": "English", "ja": "Japanese"}

# Per-user state, never shared: the last translation (so the hotkey can tell when the clipboard still holds it),
# the hotkey lock, settings, logs and the cache.
STATE_DIR = Path(os.environ.get("JT_STATE_DIR", str(Path.home() / ".jt")))
LAST_FILE = STATE_DIR / "last.json"
LOCK_FILE = STATE_DIR / "hotkey.lock"
LOG_FILE = STATE_DIR / "jt.log"
SETTINGS_FILE = STATE_DIR / "settings.json"
TIMING_FILE = STATE_DIR / "timing.log"  # one JSON line per hotkey translation (speed, memory), for `jt --doctor`
CACHE_FILE = STATE_DIR / "cache.json"    # recent hotkey results: an identical message comes back instantly
CACHE_LIMIT = 500                        # entries kept (least recently used dropped): ~1 MB at most
SHORT_ENGLISH_WORDS = 4                  # one-line English shorter than this skips the correction step
PROGRESS_MODES = ("corner", "center", "off")  # where the hotkey's progress window appears; "off" = a notification
FORMATS = ("bilingual", "plain")  # hotkey output: "English: …/Japanese: …" (English corrected first), or the translation only
CONFIRM_ABOVE = 2000  # clipboard characters (~4+ paragraphs): the hotkey asks before a long translation
AUTOSTART = os.environ.get("JT_AUTOSTART", "1") != "0"  # start Ollama when it isn't running (0 = never)

JA_CHARS = re.compile(r"[぀-ヿ㐀-鿿ｦ-ﾟ]")
EN_CHARS = re.compile(r"[A-Za-z]")

RULES = """You are a translator for a technical work chat between Japanese and English speakers.
Translate the user's message into {target}.
Rules:
- Translate exactly. Do not add, drop, or soften anything. Never add greetings, names, or forms of address (no "Ma'am", "Sir", "Dear").
- Keep every question and request as a separate item. Never merge two asks into one.
- Keep requests as requests: if the message asks someone to tell, confirm or check something (教えていただけますか, ご確認いただけますでしょうか, "Could you tell us", "Could you confirm"), keep that framing in the translation. Never turn a request into a bare question.
- Keep question types: どの/何/いつ = which/what/when (asks for information); かどうか = whether (yes/no).
- Keep certainty: 可能性があります = may/might; 必要があります = need to; と思います = I think.
- Keep tentative requests tentative: 一度〜してみて / 一度〜していただき = "try ... (once, as a test)". Always use the word "try" for this, not just "once".
- こちら / こちらの / こちら側 = this / here / our side (the writer's side). そちら = your side. Never swap them.
- Keep the line breaks and blank lines of the message: every line stays a separate line, even a greeting line.
{style}
- Always use the GLOSSARY below, in both directions (Japanese -> English and English -> Japanese).
- Output only the translation. No notes, no explanations, no quotation marks around it.

GLOSSARY (Japanese = English):
{glossary}"""

PROJECT_HEADER = "PROJECT TERMS found in the next message (names used in the system being discussed; translate them exactly like this):\n"

STYLE = {
    "en": "- Write clear, natural business English.",
    "ja": "- Write natural, polite business Japanese (です/ます; use 〜いただけますでしょうか for requests).",
}

GUI = False  # set in clipboard mode so errors show as a popup


# ---------------------------------------------------------------- helpers

class JtError(Exception):
    """A problem the user can fix (Ollama not running, model missing, ...)."""


class JtNotice(JtError):
    """Not an error: jt didn't translate on purpose (nothing new on the clipboard, already busy, ...)."""


class JtCancelled(JtNotice):
    """The user pressed Cancel in the progress window. Nothing is copied and nothing more is shown."""


def die(msg, notice=False):
    if GUI:
        popup(msg, title="jt" if notice else "jt - error")
        sys.exit(0)  # the popup already told the user; a failing exit would make Shortcuts add its own alert
    sys.exit("jt: " + msg)


def log_error(what):
    """Append a traceback to ~/.jt/jt.log, for problems that aren't the user's to fix."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write("--- %s %s\n%s\n" % (datetime.datetime.now().isoformat(timespec="seconds"), what, traceback.format_exc()))
    except OSError:
        pass


def has_words(text):
    """False when there's nothing to translate: only links, paths, e-mail addresses, numbers or symbols."""
    rest = re.sub(r"\S+://\S+|\S+@\S+|\S*[/\\]\S*", " ", text)
    return bool(JA_CHARS.search(rest)) or re.search(r"[A-Za-z]{2,}", rest) is not None


def swap_used_gb():
    """Swap in use on macOS, in GB (None elsewhere or if sysctl can't tell)."""
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        return round(float(re.search(r"used = ([\d.]+)M", out).group(1)) / 1024, 1)
    except (OSError, ValueError, AttributeError):
        return None


def median(values):
    values = sorted(values)
    return values[len(values) // 2] if values else None


def log_timing(total, text, src, tgt):
    """Append one line to ~/.jt/timing.log: how long this hotkey translation took and why (Ollama's stats for the last
    request, how much of the model was in memory, swap). `jt --doctor` summarizes it."""
    entry = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "total_s": round(total, 2),
             "chars": len(text), "dir": "%s>%s" % (src, tgt)}
    st = LAST_STATS
    if st.get("load_duration"):
        entry["load_s"] = round(st["load_duration"] / 1e9, 2)
    if st.get("prompt_eval_duration"):
        entry["read_tps"] = round(st.get("prompt_eval_count", 0) / (st["prompt_eval_duration"] / 1e9))
    if st.get("eval_duration"):
        entry["write_tps"] = round(st.get("eval_count", 0) / (st["eval_duration"] / 1e9), 1)
    try:
        pids = subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True).stdout.split()
        if pids:
            rss = subprocess.run(["ps", "-o", "rss=", "-p", pids[0]], capture_output=True, text=True).stdout.strip()
            entry["resident_gb"] = round(int(rss) / 1048576, 1)
    except (OSError, ValueError):
        pass
    swap = swap_used_gb()
    if swap is not None:
        entry["swap_gb"] = swap
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with TIMING_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def timing_summary(last=30):
    """"median X s, slowest Y s over the last N hotkey translations", or None."""
    try:
        lines = TIMING_FILE.read_text(encoding="utf-8").splitlines()[-last:]
        totals = sorted(json.loads(l)["total_s"] for l in lines if l.strip())
    except (OSError, ValueError, KeyError):
        return None
    if not totals:
        return None
    return "median %.1f s, slowest %.1f s over the last %d hotkey translations (%s)" % (
        median(totals), totals[-1], len(totals), TIMING_FILE)


def _file_text(path):
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def cache_key(text, target, fmt):
    """Everything that can change the hotkey's result: the message, forced direction, format, model and load settings,
    the prompts, glossaries and examples. Change any of them and old entries simply stop matching."""
    parts = [text, str(target), fmt, active_model(), json.dumps(LOAD_OPTIONS, sort_keys=True), RULES, json.dumps(STYLE),
             PROJECT_HEADER, FIX_PROMPT, json.dumps(FIX_EXAMPLES), str(SHORT_ENGLISH_WORDS)]
    parts += [_file_text(f) for f in (GLOSSARY_FILE, PROJECT_FILE, EXAMPLES_FILE)]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def _load_cache():
    try:
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        return cache if isinstance(cache, dict) else {}
    except (OSError, ValueError):
        return {}


def cache_get(key):
    entry = _load_cache().get(key)
    return entry if isinstance(entry, dict) and entry.get("out") else None


def cache_put(key, entry):
    cache = _load_cache()
    entry = dict(entry, used=time.time())
    cache[key] = entry
    if len(cache) > CACHE_LIMIT:  # forget the least recently used
        for old in sorted(cache, key=lambda k: cache[k].get("used", 0))[:len(cache) - CACHE_LIMIT]:
            del cache[old]
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(CACHE_FILE) + ".tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        os.replace(str(tmp), str(CACHE_FILE))
    except OSError:
        pass


def cache_touch(key):
    cache = _load_cache()
    if key in cache:
        cache_put(key, cache[key])


def cache_summary():
    cache = _load_cache()
    try:
        size = CACHE_FILE.stat().st_size
    except OSError:
        size = 0
    return "%d translations remembered (%.0f KB, at most %d; clear with: jt --clear-cache)" % (
        len(cache), size / 1024, CACHE_LIMIT)


def save_last(translation, copied=None):
    """The last translation and what went to the clipboard (the bilingual block), so the hotkey can tell when the
    clipboard still holds it (the user forgot ⌘C)."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        LAST_FILE.write_text(json.dumps({"translation": translation, "copied": copied or translation,
                                         "time": datetime.datetime.now().isoformat(timespec="minutes")},
                                        ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass  # only a convenience


def load_settings():
    try:
        settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return settings if isinstance(settings, dict) else {}
    except (OSError, ValueError):
        return {}


def save_setting(key, value):
    settings = load_settings()
    settings[key] = value
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings, indent=1), encoding="utf-8")


def progress_mode():
    """Where the progress window goes: JT_PROGRESS for one run, else `jt --progress` (saved), else the top-right corner."""
    mode = os.environ.get("JT_PROGRESS") or load_settings().get("progress")
    return mode if mode in PROGRESS_MODES else "corner"


def keep_alive():
    """How long the model stays in memory after the last translation: JT_KEEP_ALIVE for one run, else
    `jt --keep-loaded` (saved), else 8h. Shorter frees ~9 GB sooner (helps with Docker); the first translation
    after that waits for a reload (~15-20 s)."""
    value = os.environ.get("JT_KEEP_ALIVE") or load_settings().get("keep_loaded") or KEEP_ALIVE
    return value if is_duration(value) else KEEP_ALIVE


def is_duration(value):
    """A keep-alive time Ollama understands: 15m, 1h, 8h, 30s."""
    return bool(re.match(r"^\d+[smh]$", str(value)))


def output_format():
    """The hotkey's output: JT_FORMAT for one run, else `jt --format` (saved), else bilingual."""
    fmt = os.environ.get("JT_FORMAT") or load_settings().get("format")
    return fmt if fmt in FORMATS else "bilingual"


def load_last():
    try:
        last = json.loads(LAST_FILE.read_text(encoding="utf-8"))
        return last if last.get("translation") else None
    except (OSError, ValueError, AttributeError):
        return None


def detect_source(text):
    ja = len(JA_CHARS.findall(text))
    en = len(EN_CHARS.findall(text))
    return "ja" if ja > 0 and ja * 2 >= en else "en"


def load_glossary():
    if not GLOSSARY_FILE.exists():
        return []
    lines = []
    for raw in GLOSSARY_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            lines.append(line)
    return lines


_project_cache = {"mtime": None, "terms": []}


def load_project_terms():
    """(japanese, english) pairs from project-glossary.txt, re-read only when the file changes."""
    try:
        mtime = PROJECT_FILE.stat().st_mtime
    except OSError:
        return []
    if _project_cache["mtime"] != mtime:
        terms = []
        for raw in PROJECT_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or " = " not in line:
                continue
            ja, en = (part.strip() for part in line.split(" = ", 1))
            if len(ja) >= 2 and len(en) >= 2:
                terms.append((ja, en))
        _project_cache.update(mtime=mtime, terms=terms)
    return _project_cache["terms"]


def relevant_terms(text, src):
    """Only the project terms that actually appear in this message, longest first.

    A term found only inside a longer matched term is skipped (e.g. "shipping" inside
    "shipping address"), so the model isn't pushed toward a word-by-word translation.
    """
    cands = []
    for ja, en in load_project_terms():
        if src == "ja":
            cands.append((ja, ja, en))
        else:
            word = re.sub(r"\s*\(.*?\)\s*$", "", en).lower()  # match "shipper", not "shipper (the client...)"
            if len(word) >= 3:
                cands.append((word, ja, en))
    cands.sort(key=lambda c: -len(c[0]))

    remaining = text if src == "ja" else text.lower()
    seen, out = set(), []
    for needle, ja, en in cands:
        if src == "ja":
            found = needle in remaining
            pattern = re.escape(needle)
        else:
            # whole words, plurals too ("shipping labels", "locations"), but not inside paths, file names or
            # identifiers (/api/v2/arrivals, arrival_date), which must stay exactly as written
            pattern = r"(?<![a-z0-9_/.\\-])" + re.escape(needle) + r"(?:s|es)?(?![a-z0-9_/\\-]|\.[a-z0-9])"
            found = re.search(pattern, remaining) is not None
        if not found or ja in seen:
            continue
        seen.add(ja)
        out.append((ja, en))
        remaining = re.sub(pattern, lambda m: " " * len(m.group(0)), remaining)  # consume this occurrence
        if len(out) >= MAX_TERMS:
            break
    return out


def load_examples():
    """Built-in example translations shipped with jt (examples.jsonl)."""
    if not EXAMPLES_FILE.exists():
        return []
    items = []
    for raw in EXAMPLES_FILE.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(raw)
            if item.get("source") and item.get("good"):
                items.append(item)
        except ValueError:
            continue
    return items


def build_system(target):
    """The fixed part of the prompt. It is identical for every message in the same direction, so Ollama can
    reuse its work on it (prompt caching) and only has to read the new part of each request."""
    glossary = "\n".join(load_glossary()) or "(empty)"
    return RULES.format(target=LANG_NAME[target], style=STYLE[target], glossary=glossary)


def project_note(text, src):
    """The project terms found in this message, sent just before it (after the cached part of the prompt)."""
    terms = relevant_terms(text, src or detect_source(text))
    return PROJECT_HEADER + "\n".join("%s = %s" % t for t in terms) if terms else ""


# ---------------------------------------------------------------- ollama

_think_param_ok = True
LAST_STATS = {}  # Ollama's timings for the last request (load, prompt read, generation); for the timing log
STAT_KEYS = ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration")


def estimate_tokens(text):
    """Rough, deliberately generous token count: ~1 per Japanese character, ~1 per 3.5 other characters."""
    ja = len(JA_CHARS.findall(text))
    return ja + (len(text) - ja) * 2 // 7 + 1


def ollama_up(timeout=1.5):
    try:
        with urllib.request.urlopen(HOST + "/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


_start_tried = False
_copy_remade = False


OLLAMA_APPS = (Path("/Applications/Ollama.app"), Path.home() / "Applications" / "Ollama.app")  # where install.sh puts it
# Where Homebrew and the Ollama app put the command. The hotkey runs with a minimal PATH (no /opt/homebrew/bin),
# so jt looks in these places too, not only on the PATH.
OLLAMA_COMMANDS = ("/opt/homebrew/bin/ollama", "/usr/local/bin/ollama") + tuple(
    str(a / "Contents" / "Resources" / "ollama") for a in OLLAMA_APPS)


# Ollama settings every jt measurement ran with (Homebrew's Ollama service sets them; install.sh gives them to the
# app): flash attention and an 8-bit KV cache. Same speed and accuracy, ~2 GB less memory than Ollama's defaults.
# LLAMA_ARG_CTX_CHECKPOINTS=2: Ollama's model server keeps up to 32 snapshots ("context checkpoints", ~100-130 MB each)
# per saved prompt; jt's prompt reuse needs 2. Saves ~0.5-0.8 GB, same speed and translations (measured 8 Oct 2026).
OLLAMA_SETTINGS = {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0", "LLAMA_ARG_CTX_CHECKPOINTS": "2"}


def ollama_command():
    """The `ollama` command on this Mac, or None."""
    for path in [shutil.which("ollama")] + list(OLLAMA_COMMANDS):
        if path and os.access(path, os.X_OK):
            return path
    return None


# jt never sends images, so it uses a text-only copy of the model: the same weights file without the image part
# ("CLIP model"), made once per Mac (~3 s, nothing downloaded). It saves ~0.3 GB of locked memory with identical
# translations (12 of 12 word for word, 8 Oct 2026), and Ollama 0.40 can't load gemma4:12b-it-qat's image part at all
# ("image_max_pixels (147456) is less than image_min_pixels (161280)", 7 Oct). A model without an image part, or one
# whose copy can't be made (e.g. an Ollama on another computer), is used as it is.
TEXT_ONLY = os.environ.get("JT_TEXT_ONLY", "1") != "0"  # dev/bench.py turns it off to measure models as published
_prepared = False


def text_only_name(model):
    name, _, tag = model.partition(":")
    return "%s-text:%s" % (name, tag or "latest")


def active_model():
    """The model jt sends requests to: MODEL's text-only copy once it's made (see prepare_model), else MODEL."""
    return text_only_name(MODEL) if load_settings().get("text_only_for") == MODEL else MODEL


def prepare_model():
    """Once per Mac and model, before the first request: switch to the text-only copy. Never fails: if the copy can't
    be made, the full model is used (and remembered, so it isn't tried again)."""
    global _prepared
    if _prepared or not TEXT_ONLY:
        return
    _prepared = True
    settings = load_settings()
    if settings.get("text_only_for") == MODEL or MODEL in settings.get("full_model_for", []):
        return
    try:
        modelfile = _show_modelfile()
    except (OSError, ValueError):
        return  # Ollama isn't running or the model isn't pulled yet: decide next time
    if not modelfile:
        return
    try:
        _make_text_only_copy(modelfile)
    except JtError:
        save_setting("full_model_for", sorted(set(settings.get("full_model_for", [])) | {MODEL}))
        return
    save_setting("text_only_for", MODEL)


def image_part_failed(error_body):
    low = error_body.lower()
    return "clip model" in low or "mmproj" in low


def _show_modelfile():
    with urllib.request.urlopen(urllib.request.Request(
            HOST + "/api/show", data=json.dumps({"model": MODEL}).encode("utf-8"),
            headers={"Content-Type": "application/json"}), timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8")).get("modelfile", "")


def use_text_only_copy():
    """Make the text-only copy of MODEL (if it isn't there yet) and use it from now on, on this Mac. For an Ollama
    that can't load the image part, where there is no going back to the full model: errors are shown."""
    try:
        modelfile = _show_modelfile()
    except (OSError, ValueError) as e:
        raise JtError("Ollama couldn't load the model's image part, and jt couldn't read the model to work around it: %s" % e)
    _make_text_only_copy(modelfile)
    save_setting("text_only_for", MODEL)


def _make_text_only_copy(modelfile):
    """`ollama create` the copy: MODEL's Modelfile without the smaller FROM (the image part). Raises JtError."""
    sources = re.findall(r"^FROM (.+)$", modelfile, flags=re.M)
    files = [f for f in sources if os.path.isfile(f)]
    cli = ollama_command()
    if len(files) < 2 or not cli:
        raise JtError("This version of Ollama can't load the model's image part (jt doesn't use it). Reinstall the "
                      "model with:  ollama pull %s   or install an older Ollama." % MODEL)
    weights = max(files, key=os.path.getsize)  # the language model; the small file is the image part
    lines = [l for l in modelfile.splitlines() if not (l.startswith("FROM ") and l[5:].strip() != weights)]
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / "text-only.Modelfile"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    env = dict(os.environ, OLLAMA_HOST=HOST)
    try:
        res = subprocess.run([cli, "create", text_only_name(MODEL), "-f", str(path)], capture_output=True, env=env,
                             timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        raise JtError("couldn't make the text-only copy of the model: %s" % e)
    finally:
        try:
            path.unlink()
        except OSError:
            pass
    if res.returncode != 0:
        raise JtError("couldn't make the text-only copy of the model: %s"
                      % res.stderr.decode("utf-8", "replace").strip()[-300:])


def start_ollama(wait=20):
    """Start Ollama if it isn't running (the Mac app, or `ollama serve`) and wait until it answers.
    Only for an Ollama on this computer; True if it's up."""
    if not re.match(r"https?://(127\.0\.0\.1|localhost|\[::1\])(:|$)", HOST):
        return False
    try:
        app = next((a for a in OLLAMA_APPS if a.exists()), None) if platform.system() == "Darwin" else None
        cli = ollama_command()
        if app:
            subprocess.run(["open", "-g", "-a", str(app)], capture_output=True)
        elif cli:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            env = dict(OLLAMA_SETTINGS, **os.environ)  # the user's own values win
            with open(str(STATE_DIR / "ollama.log"), "ab") as out:
                subprocess.Popen([cli, "serve"], stdout=out, stderr=out, stdin=subprocess.DEVNULL, start_new_session=True,
                                 env=env)
        else:
            return False
    except OSError:
        return False
    for _ in range(int(wait * 2)):
        if ollama_up():
            return True
        time.sleep(0.5)
    return False


def chat(messages, progress=None):
    """One request to Ollama. With a `progress` (TranslationProgress), the answer is streamed so the progress window
    can show it being written; otherwise it arrives in one piece."""
    global _think_param_ok, _start_tried, _copy_remade
    prepare_model()
    payload = {
        "model": active_model(),
        "messages": messages,
        "stream": progress is not None,
        "keep_alive": keep_alive(),
        "options": dict(LOAD_OPTIONS, temperature=0.2, seed=42),
    }
    if _think_param_ok:
        payload["think"] = False
    req = urllib.request.Request(
        HOST + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    raw = ""
    try:
        if progress is not None:
            progress.start()
        with urllib.request.urlopen(req, timeout=300) as resp:
            if progress is None:
                raw = resp.read().decode("utf-8", "replace")
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("not a JSON object")
            else:
                data = _read_stream(resp, progress)
    except ValueError:
        raise JtError("Ollama sent an unexpected reply, so nothing was translated. Try again; if it keeps happening, "
                      "restart Ollama. Reply started with: %s" % raw[:120])
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        e.close()
        if _think_param_ok and "think" in body.lower():
            _think_param_ok = False  # older Ollama or a model without a thinking switch
            return chat(messages, progress)
        if image_part_failed(body) and active_model() == MODEL:
            if progress is not None:
                progress.phase("Preparing the model (one time only)…", extra_seconds=15)
            use_text_only_copy()
            return chat(messages, progress)
        if (e.code == 404 or "not found" in body.lower()) and active_model() != MODEL and not _copy_remade:
            _copy_remade = True  # the text-only copy was deleted: make it again (once per run)
            use_text_only_copy()
            return chat(messages, progress)
        if e.code == 404 or "not found" in body.lower():
            raise JtError("model '%s' isn't installed. Run:  ollama pull %s" % (MODEL, MODEL))
        raise JtError("Ollama returned an error (%s): %s" % (e.code, body))
    except (urllib.error.URLError, ConnectionError, TimeoutError, socket.timeout):  # socket.timeout: Python 3.9
        if AUTOSTART and not _start_tried and not ollama_up():
            _start_tried = True  # once per run: start Ollama, then try again
            if progress is not None:
                progress.phase("Starting Ollama…", extra_seconds=15)
            if start_ollama():
                return chat(messages, progress)
        raise JtError("can't reach Ollama at %s. Open the Ollama app (or run `ollama serve`) and try again." % HOST)
    finally:
        if progress is not None:
            progress.stop()
    LAST_STATS.clear()
    LAST_STATS.update((k, data[k]) for k in STAT_KEYS if k in data)
    out = data.get("message", {}).get("content", "")
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()  # in case the model thinks anyway
    if not out:
        raise JtError("the model returned an empty translation, so nothing was copied (your clipboard is unchanged). "
                      "Try again.")
    return out


def _read_stream(resp, progress):
    """Read Ollama's streamed answer (one JSON object per line), reporting each piece to the progress window.
    Returns the last object with the whole text in message.content, like a non-streamed answer."""
    text, pieces, last = "", 0, {}
    for line in resp:
        line = line.strip()
        if not line:
            continue
        part = json.loads(line.decode("utf-8", "replace"))
        if not isinstance(part, dict):
            raise ValueError("not a JSON object")
        if part.get("error"):
            raise JtError("Ollama returned an error: %s" % part["error"])
        text += (part.get("message") or {}).get("content", "")
        pieces += 1
        last = part
        progress.wrote(text, pieces)  # raises JtCancelled when the user pressed Cancel
        if part.get("done"):
            progress.finished()
            break
    last = dict(last)
    last["message"] = {"role": "assistant", "content": text}
    return last


CONTEXT_WRAPPER = """[EARLIER PART OF THE SAME MESSAGE - already translated, for context only. Do NOT translate or repeat it.]
{previous}

[TRANSLATE ONLY THIS PART. Use the earlier part to resolve references such as 上記, その件, "the above", "it".]
{part}"""


def _messages_for(text_for_terms, user_content, src, tgt):
    # Order matters for speed: everything up to the last example is the same for every message, so Ollama
    # reuses it; only the terms note and the message itself are new.
    messages = [{"role": "system", "content": build_system(tgt)}]
    for ex in (c for c in load_examples() if c.get("target_lang") == tgt):
        messages.append({"role": "user", "content": ex["source"]})
        messages.append({"role": "assistant", "content": ex["good"]})
    note = project_note(text_for_terms, src)
    if note:
        messages.append({"role": "system", "content": note})
    messages.append({"role": "user", "content": user_content})
    return messages


def split_long(text, limit=CHUNK_SIZE):
    """Split at paragraph breaks; a paragraph that is still too long is split after sentence ends."""
    pieces = []
    for para in re.split(r"\n\s*\n", text.strip()):
        if estimate_tokens(para) <= limit:
            pieces.append(para)
            continue
        sentences = re.findall(r"[^。！？!?\n]+[。！？!?]*\n?|\n", para)
        buf = ""
        for s in sentences:
            if buf and estimate_tokens(buf + s) > limit:
                pieces.append(buf.strip())
                buf = ""
            buf += s
        if buf.strip():
            pieces.append(buf.strip())
    # merge small neighbours back together so each part keeps as much context as possible
    parts, cur = [], ""
    for p in pieces:
        if cur and estimate_tokens(cur + "\n\n" + p) > limit:
            parts.append(cur)
            cur = p
        else:
            cur = p if not cur else cur + "\n\n" + p
    if cur:
        parts.append(cur)
    return parts


def directions(text, target=None):
    """(source, target) language: the other language from the message's, or `target` if one is forced."""
    src = detect_source(text)
    tgt = target or ("en" if src == "ja" else "ja")
    if tgt == src:
        src = "ja" if tgt == "en" else "en"
    return src, tgt


def translate(text, target=None, window=None, title=None):
    """Translate `text`. With a ProgressWindow, the window shows progress and the translation as it's written."""
    src, tgt = directions(text, target)
    title = title or "Translating %s → %s" % (LANG_NAME[src], LANG_NAME[tgt])
    watch = (lambda part, label: TranslationProgress(window, part, tgt, label)) if window else (lambda part, label: None)
    if estimate_tokens(text) <= CHUNK_ABOVE:
        # Normal case: the whole message in one go, so the model sees all of its context.
        return chat(_messages_for(text, text, src, tgt), watch(text, title)), src, tgt
    # Very long message: part by part, each part seeing the previous one as context.
    outputs, previous, parts = [], "", split_long(text)
    for n, part in enumerate(parts, 1):
        content = CONTEXT_WRAPPER.format(previous=previous, part=part) if previous else part
        outputs.append(chat(_messages_for(part, content, src, tgt), watch(part, "%s (part %d of %d)" % (title, n, len(parts)))))
        previous = part
    return "\n\n".join(outputs), src, tgt


FIX_PROMPT = """You edit English messages for a technical work chat between Japanese and English speakers.
Correct grammar, spelling, punctuation and awkward or broken sentence structure, so the message reads as clear,
natural business English.
Rules:
- Keep the meaning exactly. Never add, drop or change facts, requests, questions, numbers, dates, names, IDs, URLs,
  file names, code or technical terms.
- Keep the tone and the level of politeness. Don't make it longer or more formal than it needs to be.
- Keep the line breaks, bullet points and blank lines.
- If the message is already correct, reply with exactly UNCHANGED and nothing else (don't repeat the message).
- Otherwise output only the corrected message. No notes, no explanations, no quotation marks around it."""

FIX_EXAMPLES = [  # (as written, corrected): the model follows examples better than rules
    ("can you checked the csv file and tell me if the arrival date are correct",
     "Can you check the CSV file and tell me if the arrival dates are correct?"),
    ("We was deploy the fix to test environment yesterday but robot still not working, please confirm in your side.",
     "We deployed the fix to the test environment yesterday, but the robot is still not working. Please confirm on your side."),
    ("Thanks, we'll take a look tomorrow.", "UNCHANGED"),
    ("Hi all,\nThe fix is deployed to the test environment. Could you check the arrival CSV import and let us know if "
     "anything looks wrong?\n\nThanks!", "UNCHANGED"),
    ("Hi,\n- order ORD-2026-00871 not showing in list\n- also the stock_qty is null for 3 items, is it expected?",
     "Hi,\n- Order ORD-2026-00871 isn't showing in the list.\n- Also, stock_qty is NULL for 3 items. Is that expected?"),
]


def needs_fixing(text):
    """Whether the hotkey runs the English correction step. Not for one-liners under SHORT_ENGLISH_WORDS ("Thanks!",
    "Got it"), and not for very long text: one request has to hold it twice (num_ctx), and it isn't split into parts."""
    short = "\n" not in text.strip() and len(text.split()) < SHORT_ENGLISH_WORDS
    return not short and estimate_tokens(text) <= CHUNK_ABOVE


def fix_english(text, window=None):
    """Correct the grammar and structure of an English message before it's translated (bilingual hotkey output).
    If the result doesn't look like a corrected version of the same message, the original is kept."""
    messages = [{"role": "system", "content": FIX_PROMPT}]
    for before, after in FIX_EXAMPLES:
        messages += [{"role": "user", "content": before}, {"role": "assistant", "content": after}]
    messages.append({"role": "user", "content": text})
    progress = TranslationProgress(window, text, "fix", "Checking the English…") if window else None
    fixed = chat(messages, progress)
    if fixed.strip().strip(".").upper() == "UNCHANGED":
        return text  # already correct: the model didn't spend time writing it out again
    plausible = (detect_source(fixed) == "en" and 0.5 <= len(fixed) / max(1, len(text)) <= 2.0
                 and fixed.count("\n") <= text.count("\n") + 2)
    return fixed if plausible else text


def bilingual_block(english, japanese):
    """English: …\nJapanese: …  (each label on its own line, and a blank line between, for multi-line messages)."""
    if "\n" in english or "\n" in japanese:
        return "English:\n%s\n\nJapanese:\n%s" % (english, japanese)
    return "English: %s\nJapanese: %s" % (english, japanese)


def translate_bilingual(text, target=None, window=None):
    """The hotkey's bilingual output. English is corrected first, then translated; Japanese is translated to English.
    Returns (block, src, tgt, translated_from, translation) — translated_from is the corrected English when src is en."""
    src, tgt = directions(text, target)
    if src == "en":
        english = fix_english(text, window) if needs_fixing(text) else text
        japanese = translate(english, "ja", window=window)[0]
        return bilingual_block(english, japanese), "en", "ja", english, japanese
    english = translate(text, "en", window=window)[0]
    return bilingual_block(english, text), "ja", "en", text, english


# ---------------------------------------------------------------- clipboard / popups

def _run(cmd, stdin_text=None):
    env = dict(os.environ)
    if platform.system() == "Darwin":
        env["LANG"] = "en_US.UTF-8"  # pbcopy/pbpaste mangle Japanese without this
    res = subprocess.run(
        cmd,
        input=stdin_text.encode("utf-8") if stdin_text is not None else None,
        capture_output=True,
        env=env,
    )
    if res.returncode != 0:
        raise OSError(res.stderr.decode("utf-8", "replace"))
    return res.stdout.decode("utf-8", "replace")


def clipboard_get():
    system = platform.system()
    try:
        if system == "Darwin":
            return _run(["pbpaste"])
        if system == "Windows":
            return _run(["powershell", "-NoProfile", "-Command",
                         "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-Clipboard -Raw"])
        if os.environ.get("WAYLAND_DISPLAY"):
            return _run(["wl-paste", "--no-newline"])
        return _run(["xclip", "-selection", "clipboard", "-o"])
    except (OSError, FileNotFoundError) as e:
        die("couldn't read the clipboard: %s" % e)


def clipboard_set(text):
    system = platform.system()
    try:
        if system == "Darwin":
            _run(["pbcopy"], text)
        elif system == "Windows":
            _run(["powershell", "-NoProfile", "-Command",
                  "[Console]::InputEncoding=[Text.Encoding]::UTF8; Set-Clipboard -Value ([Console]::In.ReadToEnd())"],
                 text)
        elif os.environ.get("WAYLAND_DISPLAY"):
            _run(["wl-copy"], text)
        else:
            _run(["xclip", "-selection", "clipboard"], text)
    except (OSError, FileNotFoundError) as e:
        die("couldn't write to the clipboard: %s" % e)


def popup(text, title="jt"):
    if platform.system() != "Darwin":
        print(text)
        return
    script = ['-e', 'on run argv',
              '-e', 'display dialog (item 1 of argv) with title (item 2 of argv) buttons {"OK"} default button 1',
              '-e', 'end run']
    subprocess.run(["osascript"] + script + [text, title], capture_output=True)


PROGRESS_JS = r"""
ObjC.import('Cocoa');
function run(argv) {
  var path = argv[0], where = argv[1] || 'corner';
  var app = $.NSApplication.sharedApplication;
  app.setActivationPolicy($.NSApplicationActivationPolicyAccessory);
  var W = where === 'center' ? 460 : 380, H = where === 'center' ? 220 : 190;
  // A floating panel that never takes the focus: you keep typing where you were.
  var style = $.NSWindowStyleMaskTitled | $.NSWindowStyleMaskUtilityWindow | $.NSWindowStyleMaskNonactivatingPanel;
  var win = $.NSPanel.alloc.initWithContentRectStyleMaskBackingDefer(
      $.NSMakeRect(0, 0, W, H), style, $.NSBackingStoreBuffered, false);
  win.title = 'jt'; win.releasedWhenClosed = false;
  win.floatingPanel = true; win.hidesOnDeactivate = false; win.becomesKeyOnlyIfNeeded = true;
  win.level = $.NSFloatingWindowLevel;
  var v = win.contentView;
  function label(y, h, size, bold, color) {
    var t = $.NSTextField.alloc.initWithFrame($.NSMakeRect(16, y, W - 32, h));
    t.editable = false; t.bezeled = false; t.drawsBackground = false; t.selectable = false;
    t.font = bold ? $.NSFont.boldSystemFontOfSize(size) : $.NSFont.systemFontOfSize(size);
    if (color) t.textColor = color;
    v.addSubview(t);
    return t;
  }
  var title = label(H - 34, 18, 12, true);
  var bar = $.NSProgressIndicator.alloc.initWithFrame($.NSMakeRect(16, H - 54, W - 32, 12));
  bar.indeterminate = false; bar.minValue = 0; bar.maxValue = 100;
  v.addSubview(bar);
  var status = label(H - 76, 16, 11, false, $.NSColor.secondaryLabelColor);
  var preview = label(44, H - 124, 12, false);
  preview.cell.wraps = true; preview.cell.truncatesLastVisibleLine = true;

  var state = {last: '', seen: false, missing: 0, start: Date.now(), cancelBtn: null};
  function finish() { win.orderOut(null); app.terminate(null); }
  function tick() {
    var s = $.NSString.stringWithContentsOfFileEncodingError(path, $.NSUTF8StringEncoding, null);
    if (Date.now() - state.start > 15 * 60 * 1000) return finish();
    if (s.isNil()) { if (state.seen && ++state.missing > 25) finish(); return; }
    state.seen = true; state.missing = 0;
    var txt = s.js;
    if (txt === state.last) return;
    state.last = txt;
    var st; try { st = JSON.parse(txt); } catch (e) { return; }
    if (st.done) return finish();
    title.stringValue = st.title || ''; bar.doubleValue = st.percent || 0;
    if (state.cancelBtn.enabled) status.stringValue = st.status || '';
    preview.stringValue = st.text || '';
  }
  // Buttons that react to the first click even though the panel isn't focused.
  ObjC.registerSubclass({name: 'JTButton', superclass: 'NSButton', methods: {
    'acceptsFirstMouse:': {types: ['bool', ['id']], implementation: function (e) { return true; }}}});
  ObjC.registerSubclass({name: 'JTActions', methods: {
    'cancel:': {types: ['void', ['id']], implementation: function (sender) {
      $('1').writeToFileAtomicallyEncodingError(path + '.cancel', true, $.NSUTF8StringEncoding, null);
      status.stringValue = 'Cancelling…'; state.cancelBtn.enabled = false; }},
    'hide:': {types: ['void', ['id']], implementation: function (sender) { finish(); }},
    'tick:': {types: ['void', ['id']], implementation: function (timer) { tick(); }}}});
  var actions = $.JTActions.alloc.init;
  function button(text, x, action) {
    var b = $.JTButton.alloc.initWithFrame($.NSMakeRect(x, 8, 84, 28));
    b.title = text; b.bezelStyle = $.NSBezelStyleRounded; b.target = actions; b.action = action;
    v.addSubview(b);
    return b;
  }
  state.cancelBtn = button('Cancel', W - 100, 'cancel:');
  button('Hide', W - 190, 'hide:');
  if (where === 'center') {
    win.center;
  } else {
    var f = $.NSScreen.mainScreen.visibleFrame;
    win.setFrameOrigin($.NSMakePoint(f.origin.x + f.size.width - W - 20, f.origin.y + f.size.height - H - 20));
  }
  win.orderFrontRegardless;
  $.NSTimer.scheduledTimerWithTimeIntervalTargetSelectorUserInfoRepeats(0.08, actions, 'tick:', null, true);
  app.run;
}
"""


class ProgressWindow:
    """A small macOS panel while the hotkey works: a progress bar, "about N s left", the translation appearing as
    it's written, Cancel and Hide. It floats in the top-right corner (or the center) and never takes the focus, so
    you can keep working; "off" shows only a notification (`jt --progress`). jt writes the state to
    ~/.jt/progress.json; the window (a JavaScript-for-Automation script run by osascript) reads it ~12 times a second
    and closes when jt is done or gone. The script runs a normal app loop (NSApplication.run + a timer reading the
    state): a hand-written event loop drew the panel but dropped mouse clicks.
    On other systems there is no window and nothing changes."""

    def __init__(self, title, where="corner"):
        self.title, self.where = title, where
        self.state = STATE_DIR / "progress.json"
        self.cancel_file = Path(str(self.state) + ".cancel")
        self.proc = None
        self._written = 0.0
        self.last = {}

    def __enter__(self):
        if platform.system() != "Darwin":
            return self
        if self.where == "off":
            notify("Translating… the result will pop up when it's ready.", title="jt")
            return self
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            if self.cancel_file.exists():
                self.cancel_file.unlink()
            self.update(title=self.title, percent=1, status="Starting…", text="", force=True)
            self.proc = subprocess.Popen(["osascript", "-l", "JavaScript", "-e", PROGRESS_JS, str(self.state), self.where],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            self.proc = None
        return self

    def update(self, force=False, **fields):
        self.last.update(fields)
        if self.where == "off" or platform.system() != "Darwin":
            return
        now = time.time()
        if not force and now - self._written < 0.08:
            return
        self._written = now
        try:
            tmp = Path(str(self.state) + ".tmp")
            tmp.write_text(json.dumps(self.last, ensure_ascii=False), encoding="utf-8")
            os.replace(str(tmp), str(self.state))
        except OSError:
            pass

    def cancelled(self):
        return self.cancel_file.exists()

    def __exit__(self, *exc):
        if not self.proc:
            return False
        self.update(force=True, done=True)
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
        for f in (self.state, self.cancel_file):
            try:
                f.unlink()
            except OSError:
                pass
        return False


class TranslationProgress:
    """Turns one streamed translation into a percentage and "about N s left".

    Reading the message gives Ollama nothing to report, so that part is timed; writing is counted in pieces (about one
    token each) against the expected length. Calibrated on the 52-message benchmark, 5 Oct 2026 (gemma4:12b-it-qat):
    a translation into English is ~0.6x the message's estimate_tokens(), into Japanese ~0.95x (±30%); reading takes
    ~1.5 s; writing runs at ~13 tokens/s. The estimate is then corrected by the speed actually measured."""

    OUT_RATIO = {"en": 0.6, "ja": 0.95, "fix": 0.8}  # "fix": corrected English is about as long as the message
    READ_SECONDS = 1.5
    WRITE_RATE = 13.0
    # Reading this much longer than expected means the Mac is swapping: Ollama's saved prompts are being brought back
    # from disk (seen 7 Oct: 30-100 s while Docker containers started or stopped). Say so instead of looking stuck.
    SLOW_AFTER = 15
    SLOW_NOTE = ("Your Mac is short on memory right now, so this one is slower (Docker containers starting or "
                 "stopping often cause this). It can take a minute or two; the result will pop up.")

    def __init__(self, window, text, tgt, title):
        self.window, self.title = window, title
        self.expected = max(6.0, estimate_tokens(text) * self.OUT_RATIO[tgt])
        self.read_seconds = self.READ_SECONDS
        self.status_prefix = "Reading the message…"
        self.t0 = self.first = None
        self._stop = threading.Event()

    def phase(self, status, extra_seconds=0):
        self.status_prefix = status
        self.read_seconds += extra_seconds

    def start(self):
        self.t0 = time.time()
        loaded = model_loaded()
        if loaded is None:
            self.phase("Starting Ollama…", 15)
        elif loaded is False:
            self.phase("Loading the model…", 8)
        self.window.update(force=True, title=self.title, percent=1, status=self._reading_status(), text="")
        self._stop.clear()
        threading.Thread(target=self._tick_while_reading, daemon=True).start()

    def _tick_while_reading(self):
        while not self._stop.wait(0.2):
            if self.first is not None:
                return
            if self.window.cancelled():
                return  # jt notices at the next piece of text
            self._reading_update()

    def _reading_update(self):
        elapsed = time.time() - self.t0
        if elapsed > self.read_seconds + self.SLOW_AFTER:
            self.window.update(percent=15.0, status="Still working, slower than usual…", text=self.SLOW_NOTE)
        else:
            self.window.update(percent=min(15.0, 15.0 * elapsed / self.read_seconds), status=self._reading_status())

    def _reading_status(self):
        left = max(0.0, self.read_seconds - (time.time() - self.t0)) + self.expected / self.WRITE_RATE
        return "%s about %s left" % (self.status_prefix, _seconds(left))

    def wrote(self, text, pieces):
        if self.window.cancelled():
            raise JtCancelled("Cancelled.")
        now = time.time()
        if self.first is None:
            self.first = now
        if pieces > self.expected * 0.95:
            self.expected = pieces * 1.15  # longer than expected: keep a little room instead of sitting at 100%
        rate = pieces / (now - self.first) if now - self.first > 0.5 and pieces > 3 else self.WRITE_RATE
        left = (self.expected - pieces) / max(rate, 1.0)
        percent = 15 + 84 * min(1.0, pieces / self.expected)
        status = "Writing… %d%% · about %s left" % (percent, _seconds(left)) if left >= 1 else "Writing… almost done"
        preview = text if len(text) <= 420 else "…" + text[-420:]
        self.window.update(title=self.title, percent=percent, status=status, text=preview)

    def finished(self):
        self.window.update(force=True, title=self.title, percent=100, status="Done")

    def stop(self):
        self._stop.set()


def _seconds(s):
    return "1 second" if s < 1.5 else "%d seconds" % round(s) if s < 90 else "%d minutes" % round(s / 60)


def model_loaded():
    """True/False if Ollama says whether the model is in memory; None if we can't tell."""
    try:
        with urllib.request.urlopen(HOST + "/api/ps", timeout=2) as resp:
            names = [m.get("name", "") for m in json.loads(resp.read().decode("utf-8")).get("models", [])]
    except Exception:
        return None
    return full_name(active_model()) in names


def full_name(model):
    """The name Ollama lists a model under ("gemma4" is "gemma4:latest")."""
    return model if ":" in model else model + ":latest"


def notify(text, title="jt"):
    if platform.system() != "Darwin":
        return
    script = ['-e', 'on run argv',
              '-e', 'display notification (item 1 of argv) with title (item 2 of argv)',
              '-e', 'end run']
    subprocess.run(["osascript"] + script + [text[:200], title], capture_output=True)


def confirm(message, ok="Translate"):
    """A Cancel/OK dialog on macOS; True elsewhere (nothing to ask with)."""
    if platform.system() != "Darwin":
        return True
    script = ['-e', 'on run argv',
              '-e', 'display dialog (item 1 of argv) with title "jt" buttons {"Cancel", (item 2 of argv)} '
                    'default button 2 cancel button 1',
              '-e', 'end run']
    return subprocess.run(["osascript"] + script + [message, ok], capture_output=True).returncode == 0


def duration_text(text):
    """A rough "about N seconds/minutes" for translating this text (Gemma writes ~10 tokens/s)."""
    seconds = 3 + estimate_tokens(text) * 1.3 / 10
    return "about %d seconds" % (round(seconds / 10) * 10) if seconds < 90 else "about %d minutes" % round(seconds / 60)


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


class HotkeyLock:
    """One hotkey translation at a time. A second press while one is running is told to wait;
    a lock left by a crashed run (its process is gone) is taken over."""

    def __enter__(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    pid = int(LOCK_FILE.read_text().strip() or 0)
                except (OSError, ValueError):
                    pid = 0
                if pid and pid != os.getpid() and _pid_alive(pid):
                    raise JtNotice("jt is still translating the previous message. Wait for its popup, then try again.")
                try:
                    LOCK_FILE.unlink()
                except OSError:
                    pass
                continue
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return self
        return self  # couldn't take the lock cleanly; translating anyway is better than refusing

    def __exit__(self, *exc):
        try:
            if LOCK_FILE.read_text().strip() == str(os.getpid()):
                LOCK_FILE.unlink()
        except OSError:
            pass
        return False


# ---------------------------------------------------------------- modes

def more_input_waiting(timeout=0.15):
    """True if more text is already waiting in the terminal (the rest of a paste), False if the user
    pressed Enter on an empty line. Always False for pipes and on Windows, where an empty line ends the input."""
    if not sys.stdin.isatty():
        return False
    try:
        import select
        return bool(select.select([sys.stdin], [], [], timeout)[0])
    except (ImportError, OSError, ValueError):
        return False


def read_block(prompt):
    """Read a multi-line message from the terminal; an empty line typed by the user ends it.

    Blank lines inside a pasted message are kept: a paste arrives all at once, so after a blank line the
    next line is already waiting, while a blank line the user types is followed by nothing."""
    print(prompt, file=sys.stderr)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            if not more_input_waiting():
                if lines:
                    break
            elif lines:
                lines.append("")
            continue
        lines.append(line)
    return "\n".join(lines).strip() if lines else None


def show_result(text, target, check):
    busy = sys.stderr.isatty()
    if busy:
        sys.stderr.write("Translating…")
        sys.stderr.flush()
    try:
        out, src, tgt = translate(text, target)
    finally:
        if busy:
            sys.stderr.write("\r" + " " * 14 + "\r")
            sys.stderr.flush()
    print(out)
    save_last(out)
    if check:
        back, _, _ = translate(out, src)
        print("\n[back-check -> %s]\n%s" % (LANG_NAME[src], back))


def mode_clipboard(target, show, check):
    global GUI
    GUI = True
    text = clipboard_get().strip()
    if not text:
        raise JtNotice("The clipboard is empty. Select a message, press ⌘C, then the hotkey.")
    last = load_last()
    if last and text in (last["translation"].strip(), last.get("copied", "").strip()):
        raise JtNotice("The clipboard still holds jt's last translation, so there's nothing new to translate.\n\n"
                       "Select the next message, press ⌘C, then the hotkey.\n\nLast translation:\n%s"
                       % last.get("copied", last["translation"]))
    if not has_words(text):
        raise JtNotice("There's nothing to translate on the clipboard (only links, numbers or code):\n\n%s" % text[:300])
    if len(text) > CONFIRM_ABOVE and not confirm(
            "The clipboard holds a long text (%d characters). Translating it takes %s.\n\nTranslate it?"
            % (len(text), duration_text(text))):
        return
    corrected = False
    started = time.time()
    fmt = output_format()
    key = cache_key(text, target, fmt)
    hit = None if check else cache_get(key)  # a back-check always runs the model
    if hit:  # translated before with exactly the same glossary, prompts and model: same answer, instantly
        clipboard_set(hit["out"])
        save_last(hit["translation"], copied=hit["out"])
        cache_touch(key)
        print(hit["out"])
        title = "jt: %s -> %s (%scopied, remembered from before)" % (
            LANG_NAME[hit["src"]], LANG_NAME[hit["tgt"]], "English corrected, " if hit.get("corrected") else "")
        if show:
            popup(hit["out"], title=title)
        else:
            notify(hit["out"], title=title)
        return
    cold = model_loaded() is not True  # the model has to be loaded: its prompts will be cold too
    try:
        with HotkeyLock(), ProgressWindow("Translating…", progress_mode()) as window:
            if fmt == "bilingual":
                out, src, tgt, translated_from, translation = translate_bilingual(text, target, window=window)
                corrected = src == "en" and translated_from.strip() != text.strip()
            else:
                translation, src, tgt = translate(text, target, window=window)
                out = translation
            # Before anything is copied, so Cancel during the back-check still leaves everything as it was.
            back = translate(translation, src, window=window,
                             title="Checking the meaning (back to %s)" % LANG_NAME[src])[0] if (show and check) else None
            clipboard_set(out)
            log_timing(time.time() - started, text, src, tgt)
            if cold:  # the prompts this translation used are warm now; warm the others while the user reads
                used = {"ja>en"} if src == "ja" else {"en>ja"}
                if src == "en" and fmt == "bilingual" and needs_fixing(text):
                    used.add("fix")
                warm_in_background(sorted(used))
            save_last(translation, copied=out)
            cache_put(key, {"out": out, "src": src, "tgt": tgt, "translation": translation, "corrected": corrected})
    except JtCancelled:
        return  # the user cancelled: nothing copied, nothing more to show
    print(out)
    note = "English corrected, " if corrected else ""
    if show:
        body = out
        if back:
            body += "\n\n[back-check -> %s]\n%s" % (LANG_NAME[src], back)
        popup(body, title="jt: %s -> %s (%scopied)" % (LANG_NAME[src], LANG_NAME[tgt], note))
    else:
        notify(out, title="jt: %stranslation copied" % note)


def check_output(test, out):
    """What's wrong with a translation according to a tests.json entry (empty list = pass)."""
    low = out.lower()
    problems = []
    for group in test.get("must", []):
        if not any(word.lower() in low for word in group):
            problems.append("missing one of: " + " / ".join(group))
    for word in test.get("must_not", []):
        if word.lower() in low:
            problems.append("should not contain: " + word)
    return problems


def mode_selftest():
    if not TESTS_FILE.exists():
        die("tests.json not found next to jt.py")
    tests = json.loads(TESTS_FILE.read_text(encoding="utf-8"))
    passed, times, kinds = 0, [], {}
    for t in tests:
        t0 = time.time()
        out = fix_english(t["source"]) if t.get("fix") else translate(t["source"])[0]  # "fix": English fixing
        times.append(time.time() - t0)
        problems = check_output(t, out)
        ok = not problems
        passed += ok
        kind = "English fixing" if t.get("fix") else "translation"
        kinds.setdefault(kind, [0, 0])
        kinds[kind][0] += ok
        kinds[kind][1] += 1
        print("%s  %4.1fs  %s: %s" % ("PASS" if ok else "FAIL", times[-1], t["id"], t.get("checks", "")))
        if not ok:
            print("      output: %s" % out.replace("\n", " "))
            for p in problems:
                print("      - " + p)
    print("\n%d/%d passed  (%s; model: %s)" % (passed, len(tests),
                                              ", ".join("%s %d/%d" % (k, v[0], v[1]) for k, v in kinds.items()), active_model()))
    if len(times) > 1:
        print("speed: first %.1fs (includes loading the model if it was not loaded), then median %.1fs per message"
              % (times[0], median(times[1:])))
    sys.exit(0 if passed == len(tests) else 1)


def mode_doctor():
    print("Ollama host : %s" % HOST)
    try:
        with urllib.request.urlopen(HOST + "/api/tags", timeout=5) as resp:
            names = [m.get("name", "") for m in json.loads(resp.read().decode("utf-8")).get("models", [])]
        print("Ollama      : running")
    except Exception:
        print("Ollama      : NOT reachable - open the Ollama app, or run `ollama serve`")
        sys.exit(1)
    wanted = full_name(MODEL)
    if wanted in names:
        print("Model       : %s installed" % MODEL)
        if active_model() != MODEL:
            print("              using its text-only copy %s (without the image part, which jt never uses: "
                  "~0.3 GB less memory, same translations)" % active_model())
    else:
        print("Model       : %s NOT installed - run: ollama pull %s" % (MODEL, MODEL))
    print("Glossary    : %d terms (%s)" % (len(load_glossary()), GLOSSARY_FILE.name))
    print("Project     : %d terms (%s, only matching ones are sent)" % (len(load_project_terms()), PROJECT_FILE.name))
    print("Format      : %s (change with: jt --format bilingual|plain)" % output_format())
    print("Keep loaded : %s after the last translation (~9 GB while loaded; change with: jt --keep-loaded 15m)" % keep_alive())
    print("Recent speed: %s" % (timing_summary() or "no hotkey translations logged yet"))
    print("Cache       : %s" % cache_summary())
    if platform.system() == "Darwin":
        print("Hotkey      : %s" % hotkey_status())
        print("Progress    : %s (change with: jt --progress corner|center|off)" % progress_mode())
        print("Memory      : %s" % memory_status())
        print("Ollama setup: %s" % ollama_settings_status())
    sys.exit(0 if wanted in names else 1)


HOTKEY_SHORTCUTS = ("Translate",)  # ⌃T. The forced-direction shortcuts went: every result has both languages


def hotkey_status():
    """Which hotkey shortcuts exist and whether the Python they run works (macOS)."""
    problems, found = [], []
    try:
        names = [n.strip() for n in
                 subprocess.run(["shortcuts", "list"], capture_output=True, text=True, timeout=20).stdout.splitlines()]
        found = [s for s in HOTKEY_SHORTCUTS if s in names]
        if "Translate" not in found:
            problems.append('no "Translate" shortcut (see README: Hotkey)')
    except (OSError, subprocess.SubprocessError):
        problems.append("couldn't list Shortcuts")
    try:
        if subprocess.run(["/usr/bin/python3", "-c", ""], capture_output=True, timeout=20).returncode != 0:
            problems.append("/usr/bin/python3 doesn't run (fix: xcode-select --install)")
    except (OSError, subprocess.SubprocessError):
        problems.append("/usr/bin/python3 is missing (fix: xcode-select --install)")
    if problems:
        return "; ".join(problems)
    return "shortcuts found: " + ", ".join('"%s"' % s for s in found)


def ollama_settings_status():
    """Whether Ollama runs the model with jt's memory settings: read from its model server's command line and
    environment (`ps -E`)."""
    try:
        lines = subprocess.run(["ps", "-axo", "pid=,args="], capture_output=True, text=True, timeout=10).stdout
        servers = [l.split(None, 1) for l in lines.splitlines() if "llama-server" in l or "ollama runner" in l]
        if not servers:
            return "can't tell until the model is loaded (run: jt --warm)"
        env = subprocess.run(["ps", "-E", "-ww", "-o", "command=", "-p", servers[0][0]], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"
    missing = []
    if not any("--cache-type-k q8_0" in s[-1] for s in servers):
        missing.append("compressed memory (~2 GB)")
    if "LLAMA_ARG_CTX_CHECKPOINTS=2" not in env:
        missing.append("fewer prompt snapshots (~0.5-0.8 GB)")
    if not missing:
        return "OK (compressed memory, fewer prompt snapshots)"
    return "missing %s: run  bash ~/jt/install.sh  again" % " and ".join(missing)


def memory_status():
    """Swap in use (macOS): above a few GB, the model gets swapped out and translations slow down."""
    swap = swap_used_gb()
    if swap is None:
        return "unknown"
    if swap >= 4:
        return ("%.1f GB of swap in use: translations may be slow at times. Closing apps or lowering Docker's memory "
                "limit (Docker Desktop > Settings > Resources) helps." % swap)
    return "OK (%.1f GB of swap in use)" % swap


# The first message through each prompt after a model load reads the whole ~700-token prompt from scratch (~7-12 s);
# after that, Ollama resumes from a saved point and the same message takes ~2 s (measured 7 Oct 2026: first Japanese
# 7.1-8.0 s cold vs 2.1 s warmed; first English with correction 11.1-12.4 s vs 4.5-5.3 s). jt has three prompts.
WARM_UPS = {"ja>en": lambda: translate("テスト", "en"),
            "en>ja": lambda: translate("Test", "ja"),
            "fix": lambda: fix_english("Test")}


def warm_prompts(skip=()):
    """Run a tiny request through each prompt not in `skip`, so the first real message of each kind is fast."""
    for key, run in WARM_UPS.items():
        if key not in skip:
            run()


def warm_in_background(skip):
    """After a hotkey translation that had to load the model, warm the other prompts in a separate process while the
    user reads the result (a translation started meanwhile waits for at most one warm-up step)."""
    try:
        subprocess.Popen([sys.executable, str(HERE / "jt.py"), "--warm-prompts"] + list(skip),
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    except OSError:
        pass  # only an optimisation


def load_model():
    """Load the model into memory with the same settings translations use (switching to the text-only copy if this
    Ollama can't load the model's image part)."""
    prepare_model()
    payload = {"model": active_model(), "messages": [], "keep_alive": keep_alive(), "options": LOAD_OPTIONS}
    req = urllib.request.Request(HOST + "/api/chat", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=300).read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        e.close()
        if image_part_failed(body) and active_model() == MODEL:
            use_text_only_copy()
            return load_model()
        raise JtError("couldn't load the model: Ollama returned an error (%s): %s" % (e.code, body))
    except Exception as e:
        raise JtError("couldn't load the model: %s" % e)


def mode_warm():
    """Load the model into memory now and warm all three prompts, so the next translation of any kind is fast."""
    t0 = time.time()
    load_model()
    loaded = time.time() - t0
    warm_prompts()
    print("Model loaded in %.1fs and warmed up in %.1fs; it stays ready for %s after the last use."
          % (loaded, time.time() - t0 - loaded, keep_alive()))


def unload_model():
    """Remove the model from memory now. Returns True if it was loaded. It reloads on the next translation."""
    was_loaded = model_loaded()
    payload = {"model": active_model(), "messages": [], "keep_alive": 0}
    req = urllib.request.Request(HOST + "/api/chat", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=60).read()
    except urllib.error.URLError:
        raise JtError("can't reach Ollama at %s, so there is nothing to unload." % HOST)
    except Exception as e:
        raise JtError("couldn't unload the model: %s" % e)
    return bool(was_loaded)


def mode_stop():
    if unload_model():
        print("Model removed from memory (about 9 GB freed). It loads again automatically on the next translation.")
    else:
        print("The model wasn't loaded, so there was nothing to free.")


# ---------------------------------------------------------------- main

def main():
    try:
        _main()
    except JtNotice as e:
        die(str(e), notice=True)
    except JtError as e:
        die(str(e))
    except Exception as e:
        log_error("unexpected error")
        die("jt hit an unexpected problem (%s: %s). Details are in %s." % (type(e).__name__, e, LOG_FILE))


def _main():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    p = argparse.ArgumentParser(prog="jt", description="Local Japanese <-> English translator (Ollama).")
    p.add_argument("text", nargs="*", help="text to translate (omit for interactive mode)")
    p.add_argument("-c", "--clipboard", action="store_true", help="translate the clipboard and copy the result back")
    p.add_argument("--show", action="store_true", help="with -c: show the translation in a popup (macOS)")
    p.add_argument("--check", action="store_true", help="also translate the result back to spot meaning drift")
    p.add_argument("--to", choices=["en", "ja"], help="force the target language")
    p.add_argument("--selftest", action="store_true", help="run the regression tests in tests.json")
    p.add_argument("--doctor", action="store_true", help="check Ollama and the model")
    p.add_argument("--warm", action="store_true", help="load the model now so the next translation is fast")
    p.add_argument("--stop", action="store_true", help="remove the model from memory now (frees ~9 GB of RAM)")
    p.add_argument("--warm-prompts", nargs="*", metavar="SKIP", help=argparse.SUPPRESS)  # internal: see warm_in_background
    p.add_argument("--clear-cache", action="store_true", help="forget the remembered hotkey translations")
    p.add_argument("--keep-loaded", metavar="TIME",
                   help="how long the model stays in memory after a translation, e.g. 15m, 1h, 8h (default; saved)")
    p.add_argument("--format", choices=FORMATS, dest="output_format",
                   help="what the hotkey copies: bilingual (English corrected + Japanese, default) or plain (translation only)")
    p.add_argument("--progress", choices=PROGRESS_MODES,
                   help="where the hotkey's progress window appears: corner (default), center, or off (saved)")
    args = p.parse_args()

    if args.warm_prompts is not None:
        return warm_prompts(skip=set(args.warm_prompts))
    if args.clear_cache:
        try:
            CACHE_FILE.unlink()
        except OSError:
            pass
        print("Remembered translations cleared.")
        return None
    if args.keep_loaded:
        if not is_duration(args.keep_loaded):
            raise JtError("--keep-loaded needs a time like 15m, 1h or 8h")
        save_setting("keep_loaded", args.keep_loaded)
        print("The model will stay in memory for %s after each translation (~9 GB while loaded), then free it. "
              "The first translation after that waits for a reload (~15-20 s)." % args.keep_loaded)
        return None
    if args.output_format:
        save_setting("format", args.output_format)
        print({"bilingual": "The hotkey will copy \"English: …\" and \"Japanese: …\" (English messages are corrected first).",
               "plain": "The hotkey will copy only the translation."}[args.output_format])
        return None
    if args.progress:
        save_setting("progress", args.progress)
        print({"corner": "The progress window will appear in the top-right corner.",
               "center": "The progress window will appear in the middle of the screen.",
               "off": "No progress window: you'll get a short notification, then the result."}[args.progress])
        return None
    if args.doctor:
        return mode_doctor()
    if args.warm:
        return mode_warm()
    if args.stop:
        return mode_stop()
    if args.selftest:
        return mode_selftest()
    if args.clipboard:
        return mode_clipboard(args.to, args.show, args.check)

    if args.text:
        show_result(" ".join(args.text), args.to, args.check)
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            show_result(text, args.to, args.check)
    else:
        print("jt - paste a message, then press Enter on an empty line. Ctrl+C to quit.", file=sys.stderr)
        try:
            while True:
                text = read_block("\n>>>")
                if text is None:
                    break
                print()
                show_result(text, args.to, args.check)
        except KeyboardInterrupt:
            print(file=sys.stderr)


if __name__ == "__main__":
    main()
