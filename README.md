# jt - local Japanese ⇄ English translator

Translates text between Japanese and English in any app on your Mac: chat, email, documents, the browser.
Everything runs on your own Mac. Nothing is sent to the internet.
It runs Google's `gemma4:12b-it-qat` model (Gemma 4 12B) through Ollama and uses a shared project glossary,
so terms like 反映, ロボット and カラム always come out the same way.

**It is a draft, not a final answer.** Before acting on a translation or sending one,
read it once. For anything important, use `--check`.

**User guide:** https://jt-guide.vercel.app/ (install, hotkey setup, every command, troubleshooting).
**日本語のガイド：** https://jt-guide.vercel.app/?lang=ja

## Install (macOS, 16 GB RAM or more)

1. In Terminal, download jt into `~/jt` (the hotkey expects it there) and run the setup:
   ```bash
   git clone https://github.com/aakash392/jt.git ~/jt
   bash ~/jt/install.sh
   ```
   The setup does everything: it installs Ollama if you don't have it (the official Mac app from ollama.com, about
   200 MB), starts it, downloads the model (about 7.2 GB, first time only), adds a `jt` command and opens the hotkey
   shortcut. Running it again is safe: it skips what's already done.
2. Open a new Terminal window and run `jt --doctor`. Everything should say installed/running.

## Use it

```bash
jt "在庫が0の場合でも変更しないでください。"     # Japanese → English
jt "Please don't deploy until it's tested."      # English → Japanese
jt --check "..."                                 # also translates back, to catch meaning drift
jt                                               # interactive: paste, then press Enter on an empty line
```

The direction is detected automatically. Use `--to en` or `--to ja` to force it.

## Hotkey: translate in any app (the main way to use jt)

Works in any app where you can copy text, with no browser and no Terminal. The jt folder must be at `~/jt`.
Set it up once:

1. In the **Shortcuts** app: **Shortcuts → Settings → Advanced → Allow Running Scripts** (on).
2. Add the ready-made shortcut: `install.sh` opens it, or double-click `~/jt/hotkey/Translate.shortcut`,
   and click **Add Shortcut**.
3. Give it its key: select **Translate** → **ⓘ** (details) → **Add Keyboard Shortcut** → press **⌃T** (Control+T).
   `jt --doctor` checks that the shortcut is there.

Without the file, make **Translate** by hand: **+** → action **Run Shell Script** →
`/usr/bin/python3 "$HOME/jt/jt.py" --clipboard --show`.

In any app: select the text → **⌘C** → **⌃T**. A popup shows the result, and it's already copied, so **⌘V**
pastes it. The result has both languages:

```
English: Could you check if the arrival CSV import works now?
Japanese: 入荷CSVのインポートが現在動作するかご確認いただけますでしょうか。
```

- **Replying:** write your reply in English, copy it, press ⌃T. jt first **corrects your English** (grammar,
  spelling, sentence structure; never the meaning, names, numbers, IDs or code), then translates the corrected
  English. You paste both lines. The popup title says "English corrected" when something was changed.
- **Reading:** a Japanese message gives the English translation, with the original Japanese below it.
- Multi-line messages put each label on its own line, with a blank line between the two parts.
- Prefer the old way (only the translation, no correcting)? `jt --format plain`; back with `jt --format bilingual`.


Line breaks and blank lines are kept, so multi-point messages stay readable.

A message you've translated before comes back **instantly** (jt remembers its last 500 results, about 1 MB, in
`~/.jt/`; the popup says "remembered from before"). It's forgotten automatically when the glossary, examples or model
change; `jt --clear-cache` empties it. Very short English ("Thanks!", "Got it") skips the correction step.

While it works, a small panel in the **top-right corner** shows a **progress bar**, **"about N seconds left"**,
and the translation **appearing as it's written**. It never takes the focus, so you can keep typing.
**Hide** puts it away for this translation (the result still pops up at the end); **Cancel** stops it and leaves
your clipboard as it was. The time left is an estimate from the message's length and Gemma's measured speed, so it
can be off by a few seconds.

Prefer it elsewhere? `jt --progress center` (middle of the screen), `jt --progress off` (no panel, just a short
notification), `jt --progress corner` (back to the default). It's saved for you only.

The hotkey looks after you:
- **Forgot ⌘C?** If the clipboard still holds jt's last translation, it tells you instead of translating it back.
- **Nothing to translate** (only a link, numbers or code): it says so.
- **Pressed twice?** The second press tells you the first is still translating; nothing runs twice.
- **Copied something huge** (over ~2,000 characters)? It asks before starting a long translation.
- **Ollama not running?** It starts it and carries on (allow ~20 seconds).
- **Something went wrong?** You always get a popup. Unexpected problems are logged in `~/.jt/jt.log`.

The first run asks for permission; allow it. If you want the back-check in the popup
too, add `--check` to the command (slower).

## Shared glossary (`glossary.txt`)

One term per line, `Japanese = English`. It applies in both directions.
When a translation gets a project term wrong, add it here. Keep the list focused
(roughly 50–100 terms); a small model starts ignoring very long lists.
Changes take effect immediately, with no rebuild needed.

## Project terms (`project-glossary.txt`)

About 300 terms (screen names, statuses, field names, carriers), each checked by hand,
plus the core vocabulary that's always translated the same way:
入荷 = arrival, 入庫 = in-stock, 格納 = put-away, 出荷 = shipping, 出庫 = stock-out,
引当 = allocation, 荷主 = shipper, 棚卸 = stocktaking, 区分 = type, 移動 = movement.

jt sends the model **only the terms that appear in the message** (up to 25, longest first),
so the list can grow to thousands without slowing down or confusing the model. It works in
both directions: English messages are matched on the English side.

- Add or fix a term: edit the file (`Japanese = English`). Changes apply on the next translation.

## Free up memory when you need it

While loaded, the model holds about 7.6 GB of RAM (for 8 hours after the last use). Before something
memory-heavy (Docker, large builds, screen-sharing in a big call), remove it from memory:

```bash
jt --stop                 # or: ollama stop gemma4:12b-it-qat
```

Nothing breaks: the next translation loads it again
automatically (about 15–20 seconds).

While loaded, the model's memory is **pinned** (macOS can't swap it out), so with Docker running everything else gets
less room. To free it automatically a while after your last translation, instead of after 8 hours:
`jt --keep-loaded 15m` (or `30m`, `1h`; back with `jt --keep-loaded 8h`). The trade-off: the first translation after
that waits for a reload (~15–20 s). Lowering Docker Desktop's memory limit (Settings → Resources) helps too.

If translations ever seem slow or stuck, `jt --doctor` shows your recent speeds ("Recent speed", from
`~/.jt/timing.log`: every hotkey translation with its time and how much of the model was in memory). `ollama ps` shows whether it is loaded. Shutting down also frees it.

## Found a bad translation?

Send it to Aakash: the original message and what jt gave you. The fix goes into the shared glossary or examples,
is checked against the self-tests so nothing else gets worse, and reaches everyone with `git pull`.

## Before changing the model or glossary: `jt --selftest`

Runs the known tricky messages in `tests.json` and reports PASS/FAIL. Run it after
editing the glossary or switching models, so you notice if something got worse.
Add a test whenever you find a new kind of mistake.

## Getting updates

```bash
cd ~/jt && git pull
```

This gives you the latest jt, glossary, examples and tests (no reinstall needed).

## Other systems

- **Windows/Linux:** install Ollama, run `ollama pull gemma4:12b-it-qat`, then `python jt.py ...`.
  The clipboard mode works too: it uses PowerShell on Windows and needs `xclip` or
  `wl-clipboard` on Linux. The popup is macOS-only; elsewhere the translation is printed.
- **Different model:** `JT_MODEL=qwen3.5:9b jt "..."` (the previous default: a bit faster, 1.7 GB less RAM,
  less accurate). Run `jt --selftest` to compare.

## Troubleshooting

| Message | Fix |
|---|---|
| can't reach Ollama | jt tries to start it; if that fails, open the Ollama app (menu bar icon) |
| Translations are sometimes slow (10–20 s) | `jt --doctor` shows swap in use. The Mac is short on memory and the model was swapped out: close apps or lower Docker's memory limit |
| model isn't installed | `ollama pull gemma4:12b-it-qat` |
| "Failed to load CLIP model" | Your Ollama can't load the model's image part (Ollama 0.40 with this model). Update jt (`cd ~/jt && git pull`): it then makes a text-only copy of the model once (about 20 seconds, nothing downloaded) and uses that. |
| Japanese shows as garbage in the hotkey | Make sure the command uses `/usr/bin/python3` and the `jt.py` from this folder |
| Slow first translation | Normal: the model loads into memory (a few seconds), then it stays ready for 8 hours after the last use (about 7.6 GB of RAM while loaded; freed after 8 idle hours or at shutdown). Run `jt --warm` to load it ahead of time. Memory tight? `jt --keep-loaded 30m` |
