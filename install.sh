#!/usr/bin/env bash
# One-time setup for jt on macOS (also works on Linux): installs Ollama if it's missing (Homebrew's service when
# Homebrew is there, otherwise the official app), gives it jt's memory settings, starts it, downloads the model, adds
# the `jt` command and opens the hotkey shortcut. Safe to run again: it skips what's already done.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${JT_MODEL:-gemma4:12b-it-qat}"
HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
case "$HOST" in http*) ;; *) HOST="http://$HOST" ;; esac
OLLAMA_APP_URL="https://ollama.com/download/Ollama-darwin.zip"  # Ollama's official, signed Mac app
APPS_DIR="${JT_APPS_DIR:-/Applications}"                         # where the app goes (~/Applications if not writable)
# jt's Ollama settings: flash attention and an 8-bit compressed KV cache (what Homebrew's Ollama service runs with:
# ~2 GB less memory than Ollama's defaults), plus at most 8 prompt snapshots ("context checkpoints") instead of 32,
# so Ollama's cache can't grow past ~1 GB. Same speed and translations (measured 7-8 Oct 2026).
SETTINGS_AGENT="$HOME/Library/LaunchAgents/com.jt.ollama-settings.plist"
SETTINGS_HINT="OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 LLAMA_ARG_CTX_CHECKPOINTS=8 ollama serve"
BREW="$(command -v brew || true)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. On macOS run:  xcode-select --install   then run this script again."
  exit 1
fi

ollama_up() { curl -fsS -m 3 "$HOST/api/tags" >/dev/null 2>&1; }

find_app() {
  for app in "$APPS_DIR/Ollama.app" "$HOME/Applications/Ollama.app"; do
    if [ -d "$app" ]; then echo "$app"; return 0; fi
  done
  return 1
}

# The `ollama` command: on the PATH (Homebrew, Linux), or the one inside the Mac app.
find_ollama() {
  if command -v ollama >/dev/null 2>&1; then command -v ollama; return 0; fi
  local app
  if app="$(find_app)" && [ -x "$app/Contents/Resources/ollama" ]; then echo "$app/Contents/Resources/ollama"; return 0; fi
  return 1
}

brew_ollama() { [ -n "$BREW" ] && "$BREW" list --formula ollama >/dev/null 2>&1; }

# Ollama (the app or Homebrew's service) reads its settings from launchd. A small login item sets them at every login
# and restarts whichever Ollama is already running without them; loading it now applies them straight away.
ollama_settings() {
  local app="$1"
  mkdir -p "$(dirname "$SETTINGS_AGENT")"
  cat > "$SETTINGS_AGENT" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.jt.ollama-settings</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/sh</string>
    <string>-c</string>
    <string>launchctl setenv OLLAMA_FLASH_ATTENTION 1; launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0; launchctl setenv LLAMA_ARG_CTX_CHECKPOINTS 8; if pgrep -xq Ollama; then osascript -e 'quit app "Ollama"'; sleep 3; open -g -a "$app"; fi; for s in sh.brew.ollama homebrew.mxcl.ollama; do launchctl kickstart -k gui/$(id -u)/\$s 2&gt;/dev/null; done; true</string>
  </array>
  <key>RunAtLoad</key><true/>
</dict>
</plist>
EOF
  launchctl bootout "gui/$(id -u)" "$SETTINGS_AGENT" >/dev/null 2>&1 || true
  launchctl bootstrap "gui/$(id -u)" "$SETTINGS_AGENT"
  echo "Gave Ollama jt's memory settings (login item: $SETTINGS_AGENT)."
}

install_ollama() {
  if [ "$(uname)" = "Darwin" ] && [ -n "$BREW" ]; then
    echo "Installing Ollama with Homebrew..."
    "$BREW" install ollama
  elif [ "$(uname)" = "Darwin" ]; then
    echo "Installing Ollama (the official Mac app from ollama.com, about 200 MB)..."
    local tmp dest
    tmp="$(mktemp -d)"
    curl -fL --progress-bar -o "$tmp/Ollama.zip" "$OLLAMA_APP_URL"
    ditto -x -k "$tmp/Ollama.zip" "$tmp"
    dest="$APPS_DIR"
    if [ ! -w "$dest" ]; then dest="$HOME/Applications"; mkdir -p "$dest"; fi
    rm -rf "$dest/Ollama.app"
    mv "$tmp/Ollama.app" "$dest/"
    rm -rf "$tmp"
    echo "Installed $dest/Ollama.app (it starts at login and updates itself)."
  else
    echo "Installing Ollama (the official install script from ollama.com)..."
    curl -fsSL https://ollama.com/install.sh | sh
  fi
}

start_ollama() {  # starts the Ollama chosen below ($KIND: app, brew or other) and waits until it answers
  if ollama_up; then return 0; fi
  echo "Starting Ollama..."
  case "$KIND" in
    app)  open -g -a "$APP" ;;  # in the background; its window can be closed, it keeps running in the menu bar
    brew) ;;                    # Homebrew's service was just started: only wait for it
    *)    mkdir -p "$HOME/.jt"
          OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 nohup "$OLLAMA" serve >>"$HOME/.jt/ollama.log" 2>&1 & ;;
  esac
  for _ in $(seq 1 60); do
    if ollama_up; then return 0; fi
    sleep 1
  done
  return 1
}

case "$HOST" in
  http://127.0.0.1*|http://localhost*|http://\[::1\]*)
    if ! OLLAMA="$(find_ollama)"; then
      install_ollama
      OLLAMA="$(find_ollama)" || { echo "Ollama didn't install. Get it from https://ollama.com/download, then run this again."; exit 1; }
    fi
    KIND=other
    if [ "$(uname)" = "Darwin" ]; then
      # Exactly one Ollama, with jt's settings: the app if it's in use or installed, else Homebrew's service
      # (which has the settings built in). Never both: they would fight over the same port.
      if APP="$(find_app)" && { pgrep -xq Ollama || ! brew_ollama; }; then
        ollama_settings "$APP"
        KIND=app
      elif brew_ollama; then
        "$BREW" services start ollama >/dev/null
        ollama_settings ""
        KIND=brew
        echo "Ollama runs as Homebrew's background service."
      else
        echo "Note: this Ollama isn't the app or Homebrew's, so jt can't give it its memory settings."
        echo "      When jt starts it, it passes them itself; to set them always: $SETTINGS_HINT"
      fi
    else
      echo "Tip: for about 2 GB less memory, run Ollama with OLLAMA_FLASH_ATTENTION=1 and OLLAMA_KV_CACHE_TYPE=q8_0"
      echo "     (sudo systemctl edit ollama, then add those two lines under [Service] as Environment=...)."
    fi
    if ! start_ollama; then
      echo "Ollama is installed but didn't start. Open the Ollama app (or run: ollama serve), then run this again."
      exit 1
    fi
    ;;
  *)  # an Ollama on another computer: use it as it is
    OLLAMA="$(find_ollama)" || { echo "Ollama's command line is needed to download the model: https://ollama.com/download"; exit 1; }
    ;;
esac
echo "Ollama is running."

echo "Downloading $MODEL (about 7.2 GB, first time only)..."
"$OLLAMA" pull "$MODEL"

# Load it once now, so any one-time preparation happens here rather than on the first hotkey press (for example,
# Ollama 0.40 can't load this model's image part, so jt makes a text-only copy of it).
echo "Loading the model once to check that it works (about a minute)..."
python3 "$DIR/jt.py" --warm || echo "It didn't load yet. jt will try again on the first translation."

chmod +x "$DIR/jt.py"

case "$(basename "${SHELL:-zsh}")" in
  bash) RC="$HOME/.bashrc" ;;
  *)    RC="$HOME/.zshrc" ;;
esac
# Replace any older jt alias (e.g. one pointing at ~/Downloads/jt), so `jt` always runs this copy.
if grep -q "alias jt=" "$RC" 2>/dev/null; then
  cp "$RC" "$RC.jt-backup"
  grep -v "alias jt=" "$RC.jt-backup" > "$RC" || true
fi
echo "alias jt='python3 \"$DIR/jt.py\"'" >> "$RC"
echo "Set the 'jt' command in $RC (points to $DIR)"

# The hotkey shortcut runs "$HOME/jt/jt.py", so jt has to live in ~/jt for it to work.
if [ "$DIR" != "$HOME/jt" ]; then
  echo
  echo "Note: jt is in $DIR, but the hotkey shortcut expects $HOME/jt."
  echo "      Move this folder to ~/jt (or edit the path in the shortcut) so the hotkey works."
fi

# The ready-made, signed hotkey shortcut: opening it asks to add it.
if [ "$(uname)" = "Darwin" ] && ls "$DIR"/hotkey/*.shortcut >/dev/null 2>&1; then
  echo
  echo "Adding the hotkey shortcut: click \"Add Shortcut\" in the window that opens."
  echo "(First turn on Shortcuts > Settings > Advanced > Allow Running Scripts, or it can't run jt.)"
  for f in "$DIR"/hotkey/*.shortcut; do open "$f"; sleep 2; done
  echo "Then give it a key: in Shortcuts, select \"Translate\" > (i) Details > Add Keyboard Shortcut > Control+T."
fi

echo
python3 "$DIR/jt.py" --doctor || true
echo
echo "Done. Open a new Terminal window and try:  jt \"在庫が0の場合でも変更しないでください。\""
echo "Hotkey setup (if no shortcut was added above): see 'Hotkey' in README.md."
