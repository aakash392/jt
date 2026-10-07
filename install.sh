#!/usr/bin/env bash
# One-time setup for jt on macOS (also works on Linux): installs Ollama if it's missing, starts it, downloads the
# model, adds the `jt` command and opens the hotkey shortcut. Safe to run again: it skips what's already done.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${JT_MODEL:-gemma4:12b-it-qat}"
HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
case "$HOST" in http*) ;; *) HOST="http://$HOST" ;; esac
OLLAMA_APP_URL="https://ollama.com/download/Ollama-darwin.zip"  # Ollama's official, signed Mac app
APPS_DIR="${JT_APPS_DIR:-/Applications}"                         # where the app goes (~/Applications if not writable)

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

install_ollama() {
  if [ "$(uname)" = "Darwin" ]; then
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

start_ollama() {
  if ollama_up; then return 0; fi
  echo "Starting Ollama..."
  local app
  if app="$(find_app)"; then
    open -g -a "$app"  # in the background; its window can be closed, it keeps running in the menu bar
  else
    mkdir -p "$HOME/.jt"
    nohup "$OLLAMA" serve >>"$HOME/.jt/ollama.log" 2>&1 &
  fi
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
