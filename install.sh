#!/usr/bin/env bash
# One-time setup for jt on macOS (also works on Linux).
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${JT_MODEL:-gemma4:12b-it-qat}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Download it from https://ollama.com/download,"
  echo "open it once, then run this script again."
  exit 1
fi
if ! ollama list >/dev/null 2>&1; then
  echo "Ollama is installed but not running. Open the Ollama app, then run this script again."
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. On macOS run:  xcode-select --install"
  exit 1
fi

echo "Downloading $MODEL (about 7.2 GB, first time only)..."
ollama pull "$MODEL"

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
