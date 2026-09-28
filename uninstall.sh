#!/usr/bin/env bash
# Удаление всего, что поставил install.sh. Запуск: bash uninstall.sh
# Спрашивает перед каждой группой. Системные пакеты (ffmpeg, библиотеки, Homebrew) и Claude Code
# не трогает: ими могут пользоваться другие программы - в конце подскажет, как удалить их вручную.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
ask() { local a; read -r -p "$1 [y/N] " a; [ "$a" = "y" ] || [ "$a" = "Y" ] || [ "$a" = "д" ] || [ "$a" = "Д" ]; }
size() { du -sh "$@" 2>/dev/null | awk '{s = s $1 " "} END {print s}'; }
rmdir_safe() { for d in "$@"; do [ -n "$d" ] && [ "$d" != "/" ] && [ "$d" != "$HOME" ] && [ -e "$d" ] && rm -rf "$d"; done; }

HF="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
MODELS=(models--istupakov--gigaam-v3-onnx models--istupakov--silero-vad-onnx
        models--Systran--faster-whisper-base models--mobiuslabsgmbh--faster-whisper-large-v3-turbo)

say "1. Модели распознавания речи ($HF)"
present=(); for m in "${MODELS[@]}"; do [ -d "$HF/$m" ] && present+=("$HF/$m"); done
if [ ${#present[@]} -gt 0 ]; then
  echo "Найдено: ${present[*]##*/}  ($(size "${present[@]}"))"
  echo "Если другие программы на этом компьютере используют эти же модели, они скачают их заново."
  if ask "Удалить модели?"; then
    if [ -x tools/.venv/bin/python ]; then
      # через API Hugging Face: он удалит и общие файлы моделей (blobs)
      tools/.venv/bin/python - "${present[@]##*/}" <<'EOF'
import sys
from huggingface_hub import scan_cache_dir
ids = {m[len("models--"):].replace("--", "/", 1) for m in sys.argv[1:]}
c = scan_cache_dir()
revs = [rev.commit_hash for r in c.repos if r.repo_id in ids for rev in r.revisions]
if revs:
    c.delete_revisions(*revs).execute()
EOF
    fi
    rmdir_safe "${present[@]}"
    echo "Удалено."
  fi
else
  echo "Не найдены - пропускаю."
fi

say "2. Chrome для отрисовки (~/.cache/hyperframes)"
if [ -d "$HOME/.cache/hyperframes" ]; then
  echo "Размер: $(size "$HOME/.cache/hyperframes")"
  ask "Удалить?" && rmdir_safe "$HOME/.cache/hyperframes" && echo "Удалено."
else
  echo "Не найден - пропускаю."
fi

say "3. Node.js, который поставил установщик (~/.local/node)"
if [ -d "$HOME/.local/node" ]; then
  echo "Размер: $(size "$HOME/.local/node")"
  if ask "Удалить?"; then
    for b in node npm npx; do [ -L "$HOME/.local/bin/$b" ] && rm -f "$HOME/.local/bin/$b"; done
    rmdir_safe "$HOME/.local/node"
    echo "Удалено."
  fi
else
  echo "Установщик Node.js не ставил (использовался системный) - пропускаю."
fi

say "4. Менеджер Python uv и его кэш"
if [ -x "$HOME/.local/bin/uv" ] || [ -d "$HOME/.cache/uv" ]; then
  echo "~/.local/bin/uv, ~/.cache/uv ($(size "$HOME/.cache/uv")), Python в ~/.local/share/uv ($(size "$HOME/.local/share/uv"))"
  echo "Если вы пользуетесь uv для других проектов - ответьте N."
  if ask "Удалить uv?"; then
    rm -f "$HOME/.local/bin/uv" "$HOME/.local/bin/uvx"
    rmdir_safe "$HOME/.cache/uv" "$HOME/.local/share/uv"
    echo "Удалено."
  fi
else
  echo "Не найден - пропускаю."
fi

say "5. Кэш npm (~/.npm)"
if [ -d "$HOME/.npm" ]; then
  echo "Размер: $(size "$HOME/.npm"). Если вы пользуетесь npm для других проектов - ответьте N."
  ask "Удалить кэш npm?" && rmdir_safe "$HOME/.npm" && echo "Удалено."
fi

say "6. Папка проекта ($ROOT)"
echo "Здесь окружение (tools/.venv, npm-пакеты) и ВАША РАБОТА: videos/ (исходники и ролики), carousels/, brand/ (стиль)."
echo "Размер: $(size "$ROOT")"
if ask "Удалить папку проекта целиком, со всеми роликами и каруселями?"; then
  cd "$HOME" && rmdir_safe "$ROOT" && echo "Удалено."
elif ask "Удалить только окружение (tools/.venv, npm-пакеты, модель лица), а работу оставить?"; then
  rmdir_safe "$ROOT/tools/.venv" "$ROOT/tools/hyperframes/node_modules" "$ROOT/tools/models" \
             "$ROOT/tools/hyperframes/.tmp" "$ROOT/tools/hyperframes/assets/brand"
  echo "Удалено. Вернуть: bash install.sh"
fi

say "Что осталось (удаляется вручную, если больше не нужно)"
if grep -qi microsoft /proc/version 2>/dev/null; then
  echo "- Весь Linux целиком вместе со всем выше: в PowerShell (Windows) выполните: wsl --unregister Ubuntu"
  echo "- Системные пакеты: sudo apt-get remove ffmpeg   (библиотеки Chrome безопасно оставить)"
elif [ "$(uname -s)" = "Darwin" ]; then
  echo "- ffmpeg и node: brew uninstall ffmpeg node"
  echo "- Homebrew целиком: https://github.com/Homebrew/install#uninstall-homebrew"
else
  echo "- Системные пакеты: sudo apt-get remove ffmpeg"
fi
echo "- Claude Code: rm ~/.local/bin/claude && rm -rf ~/.local/share/claude ~/.claude   (~/.claude - настройки и история всех проектов)"
echo "- Строка PATH в ~/.bashrc или ~/.zshrc: export PATH=\"\$HOME/.local/bin:\$PATH\" - можно оставить"
