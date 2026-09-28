#!/usr/bin/env bash
# Установка всего, что нужно для монтажа и каруселей. Запуск: bash install.sh
# Можно запускать повторно: уже установленное пропускается.
#   --user-only   без sudo: только то, что ставится в папку пользователя (Python-окружение,
#                 npm-пакеты, модели); системные пакеты должны уже стоять.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
USER_ONLY=0
[ "${1:-}" = "--user-only" ] && USER_ONLY=1

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$1"; }
fail() { printf '\033[31mОшибка: %s\033[0m\n' "$1"; exit 1; }

mkdir -p "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"

OS="$(uname -s)"
WSL=0
grep -qi microsoft /proc/version 2>/dev/null && WSL=1

if [ "$WSL" = 1 ] && [[ "$ROOT" == /mnt/* ]]; then
  warn "Папка проекта лежит на диске Windows ($ROOT). Так всё будет работать очень медленно."
  warn "Лучше скачать проект внутрь Ubuntu: cd ~ && git clone ... (см. README)."
  read -r -p "Продолжить всё равно? [y/N] " a
  [ "$a" = "y" ] || exit 1
fi

# ---------- системные пакеты (нужен пароль) ----------
node_ok() { command -v node >/dev/null && [ "$(node -p 'process.versions.node.split(".")[0]')" -ge 18 ]; }

if [ "$USER_ONLY" = 0 ]; then
  if [ "$OS" = "Linux" ]; then
    command -v apt-get >/dev/null || fail "нужен Ubuntu/Debian (apt-get). В Windows используйте WSL с Ubuntu, см. README."
    say "Системные пакеты (попросит пароль от Ubuntu)"
    sudo apt-get update -q
    # библиотеки для встроенного Chrome, который рисует кадры; в Ubuntu 24.04 часть называется ...t64
    pkgs=(ffmpeg curl ca-certificates git xz-utils fonts-liberation libnss3 libnspr4 libdrm2 libxkbcommon0
          libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 libcairo2
          "libatk1.0-0t64|libatk1.0-0" "libatk-bridge2.0-0t64|libatk-bridge2.0-0" "libcups2t64|libcups2"
          "libasound2t64|libasound2" "libglib2.0-0t64|libglib2.0-0")
    sel=()
    for alt in "${pkgs[@]}"; do
      IFS='|' read -ra opts <<< "$alt"
      for p in "${opts[@]}"; do
        if apt-cache policy "$p" 2>/dev/null | grep -q 'Candidate: [0-9]'; then sel+=("$p"); break; fi
      done
    done
    sudo apt-get install -y -q "${sel[@]}"
  elif [ "$OS" = "Darwin" ]; then
    if ! command -v brew >/dev/null; then
      say "Homebrew (менеджер программ для macOS, попросит пароль)"
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv)"
    fi
    say "ffmpeg и Node.js"
    brew install ffmpeg node
  else
    fail "эта система не поддерживается ($OS). Нужны Ubuntu (в том числе WSL в Windows) или macOS."
  fi
fi

command -v ffmpeg >/dev/null || fail "не найден ffmpeg; запустите bash install.sh без --user-only"

# ---------- Node.js 18+ (если в системе старый - ставим свой в ~/.local) ----------
if ! node_ok; then
  say "Node.js 22 в ~/.local/node"
  arch="$(uname -m)"; case "$arch" in x86_64) arch=x64 ;; aarch64|arm64) arch=arm64 ;; esac
  plat="linux"; [ "$OS" = "Darwin" ] && plat="darwin"
  ver="$(curl -fsSL https://nodejs.org/dist/latest-v22.x/SHASUMS256.txt | grep -o "node-v22[0-9.]*-$plat-$arch.tar.xz" | head -1)"
  [ -n "$ver" ] || fail "не удалось узнать версию Node.js (нет интернета?)"
  rm -rf "$HOME/.local/node" && mkdir -p "$HOME/.local/node"
  curl -fsSL "https://nodejs.org/dist/latest-v22.x/$ver" | tar -xJ -C "$HOME/.local/node" --strip-components 1
  for b in node npm npx; do ln -sf "$HOME/.local/node/bin/$b" "$HOME/.local/bin/$b"; done
  node_ok || fail "Node.js не запускается"
fi

# ---------- Python-окружение ----------
say "Python 3.12 и библиотеки (распознавание речи, работа с видео)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
[ -x tools/.venv/bin/python ] || "$UV" venv -q --python 3.12 tools/.venv
"$UV" pip install -q --python tools/.venv/bin/python -r tools/requirements.txt

say "Модель поиска лица"
mkdir -p tools/models
[ -f tools/models/face_detection_yunet_2023mar.onnx ] || curl -fsSL -o tools/models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

say "HyperFrames (отрисовка кадров) и иконки"
(cd tools/hyperframes && npm ci --silent --no-audit --no-fund)
say "Chrome для отрисовки (скачивается один раз)"
(cd tools/hyperframes && HYPERFRAMES_NO_TELEMETRY=1 npx hyperframes browser ensure)

say "Модели распознавания речи (около 1 ГБ, один раз)"
tools/vid fetch-models

# ---------- Claude Code ----------
if ! command -v claude >/dev/null; then
  say "Claude Code"
  curl -fsSL https://claude.ai/install.sh | bash
fi

# ~/.local/bin в PATH для новых окон терминала
for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
  [ -f "$rc" ] || continue
  grep -q '.local/bin' "$rc" || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$rc"
done

say "Проверка"
tools/vid doctor && printf '\n\033[32mГотово.\033[0m Откройте новое окно терминала, затем:\n  cd %s\n  claude\n' "$ROOT"
