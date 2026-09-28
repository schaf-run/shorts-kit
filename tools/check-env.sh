#!/usr/bin/env bash
# Быстрая проверка окружения при старте сессии Claude (хук SessionStart). Молчит, если всё на месте.
cd "$(dirname "$0")/.." || exit 0
missing=()
command -v ffmpeg >/dev/null || missing+=("ffmpeg")
command -v node >/dev/null || [ -x "$HOME/.local/bin/node" ] || missing+=("Node.js")
[ -x tools/.venv/bin/python ] || missing+=("Python-окружение")
[ -d tools/hyperframes/node_modules/hyperframes ] || missing+=("HyperFrames")
[ -f tools/models/face_detection_yunet_2023mar.onnx ] || missing+=("модель лица")
if [ ${#missing[@]} -gt 0 ]; then
  echo "ОКРУЖЕНИЕ НЕ УСТАНОВЛЕНО (нет: ${missing[*]}). Первым делом скажи пользователю об этом простыми словами и запусти скилл setup."
fi
exit 0
