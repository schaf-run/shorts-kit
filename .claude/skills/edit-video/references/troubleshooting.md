# Если что-то пошло не так

## vid edit остановился или шаг нужно переделать
- `NEXT: ... cuts.json` / `NEXT: ... composition.json`: запиши этот файл (edit-rules §1 / §2-4) и запусти ту же команду `tools/vid edit ПУТЬ`. Каждый шаг перезапускается, только если его входы новее результата: правка cuts.json переделывает нарезку и рендер, правка composition.json или brand/theme.css - только рендер.
- `--from ШАГ` (init, transcribe, takes, mistakes, cut, compose, sheet) принудительно переделывает шаг и всё после него, например `--from transcribe` после смены `--lang`.
- Рабочая папка по умолчанию: `videos/<имя файла>.work`. Другая: `-w ПАПКА`.
- Отдельные шаги вручную: `tools/vid mistakes WORK`, `tools/vid cut WORK [--cuts F]`, `tools/vid compose WORK [composition.json] [--keep-tmp]`, `tools/vid say WORK "фраза" --cut`, `tools/vid icons СЛОВО`.

## Результат выглядит не так
- Кружок обрезает лицо: рамка кружка - медианное лицо (детектор YuNet) по всему cut.mp4; второй человек в кадре или отсутствие лица её ломают (compose предупредит «no face found»). Масштаб кружка - `k1` в talking-head.html.
- Слово в субтитрах написано неправильно: `fix` в composition.json (ключ - как в cut.txt). Пропало слово: проверь слова в cut.json и вырезы рядом.
- Обрезан конец слова на стыке: расширь край диапазона в cuts.json до начала следующего слова; куски короче 0.2 с выбрасываются.
- Рендер падает с ошибкой HDR или диска: клип должен быть SDR (vid cut сам переводит HDR с iPhone в SDR); см. `lessons.md`.
- Рендер медленный: ~3-4x реального времени на обычном процессоре; длинные ролики - минуты, запускай в фоне.
- Custom-бит сломан: `tools/vid compose WORK --keep-tmp` оставит `tools/hyperframes/.tmp/<hash>.html` и `.json` для проверки; `npx hyperframes snapshot` в `tools/hyperframes` отрисует отдельные кадры.
- Неверный язык или странные слова: `tools/vid edit ПУТЬ --lang ru --from transcribe`. Для английской речи используется Whisper; подсказка словаря: `tools/vid transcribe WORK --engine whisper --prompt "Claude, RAG"`.

## Цвета или шрифты не те
- Стиль берётся из `brand/theme.css` при каждом рендере. `tools/vid doctor` проверит, что все переменные на месте и файлы шрифтов лежат в `brand/fonts/`.
- Хардкод цвета (`#c0282c`) в composition.json или slides.json не меняется вместе со стилем: замени на `var(--accent-hi)` и т.п.

## Установка
- `tools/vid doctor` - что не так; скилл setup - как починить.
