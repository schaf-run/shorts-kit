"""Builds slides.json for the example carousel (run: python3 gen_slides.py).
Shows the building blocks: .lbl, .h1/.h2/.body/.small, .mark, .ser, .card(.hot), .chip, .accent,
Lucide icons (<i data-icon>) and an SVG scheme drawn with the theme tokens."""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slides.json")
SVG_CSS = """
svg { display: block; overflow: visible; }
svg text { font-family: var(--font-sans); fill: var(--text); font-weight: 600; font-size: 28px; }
svg .mu { fill: var(--muted); } svg .ac { fill: var(--accent-hi); } svg .b8 { font-weight: 800; }
svg .mid { text-anchor: middle; }
svg .frame { fill: var(--panel); stroke: var(--edge); stroke-width: 3; }
svg .zone { fill: var(--accent); opacity: 0.85; }
svg .head { fill: var(--panel-2); stroke: var(--text); stroke-width: 3; }
svg .dash { stroke: var(--accent-hi); stroke-width: 3; stroke-dasharray: 12 10; fill: none; }
"""


def phone(x, head_y, r, label, bad):
    """A 9:16 phone frame: the speaker's head (radius r) at head_y, the caption band at y 420-480."""
    return (f'<rect class="frame" x="{x}" y="0" width="360" height="640"/>'
            f'<rect class="head" x="{x + 180 - r * 1.6}" y="{head_y + r + 10}" width="{r * 3.2}" height="{630 - head_y - r}"/>'
            f'<circle class="head" cx="{x + 180}" cy="{head_y}" r="{r}"/>'
            f'<rect class="zone" x="{x + 30}" y="420" width="300" height="60"/>'
            f'<text class="mid b8" x="{x + 180}" y="460">СУБТИТРЫ</text>'
            f'<text class="mid {"ac" if bad else "mu"} b8" x="{x + 180}" y="700">{label}</text>')


slides = [
    {"html": '<div class="lbl">Съёмка · 3 правила</div>'
             '<div class="h1">Снимите так, чтобы монтаж занял <span class="mark">минуты</span></div>'
             '<div class="body muted">Три настройки до нажатия «запись», которые экономят час на монтаже.</div>'},
    {"html": '<div class="lbl">1 · Кадр</div>'
             '<div class="h2">Отойдите на шаг: <span class="ser">место под субтитры</span></div>'
             '<svg viewBox="0 0 920 720" width="920" height="720">'
             + phone(40, 330, 130, "ТЕКСТ НА ЛИЦЕ", True) + phone(520, 170, 70, "ЛИЦО ОТКРЫТО", False) +
             '</svg>',
     "css": SVG_CSS},
    {"html": '<div class="lbl">2 · Звук</div>'
             '<div class="h2">Звук важнее картинки</div>'
             '<div class="col">'
             '<div class="card row"><i data-icon="mic" class="accent"></i><div><div class="body">Петличка или телефон ближе</div>'
             '<div class="small">чем ближе микрофон, тем меньше эха</div></div></div>'
             '<div class="card row"><i data-icon="volume-x" class="accent"></i><div><div class="body">Тихая комната</div>'
             '<div class="small">шум сложнее всего убрать потом</div></div></div>'
             '<div class="card hot row"><i data-icon="scissors"></i><div><div class="body">Ошиблись — повторите фразу целиком</div>'
             '<div class="small" style="color: var(--text)">монтаж найдёт дубль и вырежет лишнее</div></div></div>'
             '</div>'},
    {"html": '<div class="lbl">Итог</div>'
             '<div class="h2">Перед записью проверьте</div>'
             '<div class="row" style="flex-wrap: wrap">'
             '<span class="chip">9:16, вертикально</span><span class="chip">голова в верхней трети</span>'
             '<span class="chip">микрофон близко</span><span class="chip hot">ошибка → повтор фразы</span></div>'
             '<div class="body muted">Сохраните, чтобы не забыть перед следующей съёмкой.</div>',
     "chrome": True},
]
json.dump({"next": "листай", "ratio": "4:5", "slides": slides}, open(OUT, "w"), ensure_ascii=False, indent=1)
print(OUT)
