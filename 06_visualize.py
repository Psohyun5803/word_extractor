"""
stage 6 · 날짜별 워드 클라우드 (HTML 출력)

CLI:
  python 06_visualize.py --date 2026-08-09
  python 06_visualize.py --date 2026-08-09 --out my_cloud.html
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

_DIR        = Path(__file__).parent
DATA_PATH   = _DIR / "data" / "outputs" / "daily_keywords.json"
FIGURES_DIR = _DIR / "data" / "outputs" / "figures"


def wordcloud_day(date: str, out: str | None = None) -> Path:
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    pairs = data.get(date, [])
    if not pairs:
        raise ValueError(f"데이터 없음: {date}")

    y, m, d = date.split("-")
    dow = ["월","화","수","목","금","토","일"][datetime(int(y), int(m), int(d)).weekday()]
    pairs_js = json.dumps(pairs, ensure_ascii=False)

    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>워드 클라우드 {date}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@700&family=Noto+Sans+KR:wght@400;500;600;700&display=swap">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ height: 100%; background: #111009; color: #ede8e0;
              font-family: 'Noto Sans KR', sans-serif; overflow: hidden; }}
header {{ padding: 18px 28px 12px; border-bottom: 1px solid #2e2820; }}
.date-main {{ font-family: 'Noto Serif KR', serif; font-size: clamp(22px,4vw,40px);
              font-weight: 700; line-height: 1.1; }}
.date-sub  {{ font-size: 12.5px; color: #7a7368; margin-top: 5px; }}
#cloud {{ position: relative; width: 100%; height: calc(100vh - 90px); overflow: hidden; }}
.w {{ position: absolute; white-space: nowrap; cursor: default; user-select: none;
      font-family: 'Noto Sans KR', sans-serif; line-height: 1; transition: opacity .12s; }}
.w:hover {{ opacity: .6; }}
.r1   {{ color: #e03b2f; }}
.r2   {{ color: #e03b2f; opacity: .78; }}
.r3   {{ color: #e03b2f; opacity: .58; }}
.rmid {{ color: #ede8e0; }}
.rlow {{ color: #7a7368; }}
#tip {{ position: fixed; background: #ede8e0; color: #111009; padding: 4px 10px;
        border-radius: 3px; font-size: 12px; pointer-events: none; opacity: 0;
        transition: opacity .08s; z-index: 99; white-space: nowrap; }}
</style>
</head><body>
<header>
  <div class="date-main">{y}년 {int(m)}월 {int(d)}일</div>
  <div class="date-sub">{dow}요일 &nbsp;·&nbsp; {len(pairs)}개 키워드</div>
</header>
<div id="cloud"></div>
<div id="tip"></div>
<script>
const PAIRS = {pairs_js};
const tip   = document.getElementById('tip');
const cloud = document.getElementById('cloud');
const mc    = document.createElement('canvas');
const ctx   = mc.getContext('2d');
let fontsReady = false;
document.fonts.ready.then(() => {{ fontsReady = true; }});

function mw(text, size, weight) {{
  if (fontsReady) {{
    ctx.font = weight + ' ' + size + 'px "Noto Sans KR",sans-serif';
    return ctx.measureText(text).width * 1.08 + 10;
  }}
  let w = 0;
  for (const c of text) w += c.charCodeAt(0) > 0x2E80 ? size * .95 : size * .58;
  return w * 1.12 + 10;
}}

function overlaps(x, y, w, h, pl) {{
  for (const p of pl)
    if (x-3 < p.x+p.w && x+w+3 > p.x && y-3 < p.y+p.h && y+h+3 > p.y) return true;
  return false;
}}

function render() {{
  cloud.innerHTML = '';
  const cW = cloud.offsetWidth, cH = cloud.offsetHeight;
  const cx = cW / 2, cy = cH / 2;
  const maxC = PAIRS[0][1];
  const placed = [], frag = document.createDocumentFragment();

  PAIRS.forEach(([text, count], i) => {{
    const t    = Math.log(count + 1) / Math.log(maxC + 1);
    const size = Math.round(13 + t * 68);
    const wt   = i < 3 ? '700' : i < 10 ? '600' : i < 20 ? '500' : '400';
    const cls  = i === 0 ? 'r1' : i === 1 ? 'r2' : i === 2 ? 'r3' : i < 12 ? 'rmid' : 'rlow';
    const tw   = mw(text, size, wt), th = size * 1.38;

    for (let s = 0; s <= 3000; s++) {{
      const angle = s * .42, r = s * 2.2;
      const x = cx + r * Math.cos(angle) - tw / 2;
      const y = cy + r * Math.sin(angle) - th / 2;
      if (!overlaps(x, y, tw, th, placed)) {{
        placed.push({{ x, y, w: tw, h: th }});
        const el = document.createElement('span');
        el.className = 'w ' + cls;
        el.textContent = text;
        el.style.cssText = `left:${{x}}px;top:${{y}}px;font-size:${{size}}px;font-weight:${{wt}}`;
        el.addEventListener('mousemove', e => {{
          tip.textContent = text + '  ' + count + '개 문서';
          tip.style.cssText = `left:${{e.clientX+12}}px;top:${{e.clientY-28}}px;opacity:1`;
        }});
        el.addEventListener('mouseleave', () => tip.style.opacity = 0);
        frag.appendChild(el);
        break;
      }}
    }}
  }});
  cloud.appendChild(frag);
}}

document.fonts.ready.then(render);
setTimeout(render, 600);
let t; window.addEventListener('resize', () => {{ clearTimeout(t); t = setTimeout(render, 120); }});
</script>
</body></html>"""

    FIGURES_DIR.mkdir(exist_ok=True)
    path = Path(out) if out else FIGURES_DIR / f"wordcloud_{date}.html"
    path.write_text(html, encoding="utf-8")
    print(f"저장: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="날짜별 워드 클라우드 HTML 생성")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out",  default=None,  help="출력 경로 (기본: figures/wordcloud_DATE.html)")
    args = parser.parse_args()
    wordcloud_day(args.date, out=args.out)
