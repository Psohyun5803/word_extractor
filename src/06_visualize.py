"""
stage 6 · 날짜별 워드 클라우드 (HTML 출력)

CLI:
  python 06_visualize.py --date 2026-08-09
  python 06_visualize.py --date 2026-08-09 --out my_cloud.html
  python 06_visualize.py --all
  python 06_visualize.py --all --out wordcloud_all.html
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_DIR          = Path(__file__).parent.parent   # word_extractor/
DATA_PATH     = _DIR / "data" / "outputs" / "daily_keywords.json"
FIGURES_DIR   = _DIR / "data" / "outputs" / "figures"
TEMPLATES_DIR = Path(__file__).parent / "templates"

_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
_env.policies["json.dumps_kwargs"] = {"ensure_ascii": False, "sort_keys": False}

DAYS_KO = ["월", "화", "수", "목", "금", "토", "일"]


def wordcloud_all(out: str | None = None) -> Path:
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    dates = sorted(data.keys())
    html = _env.get_template("wordcloud_all.html.jinja").render(
        dates=dates, data=data, cloud_height_offset=86,
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    path = Path(out) if out else FIGURES_DIR / "wordcloud_all.html"
    path.write_text(html, encoding="utf-8")
    print(f"저장: {path}  ({len(dates)}일)")
    return path


def wordcloud_day(date: str, out: str | None = None) -> Path:
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    pairs = data.get(date, [])
    if not pairs:
        raise ValueError(f"데이터 없음: {date}")

    y, m, d = date.split("-")
    dow = DAYS_KO[datetime(int(y), int(m), int(d)).weekday()]
    html = _env.get_template("wordcloud_day.html.jinja").render(
        date=date, year=y, month=int(m), day=int(d), dow=dow,
        pairs=pairs, cloud_height_offset=90,
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    path = Path(out) if out else FIGURES_DIR / f"wordcloud_{date}.html"
    path.write_text(html, encoding="utf-8")
    print(f"저장: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="날짜별 워드 클라우드 HTML 생성")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--date", metavar="YYYY-MM-DD", help="특정 날짜 단일 생성")
    grp.add_argument("--all",  action="store_true",  help="전체 날짜 슬라이더 HTML 생성")
    parser.add_argument("--out", default=None, help="출력 경로")
    args = parser.parse_args()
    if args.all:
        wordcloud_all(out=args.out)
    else:
        wordcloud_day(args.date, out=args.out)
