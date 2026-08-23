"""
stage 1 · 텍스트 전처리

- URL 제거, # 기호 제거, 공백 정규화
- description 상투 문구 제거 (저작권 고지, 타임코드 목차, 채움 문자 등)
- title + description 결합
- 불용어 집합 (DEFAULT_STOPWORDS)
"""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+")
_HASHTAG_MARK_RE = re.compile(r"#")
_WHITESPACE_RE = re.compile(r"\s+")

_COPYRIGHT_LINE_RE = re.compile(
    r"^.*(ⓒ|©|무단\s*전재|재배포|저작권).*$", re.MULTILINE
)
_TIMECODE_LINE_RE = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s+.*$", re.MULTILINE)
_FILLER_CHAR_RE = re.compile(r"[ㅤ​]")  # 한글 채움 문자, zero-width space

DEFAULT_STOPWORDS: set[str] = {
    # 유튜브 UI / 포맷 용어
    "영상", "구독", "채널", "댓글", "좋아요", "알림", "설정", "안내", "공지",
    "생방송", "라이브", "속보", "자막",
    "Shorts", "shorts", "SHORTS", "Short",
    "zip", "ZIP", "Zip",
    "LIVE", "Live",
    # 플랫폼 / SNS 상투어 (description 내 반복 문구)
    "유튜브", "페이스북", "링크", "동영상", "편성",
    # 뉴스 보도 상투어
    "뉴스", "오늘", "이번", "투데이",
    "출처", "저작권", "무단", "전재", "재배포", "금지",
    "이용", "학습", "제공", "촬영", "편집",
    "방송", "중계", "이유", "상황", "관련", "내용", "문제",
    # 방송사 채널명 / 뉴스 프로그램명
    "MBC뉴스", "KBS뉴스", "SBS뉴스", "MBC", "KBS", "SBS", "YTN", "JTBC",
    "채널A", "TV조선", "연합뉴스", "뉴스1", "뉴시스",
    "와글와글", "이슈톡", "Pick", "PD수첩",
    "뉴스투데이", "뉴스데스크", "뉴스외전", "뉴스특보", "뉴스플러스",
    "newsdesk", "newstoday", "MBCNEWS", "KBSNEWS",
    "데스크", "수첩","PD",
    # 날씨/시간 일반 명사
    "날씨", "오전", "오후", "내일", "어제", "주말", "주간",
}


def clean_text(text: str | None) -> str:
    """URL·#기호 제거 후 공백 정규화."""
    if not text:
        return ""
    text = _URL_RE.sub(" ", text)
    text = _HASHTAG_MARK_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def strip_boilerplate(description: str | None) -> str:
    """description에서 저작권 고지, 타임코드 목차 줄, 채움 문자 등 상투 문구를 제거."""
    if not description:
        return ""
    text = _FILLER_CHAR_RE.sub(" ", description)
    text = _COPYRIGHT_LINE_RE.sub(" ", text)
    text = _TIMECODE_LINE_RE.sub(" ", text)
    return text


def combined_text(title: str | None, description: str | None) -> str:
    """title + 전처리된 description을 하나의 문자열로 합쳐 반환."""
    return f"{clean_text(title)} {clean_text(strip_boilerplate(description))}".strip()
