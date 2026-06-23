"""
render_subs.py — TikTok-Style Karaoke Subtitle Renderer
========================================================
Renders word-by-word highlighted subtitles onto a video.
Subtitles can be loaded from an ASS file OR generated directly
from the video using Faster-Whisper (no external file needed).

New in this version:
  - Integrated Faster-Whisper transcription (no ASS file required)
  - Highlight mode: "Box" (original) or "Color" (word changes colour, no box)
  - Words-per-line selector (2–10 words)
  - Expanded style presets (10 presets)
  - Highlight colour also used as word colour in Color mode
  - Improved word timing: next-word start used as current-word end
  - Language selector for transcription
  - Model selector (tiny / base / small / medium / large-v2 / large-v3)
"""

from __future__ import annotations

import os
import re
import platform
import threading
import tempfile
import time
from contextlib import contextmanager
from typing import Callable, Optional

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip
from tkinter import filedialog, messagebox


# ─────────────────────────────────────────────────────────────
#  Constants & presets
# ─────────────────────────────────────────────────────────────

STYLE_PRESETS: dict[str, dict] = {
    "Purple Haze":    {"hex": "#8A2BE2", "alpha": 230, "text": "white"},
    "TikTok Yellow":  {"hex": "#FFD700", "alpha": 240, "text": "auto"},
    "Neon Green":     {"hex": "#39FF14", "alpha": 220, "text": "auto"},
    "Ocean Blue":     {"hex": "#1E90FF", "alpha": 230, "text": "white"},
    "Hot Pink":       {"hex": "#FF1493", "alpha": 230, "text": "white"},
    "Snow White":     {"hex": "#FFFFFF", "alpha": 255, "text": "black"},
    "Crimson Red":    {"hex": "#DC143C", "alpha": 230, "text": "white"},
    "Electric Cyan":  {"hex": "#00FFFF", "alpha": 220, "text": "black"},
    "Sunset Orange":  {"hex": "#FF6B35", "alpha": 235, "text": "white"},
    "Mint Fresh":     {"hex": "#00E5A0", "alpha": 225, "text": "black"},
}

TEXT_COLOR_OPTIONS = ("white", "black", "auto")

HIGHLIGHT_MODES = ("Box", "Color")
"""
Box   — active word gets a filled rounded-rectangle behind it (original style).
Color — no box; the active word is rendered in the highlight colour,
        inactive words are rendered dimmed white/grey.
"""

# Quality presets: (label_key, crf, preset)
QUALITY_OPTIONS = [
    ("q_visually_lossless", 18, "fast"),
    ("q_high",              22, "fast"),
    ("q_balanced",          26, "medium"),
    ("q_small",             30, "medium"),
]

WHISPER_MODELS = (
    # Standard multilingual models
    "Systran/faster-whisper-tiny",
    "Systran/faster-whisper-base",
    "Systran/faster-whisper-small",
    "Systran/faster-whisper-medium",
    "Systran/faster-whisper-large-v1",
    "Systran/faster-whisper-large-v2",
    "Systran/faster-whisper-large-v3",
    # English-only models
    "Systran/faster-whisper-tiny.en",
    "Systran/faster-whisper-base.en",
    "Systran/faster-whisper-small.en",
    "Systran/faster-whisper-medium.en",
    # Distil-Whisper (distilled / faster)
    "Systran/faster-distil-whisper-small.en",
    "Systran/faster-distil-whisper-medium.en",
    "Systran/faster-distil-whisper-large-v2",
    "Systran/faster-distil-whisper-large-v3",
)

WHISPER_LANGUAGES = (
    "Auto-detect",
    "uk", "en", "de", "fr", "es", "it", "pt", "ru", "pl",
    "nl", "cs", "tr", "ja", "zh", "ko", "ar", "vi", "sv",
)

MIN_WORD_DURATION = 0.03

PREVIEW_SIZE = (540, 200)
PREVIEW_SAMPLE_WORDS = {
    "English": [
        {"text": "This",    "start": 0.0,  "end": 0.4},
        {"text": "is",      "start": 0.4,  "end": 0.65},
        {"text": "a",       "start": 0.65, "end": 0.85},
        {"text": "live",    "start": 0.85, "end": 1.3},
        {"text": "preview", "start": 1.3,  "end": 2.0},
        {"text": "of",      "start": 2.0,  "end": 2.3},
        {"text": "style",   "start": 2.3,  "end": 3.0},
    ],
    "Українська": [
        {"text": "Це",        "start": 0.0,  "end": 0.45},
        {"text": "живий",     "start": 0.45, "end": 0.9},
        {"text": "перегляд",  "start": 0.9,  "end": 1.6},
        {"text": "стилю",     "start": 1.6,  "end": 2.1},
        {"text": "субтитрів", "start": 2.1,  "end": 3.0},
    ],
}
PREVIEW_ACTIVE_INDEX = {
    "English": 3,      # "live"
    "Українська": 2,   # "перегляд"
}

# ─────────────────────────────────────────────────────────────
#  i18n – UI language strings
# ─────────────────────────────────────────────────────────────

UI_LANGUAGES = ("English", "Українська")

_STRINGS: dict[str, dict[str, str]] = {
    "English": {
        "title":            "🎵  TikTok Karaoke Subtitle Renderer",
        "subtitle":         "Generate subtitles from video via Whisper",
        "video_file":       "Video file:",
        "output_file":      "Output file:",
        "browse":           "Browse",
        "whisper_section":  "Whisper Transcription",
        "model":            "Model:",
        "language":         "Language:",
        "words_per_line":   "Words/line:",
        "words_unit":       "{} words",
        "transcribe_btn":   "🎙 Transcribe",
        "transcribing":     "⏳ Transcribing…",
        "no_transcription": "No transcription yet. Click 'Transcribe'.",
        "transcribe_done":  "Transcription complete: {} lines, {} words.",
        "transcribe_fail":  "Transcription failed: {}",
        "words_changed":    "Words/line changed — re-transcribe to apply.",
        "new_video":        "New video selected — transcribe before rendering.",
        "highlight_mode":   "Highlight mode:",
        "style_preset":     "Style preset:",
        "font":             "Font:",
        "font_search":      "Search font…",
        "font_size":        "Font size:",
        "px":               "{} px",
        "pct":              "{}%",
        "highlight_color":  "Highlight colour:",
        "box_opacity":      "Box opacity:",
        "inactive_opacity": "Inactive opacity:",
        "vertical_pos":     "Vertical position:",
        "text_color":       "Text colour:",
        "drop_shadow":      "Drop shadow:",
        "stroke_width":     "Stroke width:",
        "stroke_color":     "Stroke colour:",
        "live_preview":     "Live Preview",
        "refresh":          "↺  Refresh",
        "render_btn":       "▶  Render",
        "rendering":        "⏳  Rendering…",
        "open_folder":      "📂 Open Folder",
        "log_label":        "Log / Errors:",
        "ui_language":      "Interface language:",
        "err_no_video":     "Please select a valid video file first.",
        "err_no_output":    "Please specify an output path.",
        "err_no_font":      "Font '{}' was not found.\nSearch for it in the font search box above.",
        "no_subs_q":        "No transcription found.\n\nTranscribe the video now with Whisper before rendering?\n(This may take several minutes.)",
        "auto_transcribe":  "Auto-transcribing before render…",
        "transcribe_fail2": "Transcription failed",
        "render_done":      "Done ✅",
        "render_fail":      "Render failed ❌",
        "loading_fonts":    "Scanning system fonts…",
        "fonts_loaded":     "Found {} fonts.",
        "ready":            "Ready to render",
        "failed":           "Failed",
        "preview_status":   "Live preview  ·  mode: {}  ·  active word highlighted",
        "preview_select":   "Select a valid font to see preview.",
        "preview_error":    "Preview error: {}",
        "preview_wait":     "Preview will appear after fonts load.",
        "preview_render":   "Rendering preview…",
        "font_no_match":    "(no match)",
        # Quality
        "quality_label":    "Output quality:",
        "quality_hint":     "CRF 18 = visually lossless  ·  CRF 28 = smallest file",
        "q_visually_lossless": "🏆 Visually Lossless (CRF 18)",
        "q_high":           "🎮 High Quality (CRF 22)",
        "q_balanced":       "⚖️ Balanced (CRF 26)",
        "q_small":          "📦 Small File (CRF 30)",
        # Subtitle editor
        "edit_subs_title":  "Edit Transcription",
        "edit_subs_hint":   "Each line = one subtitle block. Words separated by spaces. Timings preserved.",
        "edit_subs_save":   "✅ Save & Close",
        "edit_subs_cancel": "Cancel",
        "edit_subs_warn":   "Edited transcription saved.",
        "edit_btn":         "✏️ Edit Subtitles",
        "no_subs_edit":     "Transcribe the video first before editing.",
        "preset_hint":      "(applies colour + text colour)",
        "color_hint":       "Box fill colour  ·  or active-word colour in Color mode",
        "text_color_hint":  '"auto" picks black/white for best contrast (Box mode)',
    },
    "Українська": {
        "title":            "🎵  Рендерер субтитрів у стилі TikTok",
        "subtitle":         "Генерація субтитрів через Whisper",
        "video_file":       "Відеофайл:",
        "output_file":      "Вихідний файл:",
        "browse":           "Огляд",
        "whisper_section":  "Транскрипція Whisper",
        "model":            "Модель:",
        "language":         "Мова:",
        "words_per_line":   "Слів у рядку:",
        "words_unit":       "{} слів",
        "transcribe_btn":   "🎙 Транскрибувати",
        "transcribing":     "⏳ Транскрибую…",
        "no_transcription": "Транскрипції немає. Натисніть 'Транскрибувати'.",
        "transcribe_done":  "Транскрипція готова: {} рядків, {} слів.",
        "transcribe_fail":  "Помилка транскрипції: {}",
        "words_changed":    "Кількість слів змінено — виконайте транскрипцію знову.",
        "new_video":        "Нове відео — транскрибуйте перед рендерингом.",
        "highlight_mode":   "Режим підсвітки:",
        "style_preset":     "Стиль (пресет):",
        "font":             "Шрифт:",
        "font_search":      "Пошук шрифту…",
        "font_size":        "Розмір шрифту:",
        "px":               "{} пкс",
        "pct":              "{}%",
        "highlight_color":  "Колір підсвітки:",
        "box_opacity":      "Прозорість рамки:",
        "inactive_opacity": "Прозорість неактивних:",
        "vertical_pos":     "Вертикальне положення:",
        "text_color":       "Колір тексту:",
        "drop_shadow":      "Тінь:",
        "stroke_width":     "Ширина обводки:",
        "stroke_color":     "Колір обводки:",
        "live_preview":     "Попередній перегляд",
        "refresh":          "↺  Оновити",
        "render_btn":       "▶  Рендер",
        "rendering":        "⏳  Рендерю…",
        "open_folder":      "📂 Відкрити папку",
        "log_label":        "Лог / Помилки:",
        "ui_language":      "Мова інтерфейсу:",
        "err_no_video":     "Виберіть дійсний відеофайл.",
        "err_no_output":    "Вкажіть шлях до вихідного файлу.",
        "err_no_font":      "Шрифт '{}' не знайдено.\nШукайте у полі пошуку шрифтів.",
        "no_subs_q":        "Транскрипція не знайдена.\n\nВиконати транскрипцію зараз перед рендерингом?\n(Може зайняти кілька хвилин.)",
        "auto_transcribe":  "Автотранскрипція перед рендерингом…",
        "transcribe_fail2": "Помилка транскрипції",
        "render_done":      "Готово ✅",
        "render_fail":      "Помилка рендерингу ❌",
        "loading_fonts":    "Сканування системних шрифтів…",
        "fonts_loaded":     "Знайдено {} шрифтів.",
        "ready":            "Готово до рендерингу",
        "failed":           "Помилка",
        "preview_status":   "Попередній перегляд  ·  режим: {}  ·  активне слово підсвічено",
        "preview_select":   "Виберіть шрифт для перегляду.",
        "preview_error":    "Помилка перегляду: {}",
        "preview_wait":     "Перегляд з'явиться після завантаження шрифтів.",
        "preview_render":   "Рендеринг перегляду…",
        "font_no_match":    "(немає збігів)",
        # Quality
        "quality_label":    "Якість виводу:",
        "quality_hint":     "CRF 18 = візуально без втрат  ·  CRF 28 = найменший файл",
        "q_visually_lossless": "🏆 Без втрат (CRF 18)",
        "q_high":           "🎮 Висока якість (CRF 22)",
        "q_balanced":       "⚖️ Баланс (CRF 26)",
        "q_small":          "📦 Малий файл (CRF 30)",
        # Subtitle editor
        "edit_subs_title":  "Редагування транскрипції",
        "edit_subs_hint":   "Кожен рядок = один субтитр. Слова через пробіл. Тайминги збережено.",
        "edit_subs_save":   "✅ Зберегти та закрити",
        "edit_subs_cancel": "Скасувати",
        "edit_subs_warn":   "Транскрипцію відредаговано і збережено.",
        "edit_btn":         "✏️ Редагувати субтитри",
        "no_subs_edit":     "Спочатку виконайте транскрипцію.",
        "preset_hint":      "(застосовує колір + колір тексту)",
        "color_hint":       "Колір заливки рамки  ·  або колір активного слова у режимі Color",
        "text_color_hint":  '"auto" обирає чорний/білий для найкращого контрасту (режим Box)',
    },
}


def _t(lang: str, key: str, *args) -> str:
    """Return translated string, formatting with *args if provided."""
    text = _STRINGS.get(lang, _STRINGS["English"]).get(key, key)
    return text.format(*args) if args else text


# ─────────────────────────────────────────────────────────────
#  Font helpers
# ─────────────────────────────────────────────────────────────

def get_system_fonts() -> dict[str, str]:
    font_dirs: list[str] = []
    system = platform.system()
    if system == "Windows":
        windir = os.environ.get("WINDIR", "C:\\Windows")
        font_dirs = [
            os.path.join(windir, "Fonts"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts"),
        ]
    elif system == "Darwin":
        font_dirs = [
            "/Library/Fonts", "/System/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
        ]
    else:
        font_dirs = [
            "/usr/share/fonts", "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/.local/share/fonts"),
        ]

    fonts: dict[str, str] = {}
    for d in font_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fname in files:
                if fname.lower().endswith((".ttf", ".otf")):
                    full = os.path.join(root, fname)
                    name = os.path.splitext(fname)[0]
                    fonts[name] = full

    return dict(sorted(fonts.items(), key=lambda x: x[0].lower()))


_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _get_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont | None:
    key = (font_path, font_size)
    if key not in _font_cache:
        try:
            _font_cache[key] = ImageFont.truetype(font_path, font_size)
        except (IOError, OSError):
            return None
    return _font_cache[key]


# ─────────────────────────────────────────────────────────────
#  Colour helpers
# ─────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b, alpha)


def _auto_text_color(bg_rgba: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r, g, b = bg_rgba[0] / 255, bg_rgba[1] / 255, bg_rgba[2] / 255
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (0, 0, 0, 255) if luminance > 0.45 else (255, 255, 255, 255)


# ─────────────────────────────────────────────────────────────
#  ASS parser (kept for backward-compat / manual load)
# ─────────────────────────────────────────────────────────────

def parse_ass_karaoke(ass_filepath: str) -> list[list[dict]]:
    """
    Parses an ASS karaoke file with {\\kN} tags.
    Returns list of lines; each line is a list of word dicts
    {"text", "start", "end"}.
    """
    subtitle_lines: list[list[dict]] = []

    with open(ass_filepath, "r", encoding="utf-8") as fh:
        raw = fh.readlines()

    for raw_line in raw:
        if not raw_line.startswith("Dialogue:"):
            continue

        parts = raw_line.strip().split(",", 9)
        if len(parts) < 10:
            continue

        start_str = parts[1].strip()
        text_part = parts[9]

        try:
            h, m, s = start_str.split(":")
            line_start = int(h) * 3600 + int(m) * 60 + float(s)
        except ValueError:
            continue

        text_part = text_part.replace("\\N", " ").replace("\\n", " ")
        text_part = re.sub(r"\{(?!\\k\d)[^}]*\}", "", text_part)

        word_matches = re.findall(r"\{\\k(\d+)\}([^\{]*)", text_part)

        current_time = line_start
        line_words: list[dict] = []

        for dur_cs, word_text in word_matches:
            dur_sec = int(dur_cs) / 100.0
            cleaned = word_text.strip()
            if cleaned and dur_sec > 0:
                line_words.append({
                    "text": cleaned,
                    "start": current_time,
                    "end": current_time + dur_sec,
                })
            current_time += dur_sec

        if not line_words:
            continue

        validated = _validate_word_timings(line_words)
        if validated:
            subtitle_lines.append(validated)

    return subtitle_lines


# ─────────────────────────────────────────────────────────────
#  Whisper transcription → subtitle lines
# ─────────────────────────────────────────────────────────────

def _detect_device() -> str:
    """Try GPU first; silently fall back to CPU on any failure."""
    try:
        import torch
        if torch.cuda.is_available():
            # Quick validation – allocate a tiny tensor to confirm CUDA works
            try:
                torch.zeros(1).cuda()
                return "cuda"
            except Exception:
                pass
    except ImportError:
        pass
    try:
        import ctranslate2
        types = ctranslate2.get_supported_compute_types("cuda")
        if types and len(types) > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def transcribe_to_subtitle_lines(
    video_path: str,
    model_name: str = "base",
    language: Optional[str] = None,
    max_words_per_line: int = 5,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_fn: Optional[Callable[[float], None]] = None,
    cancel_flag: Optional[threading.Event] = None,
) -> list[list[dict]]:
    """
    Transcribes *video_path* with Faster-Whisper and returns subtitle lines
    in the same format used by parse_ass_karaoke():
        [ [{"text": str, "start": float, "end": float}, ...], ... ]

    Timing improvement: each word's end time is set to the *next* word's
    start time (rather than the raw Whisper end), which removes gaps and
    produces tighter karaoke feel.

    Parameters
    ----------
    video_path        : path to the video (audio extracted internally)
    model_name        : Faster-Whisper model size
    language          : BCP-47 language code, or None for auto-detect
    max_words_per_line: how many words appear per subtitle line
    log_fn            : optional callable(str) for status messages
    progress_fn       : optional callable(float 0–1) for progress updates
    cancel_flag       : optional threading.Event; checked between segments
    """
    def _log(msg: str):
        if log_fn:
            log_fn(msg)

    def _prog(val: float):
        if progress_fn:
            progress_fn(val)

    _log("Detecting compute device…")
    device = _detect_device()
    compute = "float16" if device == "cuda" else "int8"
    _log(f"Using device: {device.upper()}  |  model: {model_name}")

    # ── 1. Extract audio to a temp WAV ──────────────────────────────────────
    _log("Extracting audio from video…")
    _prog(0.02)

    import subprocess
    fd, tmp_audio = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
             tmp_audio],
            capture_output=True, check=True, **flags,
        )
    except Exception as exc:
        raise RuntimeError(f"ffmpeg audio extraction failed: {exc}") from exc

    try:
        # ── 2. Load model ────────────────────────────────────────────────────
        _log(f"Loading Faster-Whisper [{model_name}]…")
        _prog(0.05)

        from faster_whisper import WhisperModel
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute)
        except Exception as gpu_err:
            _log(f"GPU init failed ({gpu_err}). Falling back to CPU…")
            model = WhisperModel(model_name, device="cpu", compute_type="int8")

        if cancel_flag and cancel_flag.is_set():
            return []

        # ── 3. Transcribe ────────────────────────────────────────────────────
        lang_arg = None if (language is None or language == "Auto-detect") else language
        _log("Transcribing…")
        _prog(0.10)

        segments_gen, info = model.transcribe(
            tmp_audio,
            language=lang_arg,
            word_timestamps=True,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=400,
                # Знижений поріг гучності — щоб ловити тихі/оброблені голоси
                # (ефект радіо, ТВ, телефону):
                min_speech_duration_ms=100,
                speech_pad_ms=400,
                threshold=0.35,          # замість дефолтного 0.5 — більш чутливо
            ),
            condition_on_previous_text=True,
            # Підказка моделі про можливий стиль мовлення (допомагає з обробленим звуком)
            initial_prompt=(
                "Нижче наведено транскрипцію відео зі звичайним мовленням, "
                "а також мовленням з ефектом радіо або телебачення."
                if lang_arg in (None, "uk")
                else None
            ),
            no_speech_threshold=0.4,     # дефолт 0.6 — менш агресивно відкидає тихі шматки
            log_prob_threshold=-1.2,     # дефолт -1.0 — трохи м'якше
            compression_ratio_threshold=2.8,  # дефолт 2.4 — дозволяє більше повторень
        )

        total_dur = info.duration if info.duration and info.duration > 0 else None
        detected = info.language or "unknown"
        _log(f"Detected language: {detected}  |  duration: "
             f"{total_dur:.1f}s" if total_dur else "Detected language: " + detected)

        # Collect all segments with word data
        all_segments = []
        t_start = time.monotonic()

        for seg in segments_gen:
            if cancel_flag and cancel_flag.is_set():
                return []

            seg_dict = {
                "start": seg.start,
                "end":   seg.end,
                "text":  seg.text,
                "words": [],
            }
            if seg.words:
                for w in seg.words:
                    seg_dict["words"].append({
                        "word":  w.word,
                        "start": w.start,
                        "end":   w.end,
                    })
            all_segments.append(seg_dict)

            if total_dur and total_dur > 0:
                ratio = min(seg.end / total_dur, 1.0)
                elapsed = time.monotonic() - t_start
                if ratio > 0.005 and elapsed > 0.3:
                    remaining = max(0.0, (elapsed / ratio) - elapsed)
                    eta = (f"{int(remaining // 60)}m {int(remaining % 60)}s"
                           if remaining >= 60 else f"{int(remaining)}s")
                    _log(f"  {seg.end:.1f}s / {total_dur:.1f}s  ETA {eta}")
                _prog(0.10 + ratio * 0.75)

        _log(f"Transcription done. {len(all_segments)} segments found.")
        _prog(0.87)

        # ── 4. Convert segments → subtitle lines ──────────────────────────
        subtitle_lines = _segments_to_word_lines(all_segments, max_words_per_line)
        _log(f"Built {len(subtitle_lines)} subtitle lines.")
        _prog(0.95)
        return subtitle_lines

    finally:
        try:
            os.remove(tmp_audio)
        except Exception:
            pass


def _clean_word_text(text: str) -> str:
    """
    Видаляє пробіл перед апострофом і дефісом, якщо після них іде літера.
    Наприклад: "зроби ть" → "зробить", "по- думати" → "по-думати"
    """
    # Пробіл перед апострофом/дефісом (тільки якщо далі є буква — не тире в прямій мові)
    text = re.sub(r"\s+([''ʼ`\u2019\-])(?=\w)", r"\1", text)
    return text.strip()


def _is_apostrophe_or_hyphen_join(prev_text: str, curr_text: str) -> bool:
    """
    Повертає True, якщо curr_text починається з апострофа або дефісу
    (тобто поточне слово — це суфікс, який треба приклеїти до попереднього,
    не переносячи на новий рядок).
    """
    return bool(curr_text) and curr_text[0] in ("'", "'", "ʼ", "`", "\u2019", "-")


def _segments_to_word_lines(
    segments: list[dict],
    max_words_per_line: int,
) -> list[list[dict]]:
    """
    Converts Whisper segments (with per-word timestamps) into subtitle lines.

    Key behaviours:
    1. Пауза між сегментами (≥ VAD-threshold) → завжди новий рядок субтитру,
       незалежно від ліміту слів.
    2. Апостроф / дефіс на початку слова → слово «приклеюється» до попереднього
       (не переноситься на новий рядок навіть якщо досягнуто ліміт).
    3. Пробіли перед апострофом і дефісом очищаються у тексті слова.

    Timing: word.end = next_word.start (tighter karaoke timing).
    """
    lines: list[list[dict]] = []

    for seg in segments:
        words = seg.get("words", [])

        # ── збираємо плоский список слів для цього сегмента ──────────────────
        seg_words: list[dict] = []
        if not words:
            text = _clean_word_text(seg.get("text", ""))
            if text:
                seg_words.append({
                    "word":  text,
                    "start": seg.get("start", 0.0),
                    "end":   seg.get("end",   0.0),
                })
        else:
            for w in words:
                word_text = _clean_word_text(w.get("word", ""))
                if not word_text:
                    continue
                seg_words.append({
                    "word":  word_text,
                    "start": w.get("start", seg.get("start", 0.0)),
                    "end":   w.get("end",   seg.get("end",   0.0)),
                })

        if not seg_words:
            continue

        # ── тайтинги: кінець слова = початок наступного ───────────────────────
        for i in range(len(seg_words) - 1):
            next_start = seg_words[i + 1]["start"]
            if next_start is not None and next_start > seg_words[i]["start"]:
                seg_words[i]["end"] = next_start

        # ── розбиваємо сегмент на рядки по ліміту слів ────────────────────────
        # Кожен сегмент починає новий рядок — це і є «пауза → новий рядок».
        # (Попередній `current` буфер закривається автоматично, бо ми запускаємо
        #  окремий цикл для кожного сегмента і одразу додаємо в lines.)
        current: list[dict] = []

        for w in seg_words:
            word_text = w["word"]
            word_entry = {"text": word_text, "start": w["start"], "end": w["end"]}

            # Якщо поточне слово — апостроф/дефіс-суфікс, «приклеюємо» його
            # до попереднього слова замість того щоб переносити на новий рядок.
            if current and _is_apostrophe_or_hyphen_join(current[-1]["text"], word_text):
                # Зливаємо з попереднім словом без пробілу
                prev = current[-1]
                current[-1] = {
                    "text":  prev["text"] + word_text,
                    "start": prev["start"],
                    "end":   word_entry["end"],
                }
                continue

            current.append(word_entry)

            # Досягли ліміту — закриваємо рядок
            if len(current) >= max_words_per_line:
                validated = _validate_word_timings(current)
                if validated:
                    lines.append(validated)
                current = []

        # Залишок сегмента → окремий рядок
        if current:
            validated = _validate_word_timings(current)
            if validated:
                lines.append(validated)

    return lines


# ─────────────────────────────────────────────────────────────
#  Timing validation
# ─────────────────────────────────────────────────────────────

def _validate_word_timings(words: list[dict]) -> list[dict]:
    result: list[dict] = []
    for i, w in enumerate(words):
        start = w["start"]
        end   = w["end"]

        if end <= start:
            end = start + MIN_WORD_DURATION

        if result:
            prev_end = result[-1]["end"]
            if start < prev_end:
                start = prev_end
            if end <= start:
                end = start + MIN_WORD_DURATION

        if i + 1 < len(words):
            next_start = words[i + 1]["start"]
            if end > next_start > start:
                end = next_start

        result.append({"text": w["text"], "start": start, "end": end})

    return result


# ─────────────────────────────────────────────────────────────
#  Word-wrap
# ─────────────────────────────────────────────────────────────

def _wrap_words_to_rows(
    words: list[dict],
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[list[int]]:
    space_w = font.getlength(" ")
    rows: list[list[int]] = []
    current_row: list[int] = []
    current_w = 0.0

    for idx, w in enumerate(words):
        word_w = font.getlength(w["text"])
        needed = word_w if not current_row else space_w + word_w

        if current_row and current_w + needed > max_width:
            rows.append(current_row)
            current_row = [idx]
            current_w = word_w
        else:
            current_row.append(idx)
            current_w += needed

    if current_row:
        rows.append(current_row)

    return rows


# ─────────────────────────────────────────────────────────────
#  Frame renderer  — supports Box and Color highlight modes
# ─────────────────────────────────────────────────────────────

def create_word_highlight_frame(
    line_words: list[dict],
    active_index: int,
    font_path: str,
    font_size: int,
    video_size: tuple[int, int],
    box_color_rgba: tuple[int, int, int, int],
    vertical_pct: float,
    text_color_mode: str = "white",
    inactive_alpha: int = 160,
    draw_shadow: bool = True,
    stroke_width: int = 2,
    stroke_color_rgba: tuple[int, int, int, int] = (0, 0, 0, 200),
    highlight_mode: str = "Box",
) -> np.ndarray | None:
    """
    Renders a single RGBA overlay frame.

    highlight_mode
    --------------
    "Box"   — active word gets a filled rounded rectangle (original behaviour).
    "Color" — active word is drawn in *box_color_rgba* colour (no rectangle);
              inactive words are drawn semi-transparent white with stroke.
    """
    font = _get_font(font_path, font_size)
    if font is None:
        return None

    vid_w, vid_h = video_size
    img  = Image.new("RGBA", (vid_w, vid_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin    = int(vid_w * 0.05)
    max_width = vid_w - 2 * margin
    space_w   = font.getlength(" ")
    word_widths = [font.getlength(w["text"]) for w in line_words]

    rows = _wrap_words_to_rows(line_words, font, max_width)
    if not rows:
        return None

    sample_bbox  = draw.textbbox((0, 0), "Agyp|", font=font)
    line_height  = (sample_bbox[3] - sample_bbox[1]) + 14
    total_rows   = len(rows)
    block_height = total_rows * line_height

    desired_top = vid_h * vertical_pct - block_height / 2
    max_top     = vid_h - block_height - int(vid_h * 0.02)
    base_y      = max(int(vid_h * 0.02), min(desired_top, max_top))

    # Active text colour (used in Box mode for text-on-box)
    if text_color_mode == "auto":
        active_fg = _auto_text_color(box_color_rgba)
    elif text_color_mode == "black":
        active_fg = (0, 0, 0, 255)
    else:
        active_fg = (255, 255, 255, 255)

    # Inactive word colour
    inactive_fg = (255, 255, 255, inactive_alpha)

    padding_x = max(6, int(font_size * 0.35))
    padding_y = max(3, int(font_size * 0.16))
    radius    = max(4, int(font_size * 0.25))

    stroke_fill = stroke_color_rgba[:3] + (
        int(stroke_color_rgba[3] * inactive_alpha / 255),
    )

    # Active word colour for Color mode = the highlight colour itself
    color_mode_active_fg = box_color_rgba[:3] + (255,)

    for row_idx, row_indices in enumerate(rows):
        row_w = (
            sum(word_widths[i] for i in row_indices)
            + space_w * (len(row_indices) - 1)
        )
        start_x = (vid_w - row_w) / 2
        start_y = base_y + row_idx * line_height
        current_x = start_x

        for i in row_indices:
            w         = line_words[i]
            tw        = word_widths[i]
            is_active = (i == active_index)

            text_bbox = draw.textbbox((current_x, start_y), w["text"], font=font)

            if highlight_mode == "Box":
                if is_active:
                    rect = [
                        text_bbox[0] - padding_x,
                        text_bbox[1] - padding_y,
                        text_bbox[2] + padding_x,
                        text_bbox[3] + padding_y,
                    ]
                    if draw_shadow:
                        shadow_rect = [r + 4 for r in rect]
                        shadow_img  = Image.new("RGBA", (vid_w, vid_h), (0, 0, 0, 0))
                        shadow_draw = ImageDraw.Draw(shadow_img)
                        shadow_draw.rounded_rectangle(shadow_rect, radius=radius, fill=(0, 0, 0, 80))
                        img  = Image.alpha_composite(img, shadow_img)
                        draw = ImageDraw.Draw(img)

                    draw.rounded_rectangle(rect, radius=radius, fill=box_color_rgba)
                    draw.text((current_x, start_y), w["text"], font=font,
                              fill=active_fg, stroke_width=0)
                else:
                    draw.text((current_x, start_y), w["text"], font=font,
                              fill=inactive_fg, stroke_width=stroke_width,
                              stroke_fill=stroke_fill)

            else:  # "Color" mode
                if is_active:
                    # Optional glow/shadow behind the highlighted word
                    if draw_shadow:
                        shadow_img  = Image.new("RGBA", (vid_w, vid_h), (0, 0, 0, 0))
                        shadow_draw = ImageDraw.Draw(shadow_img)
                        offset = max(2, font_size // 20)
                        shadow_draw.text(
                            (current_x + offset, start_y + offset),
                            w["text"], font=font,
                            fill=(0, 0, 0, 100),
                            stroke_width=stroke_width + 1,
                            stroke_fill=(0, 0, 0, 80),
                        )
                        img  = Image.alpha_composite(img, shadow_img)
                        draw = ImageDraw.Draw(img)

                    draw.text((current_x, start_y), w["text"], font=font,
                              fill=color_mode_active_fg,
                              stroke_width=max(0, stroke_width - 1),
                              stroke_fill=(0, 0, 0, 180))
                else:
                    draw.text((current_x, start_y), w["text"], font=font,
                              fill=inactive_fg,
                              stroke_width=stroke_width,
                              stroke_fill=stroke_fill)

            current_x += tw + space_w

    return np.array(img)


# ─────────────────────────────────────────────────────────────
#  Render pipeline
# ─────────────────────────────────────────────────────────────

def render_dynamic_subs(
    video_path: str,
    subtitle_source: list[list[dict]],  # pre-built lines from Whisper
    output_path: str,
    font_path: str,
    font_size: int,
    box_color_rgba: tuple[int, int, int, int],
    vertical_pct: float,
    text_color_mode: str,
    inactive_alpha: int,
    draw_shadow: bool,
    stroke_width: int,
    stroke_color_rgba: tuple[int, int, int, int],
    highlight_mode: str,
    log_fn: Callable[[str], None],
    done_fn: Callable[[bool, str], None],
    progress_fn: Callable[[float], None] | None = None,
    crf: int = 18,
    ffmpeg_preset: str = "fast",
) -> None:
    """
    Full render pipeline — runs in a background thread.

    *subtitle_source*: already-built subtitle lines from Whisper transcription.
    """
    frame_cache: dict[tuple, np.ndarray | None] = {}

    @contextmanager
    def open_video(path: str):
        clip = VideoFileClip(path)
        try:
            yield clip
        finally:
            clip.close()

    try:
        log_fn("Loading video…")
        with open_video(video_path) as video:
            vid_size: tuple[int, int] = tuple(video.size)

            lines = subtitle_source

            total_words = sum(len(ln) for ln in lines)
            log_fn(f"Found {len(lines)} lines, {total_words} words total.")

            if total_words == 0:
                done_fn(False, "No subtitle words found.")
                return

            clips = [video]
            processed = 0

            log_fn("Generating highlight frames…")

            for line_words in lines:
                line_key_base = tuple(w["text"] for w in line_words)

                for i, w in enumerate(line_words):
                    duration = w["end"] - w["start"]
                    if duration < MIN_WORD_DURATION:
                        processed += 1
                        continue

                    cache_key = (
                        line_key_base, i,
                        font_path, font_size,
                        vid_size,
                        box_color_rgba,
                        vertical_pct,
                        text_color_mode,
                        inactive_alpha,
                        draw_shadow,
                        stroke_width,
                        stroke_color_rgba,
                        highlight_mode,
                    )

                    if cache_key in frame_cache:
                        frame_array = frame_cache[cache_key]
                    else:
                        frame_array = create_word_highlight_frame(
                            line_words,
                            active_index=i,
                            font_path=font_path,
                            font_size=font_size,
                            video_size=vid_size,
                            box_color_rgba=box_color_rgba,
                            vertical_pct=vertical_pct,
                            text_color_mode=text_color_mode,
                            inactive_alpha=inactive_alpha,
                            draw_shadow=draw_shadow,
                            stroke_width=stroke_width,
                            stroke_color_rgba=stroke_color_rgba,
                            highlight_mode=highlight_mode,
                        )
                        frame_cache[cache_key] = frame_array

                    if frame_array is not None:
                        clip = (
                            ImageClip(frame_array)
                            .with_start(w["start"])
                            .with_duration(duration)
                        )
                        clips.append(clip)

                    processed += 1
                    if progress_fn and total_words > 0:
                        progress_fn(0.6 * processed / total_words)
                    if processed % 10 == 0:
                        log_fn(
                            f"  Processed {processed}/{total_words} words "
                            f"({int(100 * processed / total_words)}%)…"
                        )

            log_fn("Compositing final video (this may take a while)…")
            if progress_fn:
                progress_fn(0.65)

            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            final_video = CompositeVideoClip(clips)
            try:
                log_fn(f"Encoding  CRF={crf}  preset={ffmpeg_preset}…")
                final_video.write_videofile(
                    output_path,
                    fps=video.fps,
                    codec="libx264",
                    audio_codec="aac",
                    preset=ffmpeg_preset,
                    ffmpeg_params=["-crf", str(crf)],
                    threads=min(os.cpu_count() or 4, 8),
                    logger=None,
                )
            finally:
                final_video.close()

            if progress_fn:
                progress_fn(1.0)

            done_fn(True, f"Done! Saved to:\n{output_path}")

    except Exception as exc:
        done_fn(False, f"Render error: {exc}")


# ─────────────────────────────────────────────────────────────
#  UI Widgets
# ─────────────────────────────────────────────────────────────

class ColorButton(ctk.CTkFrame):
    """Small square button that displays and lets the user pick a hex colour."""

    def __init__(self, master, initial_hex: str = "#8A2BE2",
                 on_change: Callable | None = None, **kwargs):
        super().__init__(master, **kwargs)
        self._hex = initial_hex
        self._on_change = on_change
        self._btn = ctk.CTkButton(
            self, text="", width=40, height=40,
            corner_radius=10,
            fg_color=initial_hex, hover_color=initial_hex,
            command=self._pick,
        )
        self._btn.pack()

    def _pick(self):
        try:
            from tkinter.colorchooser import askcolor
            result = askcolor(color=self._hex, title="Pick colour")
            if result and result[1]:
                self.set_hex(result[1])
                if self._on_change:
                    self._on_change()
        except Exception:
            pass

    def set_hex(self, hex_color: str):
        self._hex = hex_color
        self._btn.configure(fg_color=hex_color, hover_color=hex_color)

    def get_hex(self) -> str:
        return self._hex

    def get_rgba(self, alpha: int = 230) -> tuple[int, int, int, int]:
        return _hex_to_rgba(self._hex, alpha)


class PreviewPanel(ctk.CTkFrame):
    """Live rendered preview of the subtitle style."""

    _BG_COLOR = (30, 30, 30)

    def __init__(self, master, width: int = 540, height: int = 200, **kwargs):
        super().__init__(master, **kwargs)
        self._prev_w = width
        self._prev_h = height
        self._tk_img: ctk.CTkImage | None = None
        self._pending_after: str | None = None
        self._is_rendering = False

        self._label = ctk.CTkLabel(self, text="")
        self._label.pack(fill="both", expand=True)

        self._status = ctk.CTkLabel(
            self, text="Preview will appear after fonts load.",
            text_color="gray50", font=ctk.CTkFont(size=11),
        )  # preview panel always English
        self._status.pack(pady=(0, 4))
        self._show_placeholder()

    def _show_placeholder(self):
        img  = Image.new("RGB", (self._prev_w, self._prev_h), self._BG_COLOR)
        draw = ImageDraw.Draw(img)
        draw.text((self._prev_w // 2, self._prev_h // 2), "Preview",
                  fill=(80, 80, 80), anchor="mm")
        self._set_image(img)

    def _set_image(self, img: Image.Image):
        self._tk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self._label.configure(image=self._tk_img, text="")

    def schedule_update(self, params: dict, delay_ms: int = 180):
        if self._pending_after is not None:
            try:
                self.after_cancel(self._pending_after)
            except Exception:
                pass
        self._pending_after = self.after(delay_ms, lambda: self._start_render(params))

    def _start_render(self, params: dict):
        self._pending_after = None
        if self._is_rendering:
            return
        self._is_rendering = True
        self._status.configure(text="Rendering preview…")
        threading.Thread(target=self._render_worker, args=(params,), daemon=True).start()

    def _render_worker(self, params: dict):
        try:
            font_path = params.get("font_path")
            if not font_path or not os.path.isfile(font_path):
                self.after(0, lambda: self._status.configure(
                    text="Select a valid font to see preview."))
                return

            font_size = max(10, min(params["font_size"], 80))

            ui_lang = params.get("ui_lang", "English")
            sample_words = PREVIEW_SAMPLE_WORDS.get(ui_lang, PREVIEW_SAMPLE_WORDS["English"])
            active_idx   = PREVIEW_ACTIVE_INDEX.get(ui_lang, 3)

            frame = create_word_highlight_frame(
                line_words=sample_words,
                active_index=active_idx,
                font_path=font_path,
                font_size=font_size,
                video_size=(self._prev_w, self._prev_h),
                box_color_rgba=params["box_color_rgba"],
                vertical_pct=0.5,
                text_color_mode=params["text_color_mode"],
                inactive_alpha=params["inactive_alpha"],
                draw_shadow=params["draw_shadow"],
                stroke_width=params["stroke_width"],
                stroke_color_rgba=params["stroke_color_rgba"],
                highlight_mode=params["highlight_mode"],
            )

            if frame is None:
                self.after(0, lambda: self._status.configure(
                    text="Could not render preview (font error)."))
                return

            bg = Image.new("RGB", (self._prev_w, self._prev_h), self._BG_COLOR)
            overlay = Image.fromarray(frame)
            bg.paste(overlay, mask=overlay.split()[3])

            mode_label = params.get("highlight_mode", "Box")

            def apply(img=bg):
                self._set_image(img)
                self._status.configure(
                    text=f"Live preview  ·  mode: {mode_label}  ·  active word highlighted")

            self.after(0, apply)

        except Exception as exc:
            self.after(0, lambda e=str(exc): self._status.configure(
                text=f"Preview error: {e}"))
        finally:
            self._is_rendering = False


# ─────────────────────────────────────────────────────────────
#  Main Application
# ─────────────────────────────────────────────────────────────

class App(ctk.CTk):
    # ── Settings persistence ─────────────────────────────────────────────────

    _SETTINGS_FILE = os.path.join(
        os.path.expanduser("~"), ".tiktok_karaoke_settings.json"
    )

    def _load_settings(self) -> dict:
        import json
        try:
            with open(self._SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_settings(self):
        import json
        data = {
            "ui_language":      self._ui_lang_var.get(),
            "whisper_model":    self._whisper_model_var.get(),
            "whisper_lang":     self._whisper_lang_var.get(),
            "words_per_line":   int(self._words_per_line_var.get()),
            "highlight_mode":   self._highlight_mode_var.get(),
            "preset":           self._preset_var.get(),
            "font_name":        self._font_var.get(),
            "font_size":        int(self._font_size_var.get()),
            "highlight_hex":    self._color_btn.get_hex(),
            "alpha":            int(self._alpha_var.get()),
            "inactive_alpha":   int(self._inactive_alpha_var.get()),
            "vpos":             float(self._vpos_var.get()),
            "text_color":       self._text_color_var.get(),
            "shadow":           self._shadow_var.get(),
            "stroke_width":     int(self._stroke_width_var.get()),
            "stroke_hex":       (self._stroke_color_btn.get_hex()
                                 if self._stroke_color_btn else "#000000"),
            "quality_idx":      getattr(self, "_quality_idx", 0),
        }
        try:
            with open(self._SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("🎵 TikTok Karaoke Subtitle Renderer")
        self.minsize(960, 740)
        self.resizable(True, True)

        self._fonts: dict[str, str] = {}
        self._all_font_names: list[str] = []
        self._render_thread: threading.Thread | None = None
        self._transcribe_thread: threading.Thread | None = None
        self._last_output_path: str = ""
        self._subtitle_lines: list[list[dict]] | None = None

        # Language selector var (needs to exist before _build_ui)
        self._ui_lang_var = ctk.StringVar(value="English")

        self._settings = self._load_settings()
        self._build_ui()
        self._apply_saved_settings(self._settings)
        self._load_fonts()
        self.after(50, self._maximize_window)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self._save_settings()
        self.destroy()

    def _apply_saved_settings(self, s: dict):
        """Restore persisted settings after UI is built."""
        if not s:
            return
        if "ui_language" in s:
            self._ui_lang_var.set(s["ui_language"])
            self._on_lang_change(s["ui_language"])
        if "whisper_model" in s:
            self._whisper_model_var.set(s["whisper_model"])
        if "whisper_lang" in s:
            self._whisper_lang_var.set(s["whisper_lang"])
        if "words_per_line" in s:
            self._words_per_line_var.set(s["words_per_line"])
            self._words_per_line_label.configure(
                text=_t(self._ui_lang_var.get(), "words_unit", s["words_per_line"]))
        if "highlight_mode" in s:
            self._highlight_mode_var.set(s["highlight_mode"])
        if "preset" in s:
            self._preset_var.set(s["preset"])
        if "font_name" in s:
            self._font_var.set(s["font_name"])
        if "font_size" in s:
            self._font_size_var.set(s["font_size"])
        if "highlight_hex" in s:
            self._color_btn.set_hex(s["highlight_hex"])
        if "alpha" in s:
            self._alpha_var.set(s["alpha"])
        if "inactive_alpha" in s:
            self._inactive_alpha_var.set(s["inactive_alpha"])
        if "vpos" in s:
            self._vpos_var.set(s["vpos"])
        if "text_color" in s:
            self._text_color_var.set(s["text_color"])
        if "shadow" in s:
            self._shadow_var.set(s["shadow"])
        if "stroke_width" in s:
            self._stroke_width_var.set(s["stroke_width"])
        if "stroke_hex" in s and self._stroke_color_btn:
            self._stroke_color_btn.set_hex(s["stroke_hex"])
        if "quality_idx" in s and hasattr(self, "_quality_combo"):
            idx = int(s["quality_idx"])
            self._quality_idx = idx
            L = self._ui_lang_var.get()
            labels = [_t(L, k) for k, _, _ in QUALITY_OPTIONS]
            self._quality_combo.set(labels[idx] if idx < len(labels) else labels[0])

    # ── Window helpers ───────────────────────────────────────────────────────

    def _maximize_window(self):
        system = platform.system()
        try:
            if system == "Windows":
                self.state("zoomed")
            elif system == "Linux":
                self.attributes("-zoomed", True)
            else:
                w, h = self.winfo_screenwidth(), self.winfo_screenheight()
                self.geometry(f"{w}x{h}+0+0")
        except Exception:
            try:
                w, h = self.winfo_screenwidth(), self.winfo_screenheight()
                self.geometry(f"{w}x{h}+0+0")
            except Exception:
                pass

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        PAD = 16
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)  # lang bar
        self.grid_rowconfigure(1, weight=0)  # title
        self.grid_rowconfigure(2, weight=1)  # scroll
        self.grid_rowconfigure(3, weight=0)  # quality
        self.grid_rowconfigure(4, weight=0)  # buttons
        self.grid_rowconfigure(5, weight=0)  # progress
        self.grid_rowconfigure(6, weight=0)  # log

        self._build_lang_bar(PAD)
        self._build_title(PAD)
        self._build_scroll_area(PAD)
        self._build_buttons(PAD)
        self._build_progress(PAD)
        self._build_log(PAD)

    def _build_lang_bar(self, PAD: int):
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=PAD, pady=(8, 0))

        self._lang_label_widget = ctk.CTkLabel(
            bar, text="Interface language:", anchor="w", text_color="gray60")
        self._lang_label_widget.pack(side="left")

        ctk.CTkSegmentedButton(
            bar,
            values=list(UI_LANGUAGES),
            variable=self._ui_lang_var,
            command=self._on_lang_change,
            width=220,
        ).pack(side="left", padx=(8, 0))

    def _on_lang_change(self, lang: str):
        """Retranslate all dynamic UI labels."""
        L = lang
        # Title / subtitle
        if hasattr(self, "_title_label"):
            self._title_label.configure(text=_t(L, "title"))
        if hasattr(self, "_subtitle_label"):
            self._subtitle_label.configure(text=_t(L, "subtitle"))
        if hasattr(self, "_lang_label_widget"):
            self._lang_label_widget.configure(text=_t(L, "ui_language"))
        # File section
        for attr, key in [
            ("_video_label",  "video_file"),
            ("_output_label", "output_file"),
        ]:
            if hasattr(self, attr):
                getattr(self, attr).configure(text=_t(L, key))
        for attr in ("_browse_video_btn", "_browse_output_btn"):
            if hasattr(self, attr):
                getattr(self, attr).configure(text=_t(L, "browse"))
        # Transcribe section
        if hasattr(self, "_whisper_section_label"):
            self._whisper_section_label.configure(text=_t(L, "whisper_section"))
        for attr, key in [
            ("_model_label",    "model"),
            ("_lang_label",     "language"),
            ("_wpl_label",      "words_per_line"),
        ]:
            if hasattr(self, attr):
                getattr(self, attr).configure(text=_t(L, key))
        if hasattr(self, "_transcribe_btn"):
            state = str(self._transcribe_btn.cget("state"))
            if "disabled" not in state:
                self._transcribe_btn.configure(text=_t(L, "transcribe_btn"))
        if hasattr(self, "_words_per_line_label"):
            n = int(self._words_per_line_var.get())
            self._words_per_line_label.configure(text=_t(L, "words_unit", n))
        # Options labels
        for attr, key in [
            ("_hl_mode_label",      "highlight_mode"),
            ("_preset_label",       "style_preset"),
            ("_font_label",         "font"),
            ("_font_size_label",    "font_size"),
            ("_hl_color_label",     "highlight_color"),
            ("_box_opacity_label",  "box_opacity"),
            ("_inact_opacity_label","inactive_opacity"),
            ("_vpos_label",         "vertical_pos"),
            ("_text_color_label",   "text_color"),
            ("_shadow_label",       "drop_shadow"),
            ("_stroke_w_label",     "stroke_width"),
            ("_stroke_c_label",     "stroke_color"),
        ]:
            if hasattr(self, attr):
                getattr(self, attr).configure(text=_t(L, key))
        if hasattr(self, "_font_search_entry"):
            self._font_search_entry.configure(placeholder_text=_t(L, "font_search"))
        # Preview
        if hasattr(self, "_preview_title_label"):
            self._preview_title_label.configure(text=_t(L, "live_preview"))
        if hasattr(self, "_refresh_btn"):
            self._refresh_btn.configure(text=_t(L, "refresh"))
        # Buttons
        if hasattr(self, "_render_btn"):
            state = str(self._render_btn.cget("state"))
            if "disabled" not in state:
                self._render_btn.configure(text=_t(L, "render_btn"))
        if hasattr(self, "_open_dir_btn"):
            self._open_dir_btn.configure(text=_t(L, "open_folder"))
        if hasattr(self, "_log_label"):
            self._log_label.configure(text=_t(L, "log_label"))
        # Update quality combobox labels
        if hasattr(self, "_quality_combo") and hasattr(self, "_quality_var"):
            opts = [_t(L, k) for k, _, _ in QUALITY_OPTIONS]
            current_idx = self._quality_var.get()
            self._quality_combo.configure(values=opts)
            self._quality_combo.set(opts[current_idx] if isinstance(current_idx, int)
                                    else opts[0])
        if hasattr(self, "_quality_label_widget"):
            self._quality_label_widget.configure(text=_t(L, "quality_label"))
        if hasattr(self, "_quality_hint_label"):
            self._quality_hint_label.configure(text=_t(L, "quality_hint"))
        if hasattr(self, "_edit_subs_btn"):
            self._edit_subs_btn.configure(text=_t(L, "edit_btn"))
        if hasattr(self, "_preset_hint_label"):
            self._preset_hint_label.configure(text=_t(L, "preset_hint"))
        if hasattr(self, "_color_hint_label"):
            self._color_hint_label.configure(text=_t(L, "color_hint"))
        if hasattr(self, "_text_color_hint_label"):
            self._text_color_hint_label.configure(text=_t(L, "text_color_hint"))
        # Refresh preview words for new language
        self._request_preview()

    def _build_title(self, PAD: int):
        title_frame = ctk.CTkFrame(self, fg_color="transparent")
        title_frame.grid(row=1, column=0, sticky="ew")

        self._title_label = ctk.CTkLabel(
            title_frame,
            text=_t(self._ui_lang_var.get(), "title"),
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self._title_label.pack(pady=(PAD, 2))

        self._subtitle_label = ctk.CTkLabel(
            title_frame,
            text=_t(self._ui_lang_var.get(), "subtitle"),
            font=ctk.CTkFont(size=13),
            text_color="gray60",
        )
        self._subtitle_label.pack(pady=(0, PAD))

    def _build_scroll_area(self, PAD: int):
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.grid(row=2, column=0, sticky="nsew", padx=PAD)
        scroll.grid_columnconfigure(0, weight=1)
        scroll.grid_columnconfigure(1, weight=1)

        left_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, PAD))

        self._build_file_section(left_frame, PAD)
        self._build_transcribe_section(left_frame, PAD)
        self._build_options_section(left_frame)

        right_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        right_frame.grid(row=0, column=1, sticky="new")
        self._build_preview_section(right_frame)

    # ── File section ─────────────────────────────────────────────────────────

    def _build_file_section(self, parent, PAD: int):
        files_frame = ctk.CTkFrame(parent)
        files_frame.pack(fill="x", pady=(0, PAD))

        self._video_var  = ctk.StringVar()
        downloads_dir    = os.path.join(os.path.expanduser("~"), "Downloads")
        os.makedirs(downloads_dir, exist_ok=True)
        self._output_var = ctk.StringVar(
            value=os.path.join(downloads_dir, "output_karaoke.mp4")
        )

        L = self._ui_lang_var.get()

        # Video row
        self._video_label = ctk.CTkLabel(
            files_frame, text=_t(L, "video_file"), anchor="w", width=110)
        self._video_label.grid(row=0, column=0, sticky="w", padx=12, pady=8)
        ctk.CTkEntry(files_frame, textvariable=self._video_var, width=420).grid(
            row=0, column=1, sticky="w", padx=4, pady=8)
        self._browse_video_btn = ctk.CTkButton(
            files_frame, text=_t(L, "browse"), width=80, command=self._browse_video)
        self._browse_video_btn.grid(row=0, column=2, padx=8, pady=8)

        # Output row
        self._output_label = ctk.CTkLabel(
            files_frame, text=_t(L, "output_file"), anchor="w", width=110)
        self._output_label.grid(row=1, column=0, sticky="w", padx=12, pady=8)
        ctk.CTkEntry(files_frame, textvariable=self._output_var, width=420).grid(
            row=1, column=1, sticky="w", padx=4, pady=8)
        self._browse_output_btn = ctk.CTkButton(
            files_frame, text=_t(L, "browse"), width=80, command=self._browse_output)
        self._browse_output_btn.grid(row=1, column=2, padx=8, pady=8)

    # ── Whisper / transcription section ─────────────────────────────────────

    def _build_transcribe_section(self, parent, PAD: int):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=(0, PAD))

        L = self._ui_lang_var.get()

        self._whisper_section_label = ctk.CTkLabel(
            frame,
            text=_t(L, "whisper_section"),
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        )
        self._whisper_section_label.grid(
            row=0, column=0, columnspan=4, sticky="w", padx=12, pady=(10, 4))

        # Model
        self._whisper_model_var = ctk.StringVar(value="Systran/faster-whisper-base")
        self._model_label = ctk.CTkLabel(frame, text=_t(L, "model"), anchor="w")
        self._model_label.grid(row=1, column=0, sticky="w", padx=12, pady=6)
        ctk.CTkComboBox(
            frame, width=260, variable=self._whisper_model_var,
            values=list(WHISPER_MODELS),
        ).grid(row=1, column=1, sticky="w", padx=4, pady=6)

        # Language
        self._whisper_lang_var = ctk.StringVar(value="Auto-detect")
        self._lang_label = ctk.CTkLabel(frame, text=_t(L, "language"), anchor="w")
        self._lang_label.grid(row=1, column=2, sticky="w", padx=12, pady=6)
        ctk.CTkComboBox(
            frame, width=130, variable=self._whisper_lang_var,
            values=list(WHISPER_LANGUAGES),
        ).grid(row=1, column=3, sticky="w", padx=4, pady=6)

        # Words per line
        self._words_per_line_var = ctk.IntVar(value=5)
        self._wpl_label = ctk.CTkLabel(frame, text=_t(L, "words_per_line"), anchor="w")
        self._wpl_label.grid(row=2, column=0, sticky="w", padx=12, pady=6)
        words_slider = ctk.CTkSlider(
            frame, from_=2, to=10,
            variable=self._words_per_line_var,
            width=160, number_of_steps=8,
            command=self._on_words_per_line_change,
        )
        words_slider.grid(row=2, column=1, sticky="w", padx=4, pady=6)
        self._words_per_line_label = ctk.CTkLabel(
            frame, text=_t(L, "words_unit", 5), width=70)
        self._words_per_line_label.grid(row=2, column=2, sticky="w", padx=4)

        # Transcribe button
        self._transcribe_btn = ctk.CTkButton(
            frame,
            text=_t(L, "transcribe_btn"),
            height=36,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._start_transcribe,
        )
        self._transcribe_btn.grid(row=2, column=3, padx=12, pady=6, sticky="e")

        # Status label
        self._transcribe_status = ctk.CTkLabel(
            frame,
            text=_t(L, "no_transcription"),
            text_color="gray60",
            font=ctk.CTkFont(size=11),
            anchor="w",
        )
        self._transcribe_status.grid(
            row=3, column=0, columnspan=4, sticky="w", padx=12, pady=(0, 8))

    def _on_words_per_line_change(self, val):
        n = int(float(val))
        L = self._ui_lang_var.get()
        self._words_per_line_label.configure(text=_t(L, "words_unit", n))
        self._subtitle_lines = None
        self._transcribe_status.configure(
            text=_t(L, "words_changed"),
            text_color="gray60",
        )

    # ── Style options ────────────────────────────────────────────────────────

    def _build_options_section(self, parent):
        opts = ctk.CTkFrame(parent)
        opts.pack(fill="x", pady=(0, 16))

        self._preset_var         = ctk.StringVar(value="Purple Haze")
        self._font_var           = ctk.StringVar(value="Loading...")
        self._font_search_var    = ctk.StringVar()
        self._font_size_var      = ctk.IntVar(value=62)
        self._alpha_var          = ctk.IntVar(value=230)
        self._inactive_alpha_var = ctk.IntVar(value=160)
        self._vpos_var           = ctk.DoubleVar(value=0.80)
        self._text_color_var     = ctk.StringVar(value="white")
        self._shadow_var         = ctk.BooleanVar(value=True)
        self._stroke_width_var   = ctk.IntVar(value=2)
        self._highlight_mode_var = ctk.StringVar(value="Box")
        self._stroke_color_btn: ColorButton | None = None

        row = 0
        L = self._ui_lang_var.get()

        # Highlight mode
        self._hl_mode_label = ctk.CTkLabel(
            opts, text=_t(L, "highlight_mode"), anchor="w")
        self._hl_mode_label.grid(row=row, column=0, sticky="w", padx=12, pady=8)
        ctk.CTkSegmentedButton(
            opts,
            values=list(HIGHLIGHT_MODES),
            variable=self._highlight_mode_var,
            command=lambda _: self._request_preview(),
            width=200,
        ).grid(row=row, column=1, sticky="w", padx=4, pady=8)
        ctk.CTkLabel(
            opts,
            text='"Box" = rectangle behind word  ·  "Color" = word changes colour',
            text_color="gray60",
        ).grid(row=row, column=2, columnspan=2, sticky="w", padx=4)
        row += 1

        # Style preset
        self._preset_label = ctk.CTkLabel(opts, text=_t(L, "style_preset"), anchor="w")
        self._preset_label.grid(
            row=row, column=0, sticky="w", padx=12, pady=8)
        ctk.CTkComboBox(
            opts, width=200, variable=self._preset_var,
            values=list(STYLE_PRESETS.keys()),
            command=self._apply_preset,
        ).grid(row=row, column=1, sticky="w", padx=4, pady=8)
        self._preset_hint_label = ctk.CTkLabel(opts, text=_t(L, "preset_hint"), text_color="gray60")
        self._preset_hint_label.grid(row=row, column=2, columnspan=2, sticky="w", padx=4)
        row += 1

        # Font
        self._font_label = ctk.CTkLabel(opts, text=_t(L, "font"), anchor="w")
        self._font_label.grid(row=row, column=0, sticky="w", padx=12, pady=8)
        self._font_search_var.trace_add("write", self._filter_fonts)
        self._font_search_entry = ctk.CTkEntry(
            opts, textvariable=self._font_search_var,
            placeholder_text=_t(L, "font_search"), width=180,
        )
        self._font_search_entry.grid(row=row, column=1, sticky="w", padx=4, pady=8)
        self._font_combo = ctk.CTkComboBox(
            opts, width=240, variable=self._font_var, values=["Loading…"],
            command=lambda _: self._request_preview(),
        )
        self._font_combo.grid(row=row, column=2, sticky="w", padx=4, pady=8)
        row += 1

        # Font size
        self._font_size_label = self._add_slider_row(
            opts, row, _t(L, "font_size"), self._font_size_var,
            from_=24, to=120, fmt=_t(L, "px", "{}"), initial=_t(L, "px", 62))
        row += 1

        # Highlight / box colour
        self._hl_color_label = ctk.CTkLabel(opts, text=_t(L, "highlight_color"), anchor="w")
        self._hl_color_label.grid(row=row, column=0, sticky="w", padx=12, pady=8)
        self._color_btn = ColorButton(
            opts, initial_hex="#8A2BE2", on_change=self._request_preview)
        self._color_btn.grid(row=row, column=1, sticky="w", padx=4, pady=8)
        self._color_hint_label = ctk.CTkLabel(opts, text=_t(L, "color_hint"), text_color="gray60")
        self._color_hint_label.grid(row=row, column=2, columnspan=2, sticky="w", padx=4)
        row += 1

        # Box opacity (only visible in Box mode, but harmless otherwise)
        self._box_opacity_label = self._add_slider_row(
            opts, row, _t(L, "box_opacity"), self._alpha_var,
            from_=50, to=255, fmt="{}", initial="230")
        row += 1

        # Inactive opacity
        self._inact_opacity_label = self._add_slider_row(
            opts, row, _t(L, "inactive_opacity"), self._inactive_alpha_var,
            from_=30, to=255, fmt="{}", initial="160")
        row += 1

        # Vertical position
        self._vpos_label = self._add_slider_row(
            opts, row, _t(L, "vertical_pos"), self._vpos_var,
            from_=0.1, to=0.95, fmt="{}%", initial="80%", is_pct=True)
        row += 1

        # Text colour
        self._text_color_label = ctk.CTkLabel(opts, text=_t(L, "text_color"), anchor="w")
        self._text_color_label.grid(row=row, column=0, sticky="w", padx=12, pady=8)
        ctk.CTkComboBox(
            opts, width=140, variable=self._text_color_var,
            values=list(TEXT_COLOR_OPTIONS),
            command=lambda _: self._request_preview(),
        ).grid(row=row, column=1, sticky="w", padx=4, pady=8)
        self._text_color_hint_label = ctk.CTkLabel(opts, text=_t(L, "text_color_hint"), text_color="gray60")
        self._text_color_hint_label.grid(row=row, column=2, columnspan=2, sticky="w", padx=4)
        row += 1

        # Drop shadow
        self._shadow_label = ctk.CTkLabel(opts, text=_t(L, "drop_shadow"), anchor="w")
        self._shadow_label.grid(row=row, column=0, sticky="w", padx=12, pady=8)
        ctk.CTkSwitch(
            opts, text="", variable=self._shadow_var,
            command=self._request_preview,
        ).grid(row=row, column=1, sticky="w", padx=4, pady=8)
        row += 1

        # Stroke width
        self._stroke_w_label = self._add_slider_row(
            opts, row, _t(L, "stroke_width"), self._stroke_width_var,
            from_=0, to=8, fmt=_t(L, "px", "{}"), initial=_t(L, "px", 2))
        row += 1

        # Stroke colour
        self._stroke_c_label = ctk.CTkLabel(opts, text=_t(L, "stroke_color"), anchor="w")
        self._stroke_c_label.grid(row=row, column=0, sticky="w", padx=12, pady=8)
        self._stroke_color_btn = ColorButton(
            opts, initial_hex="#000000", on_change=self._request_preview)
        self._stroke_color_btn.grid(row=row, column=1, sticky="w", padx=4, pady=8)
        ctk.CTkLabel(opts, text="outline for inactive words", text_color="gray60").grid(
            row=row, column=2, columnspan=2, sticky="w", padx=4)
        row += 1

    # ── Preview ──────────────────────────────────────────────────────────────

    def _build_preview_section(self, parent):
        # Outer frame that stays at the top of the right column
        preview_frame = ctk.CTkFrame(parent)
        preview_frame.pack(fill="x", pady=(0, 16), anchor="n")

        header = ctk.CTkFrame(preview_frame, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(8, 4))

        L = self._ui_lang_var.get()
        self._preview_title_label = ctk.CTkLabel(
            header, text=_t(L, "live_preview"),
            font=ctk.CTkFont(size=14, weight="bold"), anchor="w",
        )
        self._preview_title_label.pack(side="left")

        self._refresh_btn = ctk.CTkButton(
            header, text=_t(L, "refresh"), width=100, height=28,
            command=self._request_preview,
        )
        self._refresh_btn.pack(side="right")

        self._preview = PreviewPanel(
            preview_frame, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1],
            fg_color="#1a1a1a", corner_radius=10,
        )
        self._preview.pack(fill="x", padx=12, pady=(0, 12))

    # ── Bottom controls ──────────────────────────────────────────────────────

    def _build_buttons(self, PAD: int):
        # ── Quality row ──────────────────────────────────────────────────────
        quality_frame = ctk.CTkFrame(self, fg_color="transparent")
        quality_frame.grid(row=3, column=0, sticky="ew", padx=PAD, pady=(8, 0))

        L = self._ui_lang_var.get()

        self._quality_label_widget = ctk.CTkLabel(
            quality_frame, text=_t(L, "quality_label"), anchor="w")
        self._quality_label_widget.pack(side="left")

        # Store selected index internally (0-3)
        self._quality_idx = 0  # default: Visually Lossless
        quality_labels = [_t(L, k) for k, _, _ in QUALITY_OPTIONS]

        self._quality_combo = ctk.CTkComboBox(
            quality_frame,
            values=quality_labels,
            width=260,
            state="readonly",
            command=self._on_quality_change,
        )
        self._quality_combo.set(quality_labels[0])
        self._quality_combo.pack(side="left", padx=(8, 12))

        self._quality_hint_label = ctk.CTkLabel(
            quality_frame, text=_t(L, "quality_hint"),
            text_color="gray60", font=ctk.CTkFont(size=11))
        self._quality_hint_label.pack(side="left")

        # ── Action buttons row ────────────────────────────────────────────────
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=4, column=0, sticky="ew", padx=PAD, pady=(6, PAD))

        self._edit_subs_btn = ctk.CTkButton(
            btn_frame, text=_t(L, "edit_btn"),
            height=44, width=180,
            command=self._open_subtitle_editor,
        )
        self._edit_subs_btn.pack(side="left", padx=(0, 6))

        self._render_btn = ctk.CTkButton(
            btn_frame, text=_t(L, "render_btn"),
            font=ctk.CTkFont(size=15, weight="bold"),
            height=44, command=self._start_render,
        )
        self._render_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self._open_dir_btn = ctk.CTkButton(
            btn_frame, text=_t(L, "open_folder"),
            height=44, width=140, state="disabled",
            command=self._open_output_folder,
        )
        self._open_dir_btn.pack(side="left")

    def _on_quality_change(self, label: str):
        L = self._ui_lang_var.get()
        for idx, (key, _, _) in enumerate(QUALITY_OPTIONS):
            if _t(L, key) == label:
                self._quality_idx = idx
                break

    def _build_progress(self, PAD: int):
        progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        progress_frame.grid(row=5, column=0, sticky="ew", padx=PAD)

        self._progress = ctk.CTkProgressBar(progress_frame)
        self._progress.pack(fill="x", pady=(0, 6))
        self._progress.set(0)

        self._progress_label = ctk.CTkLabel(progress_frame, text="", text_color="gray60")
        self._progress_label.pack(pady=(0, 4))

    def _build_log(self, PAD: int):
        log_frame = ctk.CTkFrame(self, fg_color="transparent")
        log_frame.grid(row=6, column=0, sticky="nsew", padx=PAD, pady=(0, 4))
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        self._log_label = ctk.CTkLabel(
            log_frame, text=_t(self._ui_lang_var.get(), "log_label"),
            anchor="w", text_color="gray60")
        self._log_label.grid(row=0, column=0, sticky="w", pady=(0, 2))

        self._log_box = ctk.CTkTextbox(
            log_frame, height=80,
            font=ctk.CTkFont(family="Courier", size=12),
        )
        self._log_box.grid(row=1, column=0, sticky="nsew")
        self._log_box.configure(state="disabled")

    # ── Shared UI helper: file row ────────────────────────────────────────────

    def _add_file_row(self, parent, row, label, var, command):
        ctk.CTkLabel(parent, text=label, anchor="w", width=110).grid(
            row=row, column=0, sticky="w", padx=12, pady=8)
        ctk.CTkEntry(parent, textvariable=var, width=420).grid(
            row=row, column=1, sticky="w", padx=4, pady=8)
        ctk.CTkButton(parent, text="Browse", width=80, command=command).grid(
            row=row, column=2, padx=8, pady=8)

    def _add_slider_row(self, parent, row: int, label: str, var,
                        from_, to, fmt: str, initial: str,
                        is_pct: bool = False) -> ctk.CTkLabel:
        lbl_widget = ctk.CTkLabel(parent, text=label, anchor="w")
        lbl_widget.grid(row=row, column=0, sticky="w", padx=12, pady=8)
        value_label = ctk.CTkLabel(parent, text=str(initial), width=70)

        def on_change(val, lbl=value_label, f=fmt, p=is_pct):
            v = float(val)
            text = f.format(f"{int(v * 100)}") if p else f.format(int(v))
            lbl.configure(text=text)
            self._request_preview()

        ctk.CTkSlider(
            parent, from_=from_, to=to, variable=var, width=300,
            command=on_change,
        ).grid(row=row, column=1, columnspan=2, sticky="w", padx=4, pady=8)
        value_label.grid(row=row, column=3, sticky="w", padx=4)
        return value_label

    # ── Browse callbacks ─────────────────────────────────────────────────────

    def _browse_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.MP4"),
                       ("All files", "*.*")])
        if path:
            self._video_var.set(path)
            # Auto-fill output: same folder, "originalname_subs.mp4"
            base, _ = os.path.splitext(path)
            self._output_var.set(base + "_subs.mp4")
            # Reset cached transcription when video changes
            self._subtitle_lines = None
            L = self._ui_lang_var.get()
            self._transcribe_status.configure(
                text=_t(L, "new_video"),
                text_color="gray60")

    def _browse_output(self):
        # Pre-fill dialog with the current auto-generated name
        current = self._output_var.get().strip()
        init_dir  = os.path.dirname(current)  if current else os.path.expanduser("~")
        init_file = os.path.basename(current) if current else "output_subs.mp4"
        path = filedialog.asksaveasfilename(
            initialdir=init_dir,
            initialfile=init_file,
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4")])
        if path:
            self._output_var.set(path)

    # ── Font management ───────────────────────────────────────────────────────

    def _load_fonts(self):
        self._log(_t(self._ui_lang_var.get(), "loading_fonts"))

        def worker():
            fonts = get_system_fonts()
            names = list(fonts.keys())
            self._fonts = fonts
            self.after(0, lambda: self._set_font_list(names))

        threading.Thread(target=worker, daemon=True).start()

    def _set_font_list(self, names: list[str]):
        self._all_font_names = names
        self._font_combo.configure(values=names)
        if names:
            self._font_var.set(names[0])
        self._log(_t(self._ui_lang_var.get(), "fonts_loaded", len(names)))
        self._request_preview()

    def _filter_fonts(self, *_):
        if not self._all_font_names:
            return
        query = self._font_search_var.get().lower()
        filtered = [n for n in self._all_font_names if query in n.lower()]
        if filtered:
            self._font_combo.configure(values=filtered)
            self._font_var.set(filtered[0])
        else:
            nm = _t(self._ui_lang_var.get(), "font_no_match")
            self._font_combo.configure(values=[nm])
            self._font_var.set(nm)
        self._request_preview()

    def _apply_preset(self, preset_name: str):
        preset = STYLE_PRESETS.get(preset_name)
        if not preset:
            return
        self._color_btn.set_hex(preset["hex"])
        self._text_color_var.set(preset["text"])
        self._request_preview()

    # ── Preview ───────────────────────────────────────────────────────────────

    def _get_preview_params(self) -> dict:
        font_name = self._font_var.get()
        font_path = self._fonts.get(font_name, "")
        stroke_alpha = min(255, int(255 * 0.78))
        return {
            "ui_lang":           self._ui_lang_var.get(),
            "font_path":         font_path,
            "font_size":         int(self._font_size_var.get()),
            "box_color_rgba":    self._color_btn.get_rgba(alpha=int(self._alpha_var.get())),
            "text_color_mode":   self._text_color_var.get(),
            "inactive_alpha":    int(self._inactive_alpha_var.get()),
            "draw_shadow":       self._shadow_var.get(),
            "stroke_width":      int(self._stroke_width_var.get()),
            "stroke_color_rgba": (
                self._stroke_color_btn.get_rgba(alpha=stroke_alpha)
                if self._stroke_color_btn else (0, 0, 0, 200)
            ),
            "highlight_mode":    self._highlight_mode_var.get(),
        }

    def _request_preview(self, *_):
        if hasattr(self, "_preview"):
            self._preview.schedule_update(self._get_preview_params())

    # ── Transcription ─────────────────────────────────────────────────────────

    def _start_transcribe(self):
        video = self._video_var.get().strip()
        L = self._ui_lang_var.get()
        if not video or not os.path.isfile(video):
            messagebox.showerror("Error", _t(L, "err_no_video"))
            return

        model_name   = self._whisper_model_var.get()
        lang         = self._whisper_lang_var.get()
        words_per_ln = int(self._words_per_line_var.get())

        self._transcribe_btn.configure(state="disabled", text=_t(L, "transcribing"))
        self._subtitle_lines = None
        self._transcribe_status.configure(
            text=_t(L, "transcribing"), text_color="gray60")

        self._progress.set(0)
        self._progress_label.configure(text="")

        def worker():
            try:
                lines = transcribe_to_subtitle_lines(
                    video_path=video,
                    model_name=model_name,
                    language=lang,
                    max_words_per_line=words_per_ln,
                    log_fn=self._log_thread_safe,
                    progress_fn=self._update_progress,
                )
                self._subtitle_lines = lines
                total_words = sum(len(ln) for ln in lines)
                L = self._ui_lang_var.get()
                msg = _t(L, "transcribe_done", len(lines), total_words)
                self.after(0, lambda m=msg: self._on_transcribe_done(True, m))
            except Exception as exc:
                L = self._ui_lang_var.get()
                msg = _t(L, "transcribe_fail", str(exc))
                self.after(0, lambda m=msg: self._on_transcribe_done(False, m))

        self._transcribe_thread = threading.Thread(target=worker, daemon=True)
        self._transcribe_thread.start()

    def _on_transcribe_done(self, success: bool, msg: str):
        self._transcribe_btn.configure(state="normal", text=_t(self._ui_lang_var.get(), "transcribe_btn"))
        colour = "#4CAF50" if success else "#F44336"
        self._transcribe_status.configure(text=msg, text_color=colour)
        self._log(msg)
        self._progress.set(1.0 if success else 0.0)
        L2 = self._ui_lang_var.get()
        self._progress_label.configure(text=_t(L2, "ready") if success else _t(L2, "failed"))
        # Auto-open editor after successful transcription
        if success and self._subtitle_lines:
            self.after(300, self._open_subtitle_editor)

    # ── Render ────────────────────────────────────────────────────────────────

    # ── Subtitle editor ───────────────────────────────────────────────────────

    def _open_subtitle_editor(self):
        L = self._ui_lang_var.get()
        if not self._subtitle_lines:
            messagebox.showinfo("", _t(L, "no_subs_edit"))
            return

        editor = ctk.CTkToplevel(self)
        editor.title(_t(L, "edit_subs_title"))
        editor.geometry("780x600")
        editor.resizable(True, True)
        editor.grab_set()  # modal

        # Header hint
        ctk.CTkLabel(
            editor,
            text=_t(L, "edit_subs_hint"),
            text_color="gray60",
            font=ctk.CTkFont(size=11),
            anchor="w",
            wraplength=740,
        ).pack(fill="x", padx=16, pady=(12, 4))

        # Text area – one line per subtitle block, words separated by spaces
        textbox = ctk.CTkTextbox(
            editor,
            font=ctk.CTkFont(family="Courier", size=13),
        )
        textbox.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        # Populate with current lines
        lines_text = []
        for line_words in self._subtitle_lines:
            lines_text.append(" ".join(w["text"] for w in line_words))
        textbox.insert("1.0", "\n".join(lines_text))

        def save_and_close():
            raw = textbox.get("1.0", "end").strip()
            text_lines = [l.strip() for l in raw.splitlines() if l.strip()]
            new_lines: list[list[dict]] = []
            for i, text_line in enumerate(text_lines):
                if i < len(self._subtitle_lines):
                    orig = self._subtitle_lines[i]
                else:
                    # Extra lines: append after last with estimated timing
                    last_end = self._subtitle_lines[-1][-1]["end"] if self._subtitle_lines else 0.0
                    gap = 2.0
                    word_list = text_line.split()
                    dur = max(gap, len(word_list) * 0.4)
                    orig = []
                    t = last_end + 0.5
                    for word in word_list:
                        wd = dur / max(len(word_list), 1)
                        orig.append({"text": word, "start": t, "end": t + wd})
                        t += wd

                new_words_text = text_line.split()
                # Redistribute timings from original line
                orig_words = [w for w in orig]
                n_orig = len(orig_words)
                n_new  = len(new_words_text)
                if n_orig == 0:
                    continue
                line_start = orig_words[0]["start"]
                line_end   = orig_words[-1]["end"]
                total_dur  = max(line_end - line_start, MIN_WORD_DURATION * n_new)
                new_word_dur = total_dur / max(n_new, 1)
                rebuilt: list[dict] = []
                for j, wt in enumerate(new_words_text):
                    ws = line_start + j * new_word_dur
                    we = ws + new_word_dur
                    rebuilt.append({"text": wt, "start": round(ws, 3), "end": round(we, 3)})
                if rebuilt:
                    new_lines.append(rebuilt)

            self._subtitle_lines = new_lines
            L2 = self._ui_lang_var.get()
            self._transcribe_status.configure(
                text=_t(L2, "edit_subs_warn"), text_color="#4CAF50")
            self._log(_t(L2, "edit_subs_warn") + f" ({len(new_lines)} lines)")
            editor.destroy()

        btn_row = ctk.CTkFrame(editor, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 12))

        ctk.CTkButton(
            btn_row,
            text=_t(L, "edit_subs_save"),
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
            command=save_and_close,
        ).pack(side="left", fill="x", expand=True, padx=(0, 8))

        ctk.CTkButton(
            btn_row,
            text=_t(L, "edit_subs_cancel"),
            height=38,
            width=120,
            fg_color="gray30",
            hover_color="gray40",
            command=editor.destroy,
        ).pack(side="left")

    def _start_render(self):
        L = self._ui_lang_var.get()
        video  = self._video_var.get().strip()
        output = self._output_var.get().strip()
        font_name = self._font_var.get().strip()

        if not video or not os.path.isfile(video):
            messagebox.showerror("Error", _t(L, "err_no_video"))
            return
        if not output:
            messagebox.showerror("Error", _t(L, "err_no_output"))
            return
        if font_name not in self._fonts:
            messagebox.showerror("Error", _t(L, "err_no_font", font_name))
            return

        # Determine subtitle source (transcription only)
        if self._subtitle_lines is not None:
            subtitle_source: list = self._subtitle_lines
            self._log("Using Whisper transcription for subtitles.")
        else:
            if not messagebox.askyesno("No subtitles", _t(L, "no_subs_q")):
                return
            self._log(_t(L, "auto_transcribe"))
            try:
                lines = transcribe_to_subtitle_lines(
                    video_path=video,
                    model_name=self._whisper_model_var.get(),
                    language=self._whisper_lang_var.get(),
                    max_words_per_line=int(self._words_per_line_var.get()),
                    log_fn=self._log_thread_safe,
                    progress_fn=self._update_progress,
                )
                self._subtitle_lines = lines
                subtitle_source = lines
            except Exception as exc:
                messagebox.showerror(_t(L, "transcribe_fail2"), str(exc))
                return

        self._last_output_path = output
        font_path = self._fonts[font_name]
        font_size = int(self._font_size_var.get())
        box_rgba  = self._color_btn.get_rgba(alpha=int(self._alpha_var.get()))
        vpos      = float(self._vpos_var.get())
        text_color_mode = self._text_color_var.get()
        inactive_alpha  = int(self._inactive_alpha_var.get())
        draw_shadow     = self._shadow_var.get()
        stroke_width    = int(self._stroke_width_var.get())
        stroke_color_rgba = (
            self._stroke_color_btn.get_rgba(alpha=200)
            if self._stroke_color_btn else (0, 0, 0, 200)
        )
        highlight_mode = self._highlight_mode_var.get()

        self._render_btn.configure(state="disabled", text=_t(self._ui_lang_var.get(), "rendering"))
        self._open_dir_btn.configure(state="disabled")
        self._progress.configure(mode="determinate")
        self._progress.set(0)
        self._progress_label.configure(text="0%")

        self._log(f"\n{'─' * 50}")
        self._log("Render started")
        self._log(f"Font : {font_name}  |  Size: {font_size}px")
        self._log(f"Mode : {highlight_mode}  |  Position: {int(vpos * 100)}%")
        self._log(f"Color: {self._color_btn.get_hex()}  |  Text: {text_color_mode}")
        self._log(f"Stroke: {stroke_width}px  |  Shadow: {draw_shadow}")

        # Read quality selection
        q_idx = getattr(self, "_quality_idx", 0)
        _, crf, ffmpeg_preset = QUALITY_OPTIONS[q_idx]
        
        self._log(f"Quality: CRF={crf}  |  Preset: {ffmpeg_preset}")
        self._log(f"{'─' * 50}")

        self._render_thread = threading.Thread(
            target=render_dynamic_subs,
            args=(
                video, subtitle_source, output,
                font_path, font_size, box_rgba, vpos,
                text_color_mode, inactive_alpha, draw_shadow,
                stroke_width, stroke_color_rgba,
                highlight_mode,
                self._log_thread_safe,
                self._on_render_done,
                self._update_progress,
                crf,
                ffmpeg_preset,
            ),
            daemon=True,
        )
        self._render_thread.start()

    # ── Progress & callbacks ──────────────────────────────────────────────────

    def _update_progress(self, value: float):
        def _set(v=value):
            self._progress.set(v)
            self._progress_label.configure(text=f"{int(v * 100)}%")
        self.after(0, _set)

    def _on_render_done(self, success: bool, msg: str):
        def update():
            self._progress.set(1.0 if success else 0.0)
            L = self._ui_lang_var.get()
            self._progress_label.configure(
                text=_t(L, "ready") if success else _t(L, "failed"))
            self._render_btn.configure(state="normal", text=_t(L, "render_btn"))
            if success:
                self._open_dir_btn.configure(state="normal")
            self._log(msg)
            if success:
                messagebox.showinfo(_t(L, "render_done"), msg)
            else:
                messagebox.showerror(_t(L, "render_fail"), msg)
        self.after(0, update)

    def _open_output_folder(self):
        folder = os.path.dirname(self._last_output_path)
        if not folder or not os.path.isdir(folder):
            folder = os.path.expanduser("~")
        system = platform.system()
        if system == "Windows":
            os.startfile(folder)
        elif system == "Darwin":
            os.system(f'open "{folder}"')
        else:
            os.system(f'xdg-open "{folder}"')

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        self._log_box.configure(state="normal")
        self._log_box.insert("end", msg + "\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _log_thread_safe(self, msg: str):
        self.after(0, lambda m=msg: self._log(m))


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()