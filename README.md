# 🎵 TikTok-Style Karaoke Subtitle Renderer

A powerful, standalone desktop application built with Python and `customtkinter` that burns dynamic, word-by-word highlighted karaoke subtitles onto videos. It features an integrated AI transcription pipeline driven by `faster-whisper`, meaning no external subtitle files (like `.ass` or `.srt`) are required to get started.

Additionally, for professional video editors, the app supports exporting a standalone, transparent subtitle track as a high-quality `.mov` video with an alpha channel.

---

## ✨ Features

- **Automated AI Transcription:** Powered by `faster-whisper` (utilizing CTranslate2 backend) for blazing-fast, word-level timestamp generation.
- **Privacy-First & Local Processing:** Audio extraction and AI transcription run entirely on your local machine—no external API keys or cloud internet dependency required.
- **Dual Presentation Modes:**
  - **Box Mode:** Active words receive a smooth, rounded rectangle background with customizable colors and drop-shadow effects.
  - **Color Mode:** Inactive words stay softly dimmed, while the active word dynamically transitions to your selected highlight color.
- **Advanced UI Customization:**
  - Dynamic words-per-line scaling (from 2 to 10 words per text block).
  - Built-in live layout & font styling system previews using actual frame grabs from your source video.
  - Over 10 stylized appearance presets (e.g., *TikTok Yellow*, *Purple Haze*, *Neon Green*).
  - Comprehensive control over font sizing, custom system font mapping, vertical canvas positioning, outline strokes, and opacities.
- **On-the-Fly Editing:** Built-in interactive subtitle editor allows manual corrections of text adjustments before final rendering while preserving strict word timings.
- **Pro Video Editor Export:** Native support for exporting subtitle overlays into transparent alpha-channel ProRes 4444 `.mov` assets for effortless compositing in Premiere Pro, DaVinci Resolve, or After Effects.
- **Multi-lingual Support:** Dual native interface layouts for English and Ukrainian languages.

---

## 🛠️ Architecture & Requirements

The core pipeline leverages `moviepy` and `Pillow` for frame composition and multi-threaded video encoding using `libx264`. 

### Prerequisites
Before running the utility, ensure you have system-wide binaries for `ffmpeg` installed and added to your system path environment variables.

### Hardware Acceleration (Optional)
The system automatically checks for a compatible CUDA-enabled environment (NVIDIA GPU). If detected, it boots `faster-whisper` in `float16` mode via GPU execution. If a compatible GPU is missing, it seamlessly drops back to multi-threaded `int8` CPU processing.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/BlackPencil-69/render_subs.git
cd render_subs

```

### 2. Install Dependencies

Install the required packages using Python's package manager:

```bash
pip install -r requirements.txt

```

### 3. Run the Tool

Fire up the graphical user interface by launching the main execution file:

```bash
python render_subs.py

```

---

## 📦 Requirements Layout (`requirements.txt`)

Make sure your directory contains a `requirements.txt` consisting of:

```text
customtkinter
numpy
Pillow
moviepy
faster-whisper
torch
ctranslate2

```

---

## 🗺️ Transliteration Note

This tool handles multi-language transcription seamlessly. For Japanese voice or audio processing tasks, subtitle generation layout and mapping structures inherently preserve standard conventions, ensuring consistency across complex linguistic translations.

---

## 📜 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).
