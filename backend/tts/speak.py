import base64
import logging
from io import BytesIO

from gtts import gTTS

logger = logging.getLogger(__name__)

# gTTS language codes for the two supported languages
GTTS_LANG_MAP = {"hi": "hi", "en": "en"}


def synthesize_speech(text: str, language: str = "en") -> str | None:
    """
    Converts text to speech using gTTS (free, no API key — uses Google
    Translate's public TTS endpoint under the hood). Returns base64-encoded
    MP3 audio, or None on failure.

    Why gTTS and not Groq: Groq doesn't currently offer a TTS endpoint —
    only STT (Whisper) and LLM chat. gTTS is the simplest free option that
    supports both Hindi and English without any account/API key setup,
    which matters for a 3-day timeline. It does require internet access
    at request time (it's a thin wrapper around a Google endpoint) — if
    your deployment environment has no outbound internet, swap this for
    the browser-native Web Speech API on the frontend instead (see the
    plan doc, Part 7).
    """
    if not text or not text.strip():
        return None

    lang_code = GTTS_LANG_MAP.get(language, "en")

    try:
        tts = gTTS(text=text, lang=lang_code)
        buffer = BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        audio_bytes = buffer.read()
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        logger.error(f"gTTS synthesis failed: {e}")
        return None
