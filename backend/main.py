import os
import logging
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from groq import Groq

from extraction.extract_skills import extract_skills
from scoring.recommend import score_all_occupations, get_occupation_detail
from nlg.reply import generate_spoken_reply
from tts.speak import synthesize_speech

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SIH26097 Voice Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your deployed frontend origin before final submission
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SUPPORTED_LANGUAGES = {"hi", "en"}


def _load_csv(*candidate_paths: str) -> pd.DataFrame:
    """Try a list of candidate relative paths (handles running from
    repo root vs. from inside backend/) and return the first that exists."""
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception as e:
                logger.warning(f"Found {path} but failed to parse it: {e}")
    logger.warning(f"None of these paths exist: {candidate_paths}")
    return pd.DataFrame()


# Real curated datasets (NOT the placeholder backend/data/occupations.csv —
# that file is Day-1 dummy data and is no longer read anywhere in this file)
occupations_df = _load_csv("../data/occupationss.csv", "data/occupationss.csv")
courses_df = _load_csv("../data/courses.csv", "data/courses.csv")
skills_vocab_df = _load_csv("../data/skills_vocab.csv", "data/skills_vocab.csv")

logger.info(
    f"Loaded datasets — occupations: {len(occupations_df)} rows, "
    f"courses: {len(courses_df)} rows, skills_vocab: {len(skills_vocab_df)} rows"
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "SIH26097 backend running",
        "datasets_loaded": {
            "occupations": len(occupations_df),
            "courses": len(courses_df),
            "skills_vocab": len(skills_vocab_df),
        },
    }


def _normalize_language(lang: str | None) -> str:
    """Only Hindi and English are supported for now (per current scope).
    Anything else falls back to auto-detect (None) on the Whisper call."""
    if lang in SUPPORTED_LANGUAGES:
        return lang
    return None


def _transcribe_audio(audio_bytes: bytes, filename: str, language_hint: str | None) -> dict:
    """Calls Groq-hosted Whisper. Passing a language hint improves accuracy
    and speed (Whisper doesn't have to run language ID first) but is optional —
    if the hint is missing/invalid we let Whisper auto-detect between hi/en."""
    if not groq_client:
        logger.warning("GROQ_API_KEY not set — returning stub transcription.")
        return {"text": "dummy transcribed text", "language": language_hint or "hi"}

    lang = _normalize_language(language_hint)
    try:
        kwargs = {
            "file": (filename, audio_bytes),
            "model": "whisper-large-v3-turbo",
        }
        if lang:
            kwargs["language"] = lang  # ISO-639-1 code, Whisper accepts "hi" / "en"

        transcription = groq_client.audio.transcriptions.create(**kwargs)
        detected_lang = lang or "en"  # Whisper's response object doesn't reliably
        # surface detected language on this endpoint; if you need true
        # auto-detected language back, call with response_format="verbose_json"
        # and read transcription.language instead.
        return {"text": transcription.text, "language": detected_lang}
    except Exception as e:
        logger.error(f"Groq Whisper transcription error: {e}")
        return {"text": "", "language": lang or "hi", "error": str(e)}


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...), language: str = Form(default=None)):
    """
    Speech-to-text only. `language` is an OPTIONAL hint: "hi" or "en".
    If omitted, Whisper auto-detects (works fine for clearly Hindi or
    clearly English speech; a hint helps most with short/noisy clips
    or Hinglish code-switching).
    """
    audio_bytes = await audio.read()
    filename = audio.filename or "speech.webm"
    result = _transcribe_audio(audio_bytes, filename, language)
    return result


@app.post("/api/analyze")
async def analyze(payload: dict):
    """
    Text-in, structured-result-out. Used for the typed-input fallback,
    and internally by /api/converse after transcription.
    """
    text = payload.get("text", "")
    language = _normalize_language(payload.get("language")) or "en"

    try:
        profile = extract_skills(text, language=language)
    except Exception as e:
        logger.error(f"extract_skills exception: {e}")
        profile = {
            "skills": [], "experience_years": None, "sector_guess": "unclear",
            "occupation_guess": None, "employment_preference": "unclear",
            "mobility_constraint": None,
        }

    try:
        ranked = score_all_occupations(profile, occupations_df, courses_df, skills_vocab_df)
    except Exception as e:
        logger.error(f"score_all_occupations exception: {e}")
        ranked = []

        top = ranked[0] if ranked else None

    try:
        reply_text = generate_spoken_reply(profile, top, language=language)
    except Exception as e:
        logger.error(f"generate_spoken_reply exception: {e}")
        reply_text = (
            "Here's what I found based on what you told me."
            if language == "en"
            else "Aapne jo bataya uske hisaab se, yeh mila."
        )

    try:
        audio_b64 = synthesize_speech(reply_text, language=language)
    except Exception as e:
        logger.error(f"synthesize_speech exception: {e}")
        audio_b64 = None

    return {
        "profile": profile,
        "matches": ranked[:3],
        "top_occupation": top["occupation_name"] if top else None,
        "reply_text": reply_text,
        "reply_audio_base64": audio_b64,
        "reply_audio_mime": "audio/mpeg" if audio_b64 else None,
    }


@app.post("/api/converse")
async def converse(audio: UploadFile = File(...), language: str = Form(default=None)):
    """
    Full pipeline in one call, matching the frontend's planned contract:
    voice in -> transcript -> extracted profile -> ranked matches
    -> conversational spoken reply -> TTS audio out.

    Response shape:
    {
      "transcript": "...",
      "language": "hi" | "en",
      "profile": {...},
      "matches": [...],
      "reply_text": "...",
      "reply_audio_base64": "...",   # mp3, base64-encoded
      "reply_audio_mime": "audio/mpeg"
    }
    """
    audio_bytes = await audio.read()
    filename = audio.filename or "speech.webm"

    stt_result = _transcribe_audio(audio_bytes, filename, language)
    transcript = stt_result.get("text", "")
    detected_language = stt_result.get("language") or "en"

    if not transcript:
        # STT failed entirely — fail loud but structured, so the frontend
        # can fall back to the typed-text box instead of hanging.
        return {
            "transcript": "",
            "language": detected_language,
            "profile": None,
            "matches": [],
            "reply_text": "Sorry, I couldn't hear that clearly. Could you try again, or type your answer instead?"
            if detected_language == "en"
            else "Maaf kijiye, mujhe theek se sunayi nahi diya. Kripya dobara boliye ya type kar dijiye.",
            "reply_audio_base64": None,
            "reply_audio_mime": None,
            "error": stt_result.get("error", "empty_transcript"),
        }

    try:
        profile = extract_skills(transcript, language=detected_language)
    except Exception as e:
        logger.error(f"extract_skills exception: {e}")
        profile = {
            "skills": [], "experience_years": None, "sector_guess": "unclear",
            "occupation_guess": None, "employment_preference": "unclear",
            "mobility_constraint": None,
        }

    try:
        ranked = score_all_occupations(profile, occupations_df, courses_df, skills_vocab_df)
    except Exception as e:
        logger.error(f"score_all_occupations exception: {e}")
        ranked = []

    top_match = ranked[0] if ranked else None

    try:
        reply_text = generate_spoken_reply(profile, top_match, language=detected_language)
    except Exception as e:
        logger.error(f"generate_spoken_reply exception: {e}")
        reply_text = (
            "Here's what I found based on what you told me."
            if detected_language == "en"
            else "Aapne jo bataya uske hisaab se, yeh mila."
        )

    try:
        audio_b64 = synthesize_speech(reply_text, language=detected_language)
    except Exception as e:
        logger.error(f"synthesize_speech exception: {e}")
        audio_b64 = None

    return {
        "transcript": transcript,
        "language": detected_language,
        "profile": profile,
        "matches": ranked[:3],
        "reply_text": reply_text,
        "reply_audio_base64": audio_b64,
        "reply_audio_mime": "audio/mpeg" if audio_b64 else None,
    }


@app.post("/api/speak")
async def speak(payload: dict):
    """
    Standalone TTS: given text (+ optional language), returns raw mp3 bytes.
    Useful for a manual "listen again" button, or for the typed-input flow
    where you want to speak /api/analyze's result without re-running STT.
    """
    text = payload.get("text", "")
    language = _normalize_language(payload.get("language")) or "en"
    if not text:
        return Response(content=b"", media_type="audio/mpeg", status_code=400)

    audio_b64 = synthesize_speech(text, language=language)
    if not audio_b64:
        return Response(content=b"", media_type="audio/mpeg", status_code=500)

    import base64
    audio_bytes = base64.b64decode(audio_b64)
    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.get("/api/occupation/{occupation_name}")
async def get_occupation(occupation_name: str):
    detail = get_occupation_detail(occupation_name, occupations_df, courses_df)
    if detail is None:
        return {"error": "occupation not found", "id": occupation_name}
    return detail


@app.get("/api/sessions/recent")
async def recent_sessions():
    # Intentionally out of scope for this pass — see Part 4 of the plan doc
    # for the SQLite `sessions` table if/when you add it.
    return {"sessions": []}
