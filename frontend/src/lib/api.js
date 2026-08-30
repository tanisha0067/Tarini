// Small fetch wrapper for talking to the FastAPI backend.
// Set VITE_API_URL in a .env file at the frontend root when you deploy
// (e.g. VITE_API_URL=https://your-backend.onrender.com). Defaults to
// localhost for local dev.

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * Sends a recorded audio blob to /api/converse and gets back the full
 * pipeline result: transcript, extracted profile, ranked matches, a
 * conversational reply, and base64 mp3 audio of that reply.
 */
export async function converse(audioBlob, language) {
  const formData = new FormData();
  formData.append("audio", audioBlob, "speech.webm");
  if (language) formData.append("language", language);

  const res = await fetch(`${API_URL}/api/converse`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    throw new Error(`/api/converse failed: ${res.status}`);
  }
  return res.json();
}

/**
 * Text-in fallback (typed input). Requires the /api/analyze patch that
 * also returns reply_text + reply_audio_base64 — see
 * backend/PATCH_analyze_endpoint.txt.
 */
export async function analyzeText(text, language) {
  const res = await fetch(`${API_URL}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, language }),
  });
  if (!res.ok) {
    throw new Error(`/api/analyze failed: ${res.status}`);
  }
  return res.json();
}

/** Fetches full detail (linked courses) for one occupation, for a "view more" click. */
export async function getOccupationDetail(occupationName) {
  const res = await fetch(`${API_URL}/api/occupation/${encodeURIComponent(occupationName)}`);
  if (!res.ok) {
    throw new Error(`/api/occupation failed: ${res.status}`);
  }
  return res.json();
}

/** Converts a base64 mp3 string (as returned by the backend) into a playable audio URL. */
export function base64ToAudioUrl(base64, mime = "audio/mpeg") {
  if (!base64) return null;
  return `data:${mime};base64,${base64}`;
}
