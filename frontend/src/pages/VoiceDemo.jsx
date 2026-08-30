import { useState, useRef } from "react";
import { useLanguage } from "../context/LanguageContext";
import Waveform from "../components/Waveform";
import { converse, analyzeText, base64ToAudioUrl } from "../lib/api";

// Real, backend-connected voice assistant screen.
// Flow: idle -> recording -> processing -> speaking -> idle
//   idle       — waiting for the user to press the mic or type
//   recording  — MediaRecorder is capturing audio
//   processing — waiting on /api/converse (STT -> extraction -> matching -> reply -> TTS)
//   speaking   — playing the returned reply audio aloud
export default function VoiceDemo() {
  const { lang, toggleLanguage } = useLanguage(); // "en" | "hi"
  const [status, setStatus] = useState("idle");
  const [errorMsg, setErrorMsg] = useState(null);
  const [typedText, setTypedText] = useState("");
  const [result, setResult] = useState(null); // last full backend response

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioPlayerRef = useRef(null);

  async function startRecording() {
    setErrorMsg(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      audioChunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        await handleVoiceSubmit(audioBlob);
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setStatus("recording");
    } catch (err) {
      console.error("Mic access error:", err);
      setErrorMsg(
        lang === "en"
          ? "Couldn't access the microphone. You can type your answer below instead."
          : "Microphone tak pahunch nahi payi. Aap neeche type kar sakte hain."
      );
    }
  }

  function stopRecording() {
    if (mediaRecorderRef.current && status === "recording") {
      mediaRecorderRef.current.stop();
      setStatus("processing");
    }
  }

  async function handleVoiceSubmit(audioBlob) {
    setStatus("processing");
    try {
      const data = await converse(audioBlob, lang);
      applyResult(data);
    } catch (err) {
      console.error("converse() failed:", err);
      setErrorMsg(
        lang === "en"
          ? "Something went wrong reaching the assistant. Please try typing instead."
          : "Assistant tak pahunchne mein dikkat aayi. Kripya type karke try karein."
      );
      setStatus("idle");
    }
  }

  async function handleTypeSubmit(e) {
    e.preventDefault();
    if (!typedText.trim()) return;
    setStatus("processing");
    setErrorMsg(null);
    try {
      const data = await analyzeText(typedText, lang);
      setTypedText("");
      applyResult(data);
    } catch (err) {
      console.error("analyzeText() failed:", err);
      setErrorMsg(
        lang === "en"
          ? "Something went wrong. Please try again."
          : "Kuch galat ho gaya. Kripya dobara koshish karein."
      );
      setStatus("idle");
    }
  }

  function applyResult(data) {
    setResult(data);
    const audioUrl = base64ToAudioUrl(data.reply_audio_base64, data.reply_audio_mime);
    if (audioUrl && audioPlayerRef.current) {
      audioPlayerRef.current.src = audioUrl;
      setStatus("speaking");
      audioPlayerRef.current
        .play()
        .catch(() => {
          // Autoplay can be blocked by the browser until the user interacts —
          // the reply text is still shown, and there's a manual "Play reply" button.
          setStatus("idle");
        });
    } else {
      setStatus("idle");
    }
  }

  function handleAudioEnded() {
    setStatus("idle");
  }

  function replayAudio() {
    if (audioPlayerRef.current && audioPlayerRef.current.src) {
      setStatus("speaking");
      audioPlayerRef.current.play().catch(() => setStatus("idle"));
    }
  }

  function reset() {
    setResult(null);
    setErrorMsg(null);
    setStatus("idle");
  }

  return (
    <div className="max-w-2xl mx-auto px-4 sm:px-8 py-12">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900">
          {lang === "en" ? "Talk to Pipo" : "Pipo se baat karein"}
        </h1>
        <button
          onClick={toggleLanguage}
          className="text-xs font-mono uppercase tracking-wider border border-emerald-900/20 rounded-full px-3 py-1.5 text-[#0a5c2b] hover:bg-emerald-900/5"
        >
          {lang === "en" ? "हिंदी" : "English"}
        </button>
      </div>

      {/* Hidden audio element used to play the assistant's spoken reply */}
      <audio ref={audioPlayerRef} onEnded={handleAudioEnded} className="hidden" />

      {!result ? (
        <>
          <div className="bg-white border border-emerald-900/10 rounded-2xl p-8 min-h-[260px] flex flex-col items-center justify-center gap-6">
            <button
              onClick={status === "recording" ? stopRecording : startRecording}
              disabled={status === "processing" || status === "speaking"}
              className={`w-24 h-24 rounded-full flex items-center justify-center shadow-lg transition-colors disabled:opacity-50 ${
                status === "recording" ? "bg-red-600 animate-pulse" : "bg-[#0a5c2b] hover:bg-[#074720]"
              }`}
            >
              <MicIcon status={status} />
            </button>

            <p className="font-mono text-xs uppercase tracking-widest text-slate-500">
              {status === "recording" && (lang === "en" ? "Listening... tap to stop" : "Sun raha hoon... rokne ke liye tap karein")}
              {status === "processing" && (lang === "en" ? "Thinking..." : "Soch raha hoon...")}
              {status === "speaking" && (lang === "en" ? "Speaking..." : "Bol raha hoon...")}
              {status === "idle" && (lang === "en" ? "Tap the mic to speak" : "Bolne ke liye mic dabayein")}
            </p>

            {status === "recording" && <Waveform active bars={20} color="emerald" className="h-8" />}
          </div>

          {errorMsg && (
            <p className="mt-4 text-sm text-red-600 text-center">{errorMsg}</p>
          )}

          <div className="mt-6">
            <p className="font-mono text-[11px] uppercase tracking-widest text-slate-400 text-center mb-3">
              {lang === "en" ? "or type instead" : "ya type karein"}
            </p>
            <form onSubmit={handleTypeSubmit} className="flex gap-2">
              <input
                type="text"
                value={typedText}
                onChange={(e) => setTypedText(e.target.value)}
                placeholder={lang === "en" ? "Type your answer here..." : "Yahan type karein..."}
                disabled={status === "processing"}
                className="flex-1 bg-white border border-emerald-900/15 rounded-full px-5 py-3 text-sm placeholder:text-slate-400 focus:outline-none focus:border-[#0a5c2b]"
              />
              <button
                type="submit"
                disabled={status === "processing"}
                className="bg-[#0a5c2b] text-white text-sm font-semibold px-6 py-3 rounded-full hover:bg-[#074720] transition-colors disabled:opacity-50"
              >
                {lang === "en" ? "Send" : "Bhejein"}
              </button>
            </form>
          </div>
        </>
      ) : (
        <ResultCard
          data={result}
          lang={lang}
          onReplay={replayAudio}
          isSpeaking={status === "speaking"}
          onReset={reset}
        />
      )}
    </div>
  );
}

function ResultCard({ data, lang, onReplay, isSpeaking, onReset }) {
  const profile = data.profile || {};
  const matches = data.matches || [];

  return (
    <div className="bg-[#0a5c2b] text-white rounded-2xl p-8 shadow-xl">
      {data.transcript && (
        <p className="text-xs text-emerald-100/70 italic mb-4">
          {lang === "en" ? "You said: " : "Aapne kaha: "}"{data.transcript}"
        </p>
      )}

      <div className="flex items-center justify-between mb-4">
        <p className="font-mono text-xs uppercase tracking-widest text-emerald-200">
          {lang === "en" ? "Profile Summary" : "Profile Summary"}
        </p>
        <span className="bg-white/10 text-emerald-100 text-xs font-mono px-3 py-1 rounded-full">
          {profile.sector_guess || (lang === "en" ? "Unclear" : "Spasht nahi")}
        </span>
      </div>

      <ul className="text-white/90 space-y-1 mb-6 text-sm bg-white/10 p-4 rounded-xl">
        <li><strong>{lang === "en" ? "Skills" : "Skills"}:</strong> {(profile.skills || []).join(", ") || "—"}</li>
        <li><strong>{lang === "en" ? "Experience" : "Anubhav"}:</strong> {profile.experience_years ?? "—"} {lang === "en" ? "years" : "saal"}</li>
        <li><strong>{lang === "en" ? "Preference" : "Pasand"}:</strong> {profile.employment_preference || "unclear"}</li>
      </ul>

      {data.reply_text && (
        <div className="mb-6 bg-white/10 rounded-xl p-4 flex items-start justify-between gap-3">
          <p className="text-sm italic text-emerald-50">"{data.reply_text}"</p>
          <button
            onClick={onReplay}
            disabled={isSpeaking}
            className="shrink-0 text-xs font-mono uppercase bg-white/20 hover:bg-white/30 disabled:opacity-50 px-3 py-1.5 rounded-full"
          >
            {isSpeaking ? "▶ …" : "🔊 " + (lang === "en" ? "Replay" : "Dobara")}
          </button>
        </div>
      )}

      <p className="font-mono text-xs uppercase tracking-widest text-emerald-200 mb-3">
        {lang === "en" ? "Recommended Pathways" : "Sujhaaye gaye raaste"}
      </p>

      <div className="space-y-3 mb-6">
        {matches.length === 0 && (
          <p className="text-sm text-white/70">
            {lang === "en" ? "No strong match found — try sharing a bit more detail." : "Koi strong match nahi mila — thoda aur bataiye."}
          </p>
        )}
        {matches.map((occ, idx) => (
          <div key={idx} className="bg-white/10 rounded-xl p-4 border border-white/10">
            <div className="flex justify-between items-start mb-1">
              <p className="font-semibold text-sm">{occ.occupation_name}</p>
              <span className="text-emerald-200 font-mono text-xs">{occ.score}</span>
            </div>
            {occ.match_reasons?.length > 0 && (
              <p className="text-xs text-white/60 mb-2">{occ.match_reasons.join(" · ")}</p>
            )}
            {occ.recommended_courses?.map((course, ci) => (
              <p key={ci} className="text-xs text-emerald-100">
                → {course.job_role} (NSQF {course.nsqf_level}, {course.duration_hours}h)
              </p>
            ))}
          </div>
        ))}
      </div>

      <button
        onClick={onReset}
        className="text-emerald-200 text-sm font-semibold underline underline-offset-4 hover:text-white"
      >
        {lang === "en" ? "← Ask again" : "← Dobara poochein"}
      </button>
    </div>
  );
}

function MicIcon({ status }) {
  if (status === "processing" || status === "speaking") {
    return (
      <svg className="w-8 h-8 text-white animate-spin" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
    );
  }
  return (
    <svg className="w-9 h-9 text-white" fill="none" viewBox="0 0 24 24">
      <rect x="9" y="2" width="6" height="12" rx="3" fill="currentColor" />
      <path d="M5 11a7 7 0 0 0 14 0M12 18v3" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  );
}
