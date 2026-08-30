import os
import logging

from groq import Groq

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a warm, encouraging voice assistant helping a Scheduled \
Caste community beneficiary under India's PM-AJAY scheme figure out their next \
skill-training step. You are given their extracted profile and their top matched \
occupation with recommended courses. Write a short SPOKEN reply (2-4 sentences, \
first person, as if you are talking to them directly) that:
- acknowledges what they already know/do
- names the top recommended occupation and, if useful, why it fits them
- mentions one concrete next step (a course/job-role name) if one is available
- sounds like a supportive person talking, NOT a form or a report — no bullet \
points, no field labels like "Score:" or "Sector:", no markdown

Respond in {language_name}. Keep it natural and conversational, the way you'd \
actually speak the sentence aloud. If there is no strong match, gently say so and \
suggest they share a bit more about what they enjoy doing.

Respond with ONLY the spoken reply text — no preamble, no quotation marks."""

LANGUAGE_NAMES = {"hi": "Hindi", "en": "English"}


def generate_spoken_reply(profile: dict, top_match: dict | None, language: str = "en") -> str:
    """Second LLM call: turns the structured extraction+matching result into
    a natural first-person sentence to be spoken aloud, instead of reading
    raw field values ("Score: 15.7, missing skills: ...") to the user —
    this is what the PS means by an "empathetic and conversational" system."""
    language_name = LANGUAGE_NAMES.get(language, "English")

    if not top_match:
        return (
            "I couldn't find a strong match yet — could you tell me a bit more "
            "about what kind of work you enjoy or have done before?"
            if language == "en"
            else "Mujhe abhi koi strong match nahi mila — kya aap thoda aur bata "
            "sakte hain ki aapko kis tarah ka kaam pasand hai ya pehle kya kiya hai?"
        )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return _template_fallback_reply(profile, top_match, language)

    try:
        client = Groq(api_key=api_key)
        user_content = (
            f"Beneficiary profile: {profile}\n"
            f"Top matched occupation: {top_match}"
        )
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(language_name=language_name)},
                {"role": "user", "content": user_content},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
        )
        reply = chat_completion.choices[0].message.content.strip()
        return reply if reply else _template_fallback_reply(profile, top_match, language)
    except Exception as e:
        logger.warning(f"generate_spoken_reply LLM call failed: {e}. Using template fallback.")
        return _template_fallback_reply(profile, top_match, language)


def _template_fallback_reply(profile: dict, top_match: dict, language: str) -> str:
    """No-API-key / API-down fallback — a filled-in template, still spoken
    in first person rather than reading raw JSON, so a demo never sounds
    robotic even if the LLM call fails."""
    occupation = top_match.get("occupation_name", "this path")
    courses = top_match.get("recommended_courses", [])
    course_name = courses[0]["job_role"] if courses else None

    if language == "hi":
        base = f"Aapke bataye gaye kaam ke hisaab se, {occupation} aapke liye ek accha vikalp ho sakta hai."
        if course_name:
            base += f" Isके liye, {course_name} training aapko shuru karne mein madad karegi."
        return base

    base = f"Based on what you've shared, {occupation} could be a great fit for you."
    if course_name:
        base += f" A good next step would be the {course_name} training program."
    return base
