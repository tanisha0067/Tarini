import os
import json
import ast
import logging
from functools import lru_cache

import pandas as pd
from groq import Groq
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

EMPTY_PROFILE = {
    "skills": [],
    "experience_years": None,
    "sector_guess": "unclear",
    "occupation_guess": None,
    "employment_preference": "unclear",  # "self" | "wage" | "unclear"
    "mobility_constraint": None,          # True | False | None (unknown)
}

SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant that turns a beneficiary's spoken \
description of their work/skills into a structured profile, for a livelihood and \
skilling recommendation tool used by Scheduled Caste community beneficiaries under \
PM-AJAY in India. The speaker may talk in Hindi, English, or a casual mix of both \
(Hinglish), and may use informal or colloquial phrasing.

Respond ONLY with a valid JSON object, no extra text, matching exactly this structure:
{{
  "skills": ["skill1", "skill2"],
  "experience_years": 3,
  "sector_guess": "Apparel",
  "occupation_guess": "Tailor",
  "employment_preference": "self",
  "mobility_constraint": false
}}

Rules:
- "skills": short plain-language skill phrases as the person described them (e.g. "tailoring", "motorcycle repair"). Empty list if truly nothing usable was said.
- "experience_years": integer if a duration was stated or clearly implied, else null. Do not guess wildly.
- "sector_guess": one short sector label (e.g. Apparel, Healthcare, Retail, Agriculture, Construction, Beauty & Wellness, Food Processing) or "unclear".
- "occupation_guess": your best single-occupation-title guess for what this person could pursue, or null if genuinely unclear.
- "employment_preference": "self" if they express wanting to run their own work/business, "wage" if they express wanting a job/employment, else "unclear".
- "mobility_constraint": true if they mention any physical limitation, disability, difficulty traveling, or being homebound; false if they clearly state no such constraint; null if not mentioned at all.
- If you are unsure about any field, use null/"unclear" rather than inventing information. Never fabricate skills the speaker didn't mention.
- Respond in valid JSON only — no markdown fences, no commentary.

The speaker's language for this message is: {language}."""


def extract_skills(text: str, language: str = "en") -> dict:
    """
    Extracts a structured profile from freeform speech/text.
    Primary path: Groq LLM (llama-3.3-70b-versatile) with a bilingual,
    field-rich JSON prompt. Falls back to vocabulary-based fuzzy matching
    against data/skills_vocab.csv if the API key is missing or the call fails,
    so the pipeline never hard-crashes during a live demo.
    """
    if not text or not text.strip():
        return dict(EMPTY_PROFILE)

    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        try:
            client = Groq(api_key=api_key)
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(language=language)
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = chat_completion.choices[0].message.content
            parsed = json.loads(content)
            # Merge onto EMPTY_PROFILE so a partial/odd LLM response never
            # produces a KeyError downstream in the scoring module.
            profile = dict(EMPTY_PROFILE)
            profile.update({k: v for k, v in parsed.items() if k in EMPTY_PROFILE})
            return profile
        except Exception as e:
            logger.warning(f"Groq LLM extraction failed: {e}. Using vocab-based fallback.")

    return _vocab_fallback_extract(text)


@lru_cache(maxsize=1)
def _load_vocab_rows() -> list[tuple[list[str], str]]:
    """Loads and parses data/skills_vocab.csv once. Each row's
    `user_phrase_variations` column is a Python-list-literal string
    (e.g. "['nurse', 'hospital job', ...]") mapping to an occupation
    name in `mapped_skill`. Cached because this file doesn't change
    at runtime."""
    for path in ("../data/skills_vocab.csv", "data/skills_vocab.csv"):
        if os.path.exists(path):
            df = pd.read_csv(path)
            rows = []
            for _, row in df.iterrows():
                try:
                    phrases = ast.literal_eval(str(row["user_phrase_variations"]))
                    if isinstance(phrases, list):
                        rows.append((phrases, str(row["mapped_skill"])))
                except (ValueError, SyntaxError):
                    continue
            return rows
    logger.warning("skills_vocab.csv not found for fallback extraction.")
    return []


def _vocab_fallback_extract(text: str) -> dict:
    """No-API-key / API-down fallback. Fuzzy-matches the input text against
    every known phrase in skills_vocab.csv and votes for the best-matching
    occupation. Deliberately simple and dependency-light (rapidfuzz only)
    so it never fails the way an LLM call can."""
    vocab_rows = _load_vocab_rows()
    text_lower = text.lower()

    best_occupation = None
    best_score = 0
    matched_phrases = []

    for phrases, occupation_name in vocab_rows:
        for phrase in phrases:
            score = fuzz.partial_ratio(phrase.lower(), text_lower)
            if score > 70 and score > best_score:
                best_score = score
                best_occupation = occupation_name
                matched_phrases = [phrase]
            elif score > 70 and score == best_score and phrase not in matched_phrases:
                matched_phrases.append(phrase)

    profile = dict(EMPTY_PROFILE)
    if best_occupation:
        profile["skills"] = matched_phrases
        profile["occupation_guess"] = best_occupation
        profile["sector_guess"] = best_occupation  # closest signal we have without the LLM

    # Extremely light experience-year detection as a bonus signal only —
    # not a substitute for the LLM, just better than nothing.
    for digit, hindi_word in [(1, "ek"), (2, "do"), (3, "teen"), (5, "paanch")]:
        if str(digit) in text or hindi_word in text_lower:
            profile["experience_years"] = digit
            break

    if "apna" in text_lower or "khud ka" in text_lower or "own business" in text_lower or "self" in text_lower:
        profile["employment_preference"] = "self"
    elif "naukri" in text_lower or "job" in text_lower or "employment" in text_lower:
        profile["employment_preference"] = "wage"

    return profile
