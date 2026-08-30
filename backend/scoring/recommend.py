import ast
import logging

import pandas as pd
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


def _parse_qp_codes(raw) -> list[str]:
    """linked_qp_codes in occupationss.csv is stored as a Python-list-literal
    string, e.g. "['HSS/Q1015', 'HSS/Q1016']"."""
    try:
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, list):
            return [str(c).strip() for c in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def _parse_vocab_phrases(raw) -> list[str]:
    try:
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, list):
            return [str(p).lower() for p in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def _to_bool(value) -> bool:
    """occupationss.csv / courses.csv store True/False as strings when
    read back through some pandas versions — normalize defensively."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def _courses_for_occupation(occupation_row, courses_df: pd.DataFrame) -> pd.DataFrame:
    qp_codes = _parse_qp_codes(occupation_row.get("linked_qp_codes", "[]"))
    if not qp_codes or courses_df.empty:
        return pd.DataFrame()
    return courses_df[courses_df["qp_code"].isin(qp_codes)]


def _rank_courses_for_profile(courses: pd.DataFrame, profile: dict) -> list[dict]:
    """Within one occupation's linked courses, prioritize the ones that
    actually fit the stated mobility constraint and employment preference —
    this is the PS's own "mobility and physical constraints" /
    "self-employment or wage employment preference" requirement, not just
    a generic top-N list."""
    wants_self_employment = profile.get("employment_preference") == "self"
    has_mobility_constraint = profile.get("mobility_constraint") is True

    scored_rows = []
    for _, row in courses.iterrows():
        fit_score = 0
        if has_mobility_constraint and _to_bool(row.get("mobility_constraint_safe", False)):
            fit_score += 2
        if wants_self_employment and _to_bool(row.get("self_employment_possible", False)):
            fit_score += 2
        scored_rows.append((fit_score, row))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    result = []
    for fit_score, row in scored_rows[:2]:
        result.append({
            "qp_code": row.get("qp_code"),
            "job_role": row.get("job_role"),
            "nsqf_level": row.get("nsqf_level"),
            "sector": row.get("sector"),
            "duration_hours": row.get("duration_hours"),
            "eligibility": row.get("eligibility"),
            "self_employment_possible": _to_bool(row.get("self_employment_possible", False)),
            "mobility_constraint_safe": _to_bool(row.get("mobility_constraint_safe", False)),
            "fits_stated_constraints": fit_score > 0,
        })
    return result


def score_all_occupations(
    profile: dict,
    occupations_df: pd.DataFrame,
    courses_df: pd.DataFrame,
    skills_vocab_df: pd.DataFrame,
) -> list[dict]:
    """
    Ranks every occupation in occupationss.csv against the extracted profile.
    Scoring signals (all explainable — kept as separate fields in the
    response rather than folded into one opaque number):
      - direct occupation_guess match from the LLM/vocab extraction
      - sector_guess fuzzy match against the occupation name
      - skill-phrase overlap via skills_vocab.csv
      - experience bonus
      - mobility-constraint alignment with the occupation's mobility_safe flag
    """
    if occupations_df is None or occupations_df.empty:
        logger.warning("occupations_df is empty — check that data/occupationss.csv was found.")
        return []

    candidate_skills = [s.lower() for s in profile.get("skills", [])]
    occupation_guess = (profile.get("occupation_guess") or "").lower()
    sector_guess = (profile.get("sector_guess") or "").lower()
    experience_years = profile.get("experience_years") or 0
    mobility_constraint = profile.get("mobility_constraint")

    # Pre-index skills_vocab.csv rows by occupation name for quick lookup
    vocab_by_occupation: dict[str, list[str]] = {}
    if not skills_vocab_df.empty:
        for _, row in skills_vocab_df.iterrows():
            occ = str(row.get("mapped_skill", ""))
            phrases = _parse_vocab_phrases(row.get("user_phrase_variations", "[]"))
            vocab_by_occupation.setdefault(occ, []).extend(phrases)

    results = []
    for _, occ_row in occupations_df.iterrows():
        occupation_name = str(occ_row.get("occupation_name", ""))
        occ_name_lower = occupation_name.lower()
        mobility_safe = _to_bool(occ_row.get("mobility_safe", False))
        is_traditional = _to_bool(occ_row.get("is_tritional_occupation", False))

        score = 0.0
        reasons = []

        # 1. Direct occupation guess from extraction
        name_similarity = fuzz.partial_ratio(occupation_guess, occ_name_lower) if occupation_guess else 0
        if name_similarity > 80:
            score += 15
            reasons.append("direct occupation match")

        # 2. Sector guess overlap
        sector_similarity = fuzz.partial_ratio(sector_guess, occ_name_lower) if sector_guess else 0
        if sector_similarity > 75:
            score += 5
            reasons.append("sector match")

        # 3. Skill-phrase overlap via skills_vocab.csv
        vocab_phrases = vocab_by_occupation.get(occupation_name, [])
        skill_hits = 0
        for skill in candidate_skills:
            for phrase in vocab_phrases:
                if fuzz.partial_ratio(skill, phrase) > 75:
                    skill_hits += 1
                    break
        if skill_hits:
            score += min(skill_hits, 3) * 8
            reasons.append(f"{skill_hits} matched skill phrase(s)")

        # 4. Experience bonus (capped, doesn't dominate the score)
        score += min(experience_years, 5) * 1.5

        # 5. Mobility-constraint alignment — surfaced explicitly, not hidden
        if mobility_constraint is True:
            if mobility_safe:
                score += 5
                reasons.append("suitable for stated mobility constraint")
            else:
                score -= 5
                reasons.append("may not suit stated mobility constraint")

        if score <= 0:
            continue  # don't show occupations with literally no signal at all

        matched_courses = _courses_for_occupation(occ_row, courses_df)
        recommended_courses = _rank_courses_for_profile(matched_courses, profile)

        results.append({
            "occupation_name": occupation_name,
            "score": round(score, 2),
            "is_traditional_occupation": is_traditional,
            "mobility_safe": mobility_safe,
            "match_reasons": reasons,
            "recommended_courses": recommended_courses,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def get_occupation_detail(occupation_name: str, occupations_df: pd.DataFrame, courses_df: pd.DataFrame) -> dict | None:
    """Backs the GET /api/occupation/{occupation_name} endpoint — full detail
    view for when the user clicks into one recommended occupation."""
    if occupations_df.empty:
        return None

    match = occupations_df[occupations_df["occupation_name"].str.lower() == occupation_name.lower()]
    if match.empty:
        return None

    occ_row = match.iloc[0]
    courses = _courses_for_occupation(occ_row, courses_df)
    course_list = []
    for _, row in courses.iterrows():
        course_list.append({
            "qp_code": row.get("qp_code"),
            "job_role": row.get("job_role"),
            "nsqf_level": row.get("nsqf_level"),
            "sector": row.get("sector"),
            "duration_hours": row.get("duration_hours"),
            "eligibility": row.get("eligibility"),
            "self_employment_possible": _to_bool(row.get("self_employment_possible", False)),
            "mobility_constraint_safe": _to_bool(row.get("mobility_constraint_safe", False)),
        })

    return {
        "occupation_name": occ_row.get("occupation_name"),
        "is_traditional_occupation": _to_bool(occ_row.get("is_tritional_occupation", False)),
        "mobility_safe": _to_bool(occ_row.get("mobility_safe", False)),
        "courses": course_list,
    }
