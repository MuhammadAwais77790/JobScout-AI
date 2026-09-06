import re


# ---------------------------------
# Normalize Text
# ---------------------------------

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    text = re.sub(
        r"[^a-z0-9+#.\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ---------------------------------
# Skill Matching
# ---------------------------------

def skills_are_matching(
    candidate_skill: str,
    required_skill: str
) -> bool:

    candidate = normalize_text(
        candidate_skill
    )

    required = normalize_text(
        required_skill
    )

    if not candidate or not required:
        return False

    if candidate == required:
        return True

    if (
        candidate in required
        or required in candidate
    ):
        return True

    # Common equivalent wording
    equivalent_groups = [
        {
            "seo",
            "search engine optimization"
        },
        {
            "content writing",
            "content writer",
            "blog writing",
            "blog writer",
            "article writing",
            "article writer"
        },
        {
            "microsoft office",
            "ms office"
        },
        {
            "communication",
            "communication skills"
        }
    ]

    for group in equivalent_groups:

        candidate_match = any(
            item in candidate
            for item in group
        )

        required_match = any(
            item in required
            for item in group
        )

        if candidate_match and required_match:
            return True

    return False


# ---------------------------------
# Remove Duplicate Skills
# ---------------------------------

def unique_skills(
    skills: list[str]
) -> list[str]:

    result = []
    seen = set()

    for skill in skills:

        if not isinstance(skill, str):
            continue

        skill = skill.strip()

        if not skill:
            continue

        normalized = normalize_text(
            skill
        )

        if normalized in seen:
            continue

        seen.add(normalized)
        result.append(skill)

    return result


# ---------------------------------
# Calculate Skill Match
# ---------------------------------

def calculate_skill_match(
    candidate_skills: list[str],
    required_skills: list[str]
) -> dict:

    candidate_skills = unique_skills(
        candidate_skills
    )

    required_skills = unique_skills(
        required_skills
    )

    if not required_skills:

        return {
            "score": 0,
            "matched": [],
            "missing": [],
            "status": "Unknown"
        }

    matched = []
    missing = []

    for required_skill in required_skills:

        found = False

        for candidate_skill in candidate_skills:

            if skills_are_matching(
                candidate_skill,
                required_skill
            ):

                matched.append(
                    required_skill
                )

                found = True
                break

        if not found:

            missing.append(
                required_skill
            )

    score = round(
        (
            len(matched)
            /
            len(required_skills)
        ) * 100
    )

    if score >= 80:
        status = "Excellent"

    elif score >= 60:
        status = "Good"

    elif score >= 40:
        status = "Moderate"

    else:
        status = "Low"

    return {
        "score": score,
        "matched": matched,
        "missing": missing,
        "status": status
    }


# ---------------------------------
# Final Job Match Score
# ---------------------------------

def calculate_match_score(
    candidate_skills: list[str],
    job_required_skills: list[str],
    candidate_experience: list[dict] | None = None,
    job_experience_requirements: list[str] | None = None,
    candidate_education: list[dict] | None = None,
    job_education_requirements: list[str] | None = None
) -> dict:

    candidate_experience = (
        candidate_experience or []
    )

    job_experience_requirements = (
        job_experience_requirements or []
    )

    candidate_education = (
        candidate_education or []
    )

    job_education_requirements = (
        job_education_requirements or []
    )

    # -------------------------------
    # Skill Score
    # -------------------------------

    skill_result = calculate_skill_match(
        candidate_skills,
        job_required_skills
    )

    skill_score = skill_result["score"]


    # -------------------------------
    # Experience Score
    # -------------------------------

    if not job_experience_requirements:

        experience_score = 100

    else:

        candidate_experience_text = (
            " ".join(
                str(item)
                for item in candidate_experience
            )
            .lower()
        )

        matched_experience = 0

        for requirement in job_experience_requirements:

            requirement_text = (
                str(requirement)
                .lower()
            )

            # Extract simple year requirement
            years = re.findall(
                r"(\d+)\+?\s*(?:years?|yrs?)",
                requirement_text
            )

            if years:

                required_years = int(
                    years[0]
                )

                candidate_years = re.findall(
                    r"(\d+)\+?\s*(?:years?|yrs?)",
                    candidate_experience_text
                )

                if candidate_years:

                    max_candidate_years = max(
                        int(year)
                        for year in candidate_years
                    )

                    if (
                        max_candidate_years
                        >= required_years
                    ):
                        matched_experience += 1

            else:

                # General relevance check
                words = [
                    word
                    for word in normalize_text(
                        requirement_text
                    ).split()
                    if len(word) > 3
                ]

                if words:

                    matches = sum(
                        1
                        for word in words
                        if word
                        in candidate_experience_text
                    )

                    if matches >= max(
                        1,
                        len(words) // 3
                    ):
                        matched_experience += 1

        experience_score = round(
            (
                matched_experience
                /
                len(job_experience_requirements)
            ) * 100
        )


    # -------------------------------
    # Education Score
    # -------------------------------

    if not job_education_requirements:

        education_score = 100

    else:

        candidate_education_text = (
            " ".join(
                str(item)
                for item in candidate_education
            )
            .lower()
        )

        matched_education = 0

        for requirement in job_education_requirements:

            requirement_text = (
                str(requirement)
                .lower()
            )

            words = [
                word
                for word in normalize_text(
                    requirement_text
                ).split()
                if len(word) > 3
            ]

            if words:

                matches = sum(
                    1
                    for word in words
                    if word
                    in candidate_education_text
                )

                if matches >= max(
                    1,
                    len(words) // 3
                ):
                    matched_education += 1

        education_score = round(
            (
                matched_education
                /
                len(job_education_requirements)
            ) * 100
        )


    # ---------------------------------
    # Weighted Final Score
    # ---------------------------------

    final_score = round(
        (
            skill_score * 0.60
            +
            experience_score * 0.25
            +
            education_score * 0.15
        )
    )


    # ---------------------------------
    # Match Level
    # ---------------------------------

    if final_score >= 80:

        match_level = "Excellent"

        recommendation = (
            "Strong match. "
            "The candidate matches most "
            "of the job requirements."
        )

    elif final_score >= 60:

        match_level = "Good"

        recommendation = (
            "Good match. "
            "The candidate has several "
            "relevant qualifications."
        )

    elif final_score >= 40:

        match_level = "Moderate"

        recommendation = (
            "Moderate match. "
            "Some relevant qualifications "
            "are present, but there are gaps."
        )

    else:

        match_level = "Low"

        recommendation = (
            "Low match. "
            "The candidate is missing "
            "several important requirements."
        )


    # ---------------------------------
    # Final Result
    # ---------------------------------

    return {

        "match_score": final_score,

        "skill_score": skill_score,

        "experience_score": experience_score,

        "education_score": education_score,

        "matched_skills": skill_result[
            "matched"
        ],

        "missing_skills": skill_result[
            "missing"
        ],

        "match_level": match_level,

        "recommendation": recommendation
    }