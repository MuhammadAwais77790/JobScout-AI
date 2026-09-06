import re


def extract_email(text: str) -> str | None:
    """
    Extract an email address from CV text.

    Handles:
    1. Normal emails
    2. Emails split across PDF line breaks

    Example:
        awaismian7894@gmail.com

    Or if PDF extracts it as:
        awaismian7894@gm
        ail.com
    """

    # -------------------------------------------------
    # 1. Try to find a normal email first
    # -------------------------------------------------
    match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    if match:
        return match.group(0).strip()

    # -------------------------------------------------
    # 2. Handle email split by a PDF line break
    # -------------------------------------------------
    match = re.search(
        r"([A-Za-z0-9._%+-]+)"
        r"@"
        r"([A-Za-z0-9-]+)"
        r"\s+"
        r"([A-Za-z0-9-]+)"
        r"\."
        r"(com|net|org|edu|gov|pk|co\.uk)",
        text,
        re.IGNORECASE
    )

    if match:
        email = (
            f"{match.group(1)}@"
            f"{match.group(2)}"
            f"{match.group(3)}."
            f"{match.group(4)}"
        )

        return email.lower()

    return None


def extract_phone(text: str) -> str | None:
    """
    Extract a Pakistani mobile phone number from CV text.
    """

    match = re.search(
        r"(?:\+92|0092|0)?\s?3\d{2}[-\s]?\d{7}",
        text
    )

    return match.group(0).strip() if match else None


def extract_name(text: str) -> str | None:
    """
    Extract the candidate name.

    Uses the first non-empty line of the CV.
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if lines:
        return lines[0]

    return None


def extract_profile(text: str) -> dict:
    """
    Extract structured profile information from CV text.
    """

    profile = {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": [],
        "education": [],
        "experience": [],
        "languages": [],
    }

    lower_text = text.lower()

    # =================================================
    # SKILLS
    # =================================================

    possible_skills = [
        "python",
        "javascript",
        "typescript",
        "html",
        "css",
        "seo",
        "microsoft office",
        "word",
        "excel",
        "powerpoint",
        "data analysis",
        "software engineering",
        "blog writer",
        "article writer",
        "teamwork",
    ]

    for skill in possible_skills:
        if skill in lower_text:
            profile["skills"].append(skill)

    # =================================================
    # LANGUAGES
    # =================================================

    possible_languages = [
        "english",
        "urdu",
        "punjabi",
    ]

    for language in possible_languages:
        if language in lower_text:
            profile["languages"].append(language)

    # =================================================
    # EDUCATION
    # =================================================

    education_keywords = [
        "matriculation",
        "intermediate",
        "f.s.c",
        "fsc",
        "bachelor",
        "software engineering",
        "computer science",
        "university",
        "college",
    ]

    for keyword in education_keywords:
        if keyword in lower_text:
            profile["education"].append(keyword)

    # =================================================
    # EXPERIENCE
    # =================================================

    experience_keywords = [
        "fiverr",
        "upwork",
        "software house",
        "pharmacy",
        "salesman",
        "blog writer",
        "seo master",
        "computer operator",
        "freelancer",
    ]

    for keyword in experience_keywords:
        if keyword in lower_text:
            profile["experience"].append(keyword)

    return profile